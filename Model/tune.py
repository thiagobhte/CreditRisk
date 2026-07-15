"""
tune.py — Busca de hiperparâmetros com Optuna para o LightGBM.

Como usar:
    python Model/tune.py                 # roda 50 trials (padrão)
    python Model/tune.py --trials 100    # roda 100 trials
    python Model/tune.py --trials 30 --timeout 3600  # para em 30 trials ou 1h

Resultado: atualiza LGBM_PARAMS no config.py com os melhores parâmetros
encontrados e imprime os resultados para revisão.

Estratégia de avaliação rápida (1 fold fixo):
    Rodar o K-Fold completo (5 folds × 10000 árvores) para cada trial
    levaria horas. Em vez disso, cada trial treina em 1 fold com
    n_estimators fixo e early_stopping — rápido o suficiente para
    explorar centenas de combinações.

    O AUC do Optuna é uma estimativa, não o número final. Após encontrar
    os melhores parâmetros, rode train.py para obter o AUC real nos 5 folds.

Compatível com AirFlow via PythonOperator:
    task = PythonOperator(task_id="tune", python_callable=run)
"""

import argparse
import gc
import re
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    ABT_DATA_PATH, NON_FEATURE_COLS,
    RANDOM_STATE, EARLY_STOPPING_ROUNDS, BEST_PARAMS_PATH,
)

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError(
        "Optuna não está instalado. Execute: pip install optuna"
    )

# Suprime logs repetitivos do LightGBM durante a busca
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ============================================================

def load_train_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Lê a ABT, converte textos remanescentes e retorna
    (X_train, y_train, lista_de_features).

    Retorna apenas o conjunto de treino (TARGET preenchido) —
    o conjunto de teste não é necessário durante a busca.
    """
    print("Carregando ABT...")
    df = pd.read_csv(ABT_DATA_PATH)

    # Converte strings remanescentes (pandas 3 pode ter dtype "str" ou "object")
    # "object" + "string": funciona em pandas 2 e 3 ("str" quebra no pandas 2).
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sanitiza nomes (LightGBM não aceita [ ] { } em nomes de features)
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in df.columns]

    train_df = df[df["TARGET"].notnull()].copy()
    del df; gc.collect()

    feats = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    X = train_df[feats]
    y = train_df["TARGET"]

    print(f"Treino carregado: {X.shape[0]} linhas × {X.shape[1]} features")
    return X, y, feats


# ============================================================
# FUNÇÃO OBJETIVO DO OPTUNA
# ============================================================

def make_objective(X: pd.DataFrame, y: pd.Series):
    """
    Retorna a função objetivo que o Optuna vai chamar em cada trial.

    Usa 1 fold fixo (o fold 0 do StratifiedKFold) — leva ~30-120s por trial
    dependendo do hardware. Com 50 trials, o tuning completo leva ~30-90 min.

    Por que 1 fold e não 5?
        Em busca de hiperparâmetros, variância entre trials importa menos que
        velocidade de exploração. Usar 1 fold reduz o tempo por trial em 5×,
        permitindo explorar muito mais o espaço nos mesmos recursos de tempo.
        O AUC resultante é uma estimativa — o número final vem do train.py.
    """
    # Prepara o fold uma única vez, fora do objetivo, para consistência entre trials
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, valid_idx = next(iter(cv.split(X, y)))

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    def objective(trial: optuna.Trial) -> float:
        """
        Sugere parâmetros, treina o LightGBM e devolve AUC de validação.
        O Optuna maximiza este valor.
        """
        params = {
            "objective":        "binary",
            "metric":           "auc",
            "boosting_type":    "gbdt",
            "verbosity":        -1,
            "random_state":     RANDOM_STATE,
            "n_jobs":           -1,

            # ---- Parâmetros sob busca ----

            # Complexidade da árvore: num_leaves é o principal controlador.
            # Valores entre 16 e 64 cobrem de muito simples a moderadamente complexo.
            # max_depth limita a profundidade máxima (compatível com num_leaves).
            "num_leaves":       trial.suggest_int("num_leaves", 16, 64),
            "max_depth":        trial.suggest_int("max_depth", 4, 10),

            # Amostras mínimas por folha: evita splits com poucos dados.
            # Valores mais altos regularizam mais fortemente.
            "min_child_samples": trial.suggest_int("min_child_samples", 40, 150),

            # Subsampling de linhas e colunas por árvore.
            # Reduzir aumenta a regularização e diversidade das árvores.
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

            # Regularização L1 (penaliza features irrelevantes → coef zero) e
            # L2 (penaliza pesos grandes → suaviza predições).
            # Ranges log-uniformes: cobrem várias ordens de magnitude eficientemente.
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),

            # Learning rate mais baixo = mais árvores necessárias, mas melhor
            # generalização. 0.01-0.05 é um range razoável para early stopping.
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.05, log=True),

            # n_estimators alto com early stopping: o modelo para sozinho
            # no ponto ótimo — o valor aqui é um teto, não um alvo.
            "n_estimators":     3000,
        }

        clf = LGBMClassifier(**params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                log_evaluation(period=-1),   # silencia logs por trial
            ],
        )

        preds = clf.predict_proba(X_valid, num_iteration=clf.best_iteration_)[:, 1]
        auc   = roc_auc_score(y_valid, preds)

        # Registra o número de árvores usadas (útil para configurar n_estimators
        # no config.py depois — usar best_iteration * 1.1 como margem de segurança)
        trial.set_user_attr("best_iteration", clf.best_iteration_)

        del clf; gc.collect()
        return auc

    return objective


# ============================================================
# ATUALIZAÇÃO DO CONFIG.PY
# ============================================================

def save_best_params(best_params: dict, best_auc: float, best_iter: int):
    """
    Salva os melhores parâmetros em best_params.json, dentro da pasta Model/.

    Estrutura do JSON:
    {
        "meta": {
            "best_auc_1fold": 0.791234,   ← AUC estimado (1 fold, não oficial)
            "best_iteration":  1420,       ← árvores usadas no trial vencedor
            "n_estimators":    1562,       ← best_iteration × 1.1, arredondado
            "tuned_at":        "2024-..."  ← timestamp para rastreabilidade
        },
        "params": {                        ← bloco pronto para o LightGBM
            "objective":       "binary",
            "num_leaves":      34,
            ...
        }
    }

    O train.py lê este arquivo automaticamente quando ele existe.
    Para forçar o train.py a usar o config.py, basta deletar o JSON.
    """
    import json
    from datetime import datetime

    safe_n_estimators = max(int(best_iter * 1.1), 500)

    # Parâmetros fixos que não variam entre trials (não são sugeridos pelo Optuna)
    fixed = {
        "objective":      "binary",
        "metric":         "auc",
        "boosting_type":  "gbdt",
        "n_estimators":   safe_n_estimators,
        "n_jobs":         -1,
        "random_state":   RANDOM_STATE,
        "verbose":        -1,
    }

    payload = {
        "meta": {
            "best_auc_1fold":  round(best_auc, 6),
            "best_iteration":  best_iter,
            "n_estimators":    safe_n_estimators,
            "tuned_at":        datetime.now().isoformat(timespec="seconds"),
            "note": (
                "AUC estimado em 1 fold — rode train.py para obter o AUC "
                "real nos 5 folds com estes parâmetros."
            ),
        },
        # Merge: parâmetros Optuna + fixos (Optuna pode sugerir learning_rate,
        # os fixos completam o restante que não está sob busca)
        "params": {**fixed, **best_params},
    }

    output_path = BEST_PARAMS_PATH
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print(f"Melhor AUC (1 fold): {best_auc:.6f}")
    print(f"Melhor iteracao:     {best_iter} -> n_estimators: {safe_n_estimators}")
    print(f"\nParâmetros salvos em: {output_path}")
    print("O train.py vai ler este JSON automaticamente na próxima execução.")
    print("Para usar o config.py no lugar, delete o JSON.")
    print("="*60)


# ============================================================
# PONTO DE ENTRADA (VS Code e AirFlow)
# ============================================================

def run(n_trials: int = 50, timeout: int = None):
    """
    Executa a busca de hiperparâmetros com Optuna.

    Args:
        n_trials:  número de combinações de parâmetros a testar.
                   50 é um bom equilíbrio entre qualidade e tempo.
                   Use 20-30 para exploração inicial, 100+ para resultado final.
        timeout:   tempo máximo em segundos (None = sem limite de tempo).
                   Optuna para no que vier primeiro: n_trials ou timeout.
    """
    X, y, feats = load_train_data()

    # TPESampler: Tree-structured Parzen Estimator — aprende com trials anteriores
    # para sugerir parâmetros mais promissores progressivamente.
    # MedianPruner: interrompe trials que claramente estão pior que a mediana
    # dos trials anteriores no mesmo ponto de treinamento (economiza tempo).
    study = optuna.create_study(
        direction="maximize",          # queremos maximizar AUC
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(
            n_startup_trials=5,        # avalia os primeiros 5 trials completamente
            n_warmup_steps=100,        # não prune antes de 100 árvores
        ),
        study_name="lgbm_home_credit",
    )

    objective = make_objective(X, y)

    print(f"\nIniciando busca Optuna — {n_trials} trials")
    print(f"Cada trial usa 1 fold para avaliação rápida.\n")

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,   # barra de progresso no terminal
        callbacks=[
            # Imprime o melhor AUC encontrado a cada novo record.
            # Só ASCII aqui: o console do Windows usa cp1252 e estoura
            # UnicodeEncodeError em caracteres como "←".
            lambda study, trial: print(
                f"  Trial {trial.number:3d} | AUC: {trial.value:.6f}"
                + (" <-- novo melhor!" if trial.value == study.best_value else "")
            ) if trial.value is not None else None
        ],
    )

    best = study.best_trial
    best_iter = best.user_attrs.get("best_iteration", 1500)

    # Remove parâmetros internos antes de salvar (n_estimators é definido por nós)
    best_params = {k: v for k, v in best.params.items() if k != "n_estimators"}
    save_best_params(best_params, best.value, best_iter)

    # Resumo dos top-5 trials para contexto
    print("\nTop 5 trials:")
    top = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]
    for i, t in enumerate(top):
        print(f"  #{i+1} Trial {t.number:3d} | AUC: {t.value:.6f} | iter: {t.user_attrs.get('best_iteration','?')}")

    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Busca de hiperparâmetros com Optuna")
    parser.add_argument("--trials",  type=int, default=50,   help="Número de trials (padrão: 50)")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout em segundos (padrão: sem limite)")
    args = parser.parse_args()

    run(n_trials=args.trials, timeout=args.timeout)