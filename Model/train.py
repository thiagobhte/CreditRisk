"""
train.py — Treinamento do modelo LightGBM com K-Fold Cross Validation.

Por que K-Fold?
    Usa TODOS os dados de treino para treinar e validar (em folds diferentes),
    dando uma estimativa mais robusta da performance real.
    A média das predições dos N folds também reduz variância.

AirFlow via PythonOperator:
    task = PythonOperator(task_id="train", python_callable=run)
"""

import gc
import re

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

import json
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    ABT_DATA_PATH, SUBMISSION_PATH,
    NUM_FOLDS, STRATIFIED, RANDOM_STATE,
    NON_FEATURE_COLS, LGBM_PARAMS,
    EARLY_STOPPING_ROUNDS, LOG_PERIOD,
    MODEL_DIR, MODEL_PATH, MODEL_FEATURES_PATH, MODEL_METADATA_PATH,
    DECISION_APPROVE_BELOW, DECISION_REJECT_ABOVE, PROJECT_NAME,
    BEST_PARAMS_PATH, FEATURE_IMPORTANCE_PATH, FEATURE_IMPORTANCE_PLOT_PATH,
    OOF_PREDICTIONS_PATH, DEMO_CLIENTS_PATH,
)

# Caminho do JSON gerado pelo tune.py (na raiz do projeto)
BEST_PARAMS_JSON = BEST_PARAMS_PATH


# ============================================================
# CARREGAMENTO DE PARÂMETROS (JSON → config.py)
# ============================================================

def load_lgbm_params() -> dict:
    """
    Carrega os parâmetros do LightGBM com a seguinte prioridade:

    1. best_params.json (gerado pelo tune.py) — se existir e for válido
    2. LGBM_PARAMS do config.py              — fallback padrão

    Por que JSON em vez de importar direto do tune.py?
        Desacopla a descoberta de parâmetros (Optuna) do treino (LightGBM).
        O train.py não precisa saber nada sobre Optuna — só lê um arquivo.
        Para forçar o uso do config.py, basta deletar o JSON.
    """
    if os.path.exists(BEST_PARAMS_JSON):
        try:
            with open(BEST_PARAMS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)

            params = data.get("params", {})
            meta   = data.get("meta", {})

            if not params:
                raise ValueError("Chave 'params' ausente ou vazia no JSON.")

            print(f"[train] Parâmetros carregados de: {BEST_PARAMS_JSON}")
            print(f"        AUC estimado (1 fold): {meta.get('best_auc_1fold', '?')}")
            print(f"        Gerado em:             {meta.get('tuned_at', '?')}")
            return params

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[train] Aviso: falha ao ler {BEST_PARAMS_JSON} ({e}).")
            print(f"        Usando LGBM_PARAMS do config.py como fallback.")

    else:
        print(f"[train] {BEST_PARAMS_JSON} não encontrado.")
        print(f"        Usando LGBM_PARAMS do config.py.")

    return LGBM_PARAMS


# ============================================================
# TREINAMENTO COM K-FOLD CROSS VALIDATION
# ============================================================

def kfold_lightgbm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treina o LightGBM com K-Fold CV e retorna a importância de features.

    Parâmetros carregados via load_lgbm_params():
      → best_params.json (Optuna) se existir
      → LGBM_PARAMS do config.py como fallback

    Retorna DataFrame com colunas: feature, importance, fold.
    """
    params = load_lgbm_params()
    # Separa treino (TARGET preenchido) de teste (TARGET = NaN)
    train_df = df[df["TARGET"].notnull()].copy()
    test_df  = df[df["TARGET"].isnull()].copy()

    # Sanitiza nomes de colunas: LightGBM não aceita [ ] { } nos nomes
    # (gerados pelo pd.get_dummies — ex: 'NAME_CONTRACT_STATUS_[Approved]')
    def clean_cols(frame):
        frame.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in frame.columns]
        return frame

    train_df = clean_cols(train_df)
    test_df  = clean_cols(test_df)

    print(f"Treino: {train_df.shape} | Teste: {test_df.shape}")
    del df; gc.collect()

    # Seleciona features (exclui IDs e target)
    feats = [c for c in train_df.columns if c not in NON_FEATURE_COLS]

    # Escolha da estratégia de fold
    if STRATIFIED:
        # StratifiedKFold: mantém proporção do TARGET em cada fold
        # Recomendado quando o dataset é muito desbalanceado
        folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    else:
        # KFold padrão: divisão aleatória sem considerar proporção do TARGET
        folds = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Arrays de predição
    # oof_preds: predição de cada amostra quando estava no fold de validação
    # (estimativa honesta sem data leakage)
    oof_preds = np.zeros(train_df.shape[0])

    # sub_preds: média das predições do teste nos N folds (ensemble)
    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()

    # Guarda o nº de árvores em que cada fold parou (early stopping) — usado
    # para dimensionar o n_estimators do modelo FINAL treinado em todo o dataset.
    best_iters = []

    # ---- Loop de validação cruzada ----
    for fold_n, (train_idx, valid_idx) in enumerate(
        folds.split(train_df[feats], train_df["TARGET"])
    ):
        train_x = train_df[feats].iloc[train_idx]
        train_y = train_df["TARGET"].iloc[train_idx]
        valid_x = train_df[feats].iloc[valid_idx]
        valid_y = train_df["TARGET"].iloc[valid_idx]

        clf = LGBMClassifier(**params)
        clf.fit(
            train_x, train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric="auc",
            callbacks=[
                early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                # Para se AUC não melhorar em N rodadas — evita overfitting
                log_evaluation(period=LOG_PERIOD),
                # Imprime métricas a cada LOG_PERIOD árvores
            ],
        )

        # Predição OOF: usa o melhor checkpoint (early stopping), não o último
        oof_preds[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_
        )[:, 1]

        # Acumula predições de teste dividindo pelo número de folds (para a média)
        sub_preds += clf.predict_proba(
            test_df[feats], num_iteration=clf.best_iteration_
        )[:, 1] / folds.n_splits

        # Registra importância das features neste fold
        fold_imp = pd.DataFrame({
            "feature":    feats,
            "importance": clf.feature_importances_,
            "fold":       fold_n + 1,
        })
        feature_importance_df = pd.concat([feature_importance_df, fold_imp], axis=0)

        best_iters.append(clf.best_iteration_ or params.get("n_estimators", 1000))

        fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])
        print(f"Fold {fold_n + 1:2d} | AUC: {fold_auc:.6f} | best iter: {clf.best_iteration_}")

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    # AUC final: concatena todas as predições OOF — estimativa mais honesta
    full_auc = roc_auc_score(train_df["TARGET"], oof_preds)
    print(f"\nAUC total (OOF): {full_auc:.6f}")

    # Salva as predições out-of-fold junto do rótulo real.
    # É o único par (predição, verdade) sem vazamento que temos: qualquer
    # métrica calculada re-prevendo o treino sairia otimista. O painel e os
    # notebooks usam este arquivo para AUC, KS e Gini.
    pd.DataFrame({
        "SK_ID_CURR": train_df["SK_ID_CURR"].to_numpy(),
        "TARGET":     train_df["TARGET"].to_numpy(),
        "PD":         oof_preds,
    }).to_csv(OOF_PREDICTIONS_PATH, index=False)
    print(f"Predicoes out-of-fold salvas em: {OOF_PREDICTIONS_PATH}")

    # Salva submissão
    test_df["TARGET"] = sub_preds
    test_df[["SK_ID_CURR", "TARGET"]].to_csv(SUBMISSION_PATH, index=False)
    print(f"Submissão salva em: {SUBMISSION_PATH}")

    # Treina e persiste o modelo FINAL (para o serviço de predição)
    persist_final_model(train_df, feats, params, best_iters, full_auc)

    # Amostra de clientes usada pelo painel Streamlit
    save_demo_clients(train_df)

    return feature_importance_df


# ============================================================
# TREINO E PERSISTÊNCIA DO MODELO FINAL
# ============================================================

def persist_final_model(train_df: pd.DataFrame, feats: list, params: dict,
                        best_iters: list, oof_auc: float) -> None:
    """
    Treina um único modelo em TODO o conjunto de treino e o salva em disco.

    Por que um modelo separado do K-Fold?
        Os modelos do K-Fold servem para ESTIMAR a performance (OOF AUC). Para
        SERVIR em produção queremos um único modelo treinado com 100% dos dados
        disponíveis — mais estável que escolher arbitrariamente um dos folds.

    n_estimators do modelo final:
        Usa a MÉDIA das melhores iterações dos folds (onde o early stopping parou).
        Como agora treinamos com mais dados (sem separar validação), damos uma
        pequena margem de +5% para compensar.

    Salva 3 artefatos em Model/artifacts/:
        - lgbm_model.joblib   : o modelo treinado
        - model_features.json : a ordem EXATA das features (o predict.py precisa alinhar)
        - model_metadata.json : AUC, data, nº de features e thresholds de decisão
    """
    final_n_estimators = int(np.mean(best_iters) * 1.05)
    print(f"\n=== Treinando modelo final em todo o treino "
          f"({train_df.shape[0]} linhas, n_estimators={final_n_estimators}) ===")

    final_params = {**params, "n_estimators": final_n_estimators}

    final_clf = LGBMClassifier(**final_params)
    final_clf.fit(train_df[feats], train_df["TARGET"])

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Modelo treinado
    joblib.dump(final_clf, MODEL_PATH)

    # 2. Lista de features na ordem esperada — o predict.py reordena/completa
    #    qualquer entrada para casar exatamente com esta lista.
    with open(MODEL_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)

    # 3. Metadados: rastreabilidade + política de decisão versionada junto ao modelo
    metadata = {
        "project":           PROJECT_NAME,
        "model_type":        "LGBMClassifier",
        "trained_at":        datetime.now().isoformat(timespec="seconds"),
        "oof_auc":           round(float(oof_auc), 6),
        "n_features":        len(feats),
        "n_estimators":      final_n_estimators,
        "params":            final_params,
        "decision_policy": {
            "approve_below": DECISION_APPROVE_BELOW,
            "reject_above":  DECISION_REJECT_ABOVE,
        },
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Modelo salvo em:    {MODEL_PATH}")
    print(f"Features salvas em: {MODEL_FEATURES_PATH}")
    print(f"Metadados em:       {MODEL_METADATA_PATH}")

    del final_clf
    gc.collect()


def save_demo_clients(train_df: pd.DataFrame, n_good: int = 1200, n_bad: int = 400) -> None:
    """
    Salva uma amostra de clientes para o painel Streamlit.

    Por que não deixar o painel ler a ABT direto: ela passa de 500 MB e o app
    travaria ao abrir. Esta amostra é pequena e propositalmente enriquecida em
    inadimplentes (25% contra os ~8% reais), para que a demonstração encontre
    casos de risco sem precisar caçar cliente por cliente.

    Cuidado ao usar: por conterem clientes do treino, estes registros NÃO servem
    para medir performance — quem faz isso é o oof_predictions.csv.
    """
    good = train_df[train_df["TARGET"] == 0]
    bad  = train_df[train_df["TARGET"] == 1]

    amostra = pd.concat([
        good.sample(n=min(n_good, len(good)), random_state=RANDOM_STATE),
        bad.sample(n=min(n_bad, len(bad)),   random_state=RANDOM_STATE),
    ]).sample(frac=1, random_state=RANDOM_STATE)

    amostra.to_csv(DEMO_CLIENTS_PATH, index=False)
    print(f"Clientes de demonstracao salvos em: {DEMO_CLIENTS_PATH} ({len(amostra)} linhas)")


# ============================================================
# PONTO DE ENTRADA (VS Code e AirFlow)
# ============================================================

def plot_feature_importance(feat_importance: pd.DataFrame, top_n: int = 40) -> None:
    """
    Gera o gráfico das features mais importantes (média entre os folds).

    Salva em Model/lgbm_importances.png — usado na apresentação e no
    evaluation.ipynb como evidência de explicabilidade do modelo.
    """
    import matplotlib
    matplotlib.use("Agg")   # backend sem GUI: funciona em servidor/container
    import matplotlib.pyplot as plt

    top = (feat_importance.groupby("feature")["importance"]
           .mean()
           .sort_values(ascending=False)
           .head(top_n)
           .iloc[::-1])   # inverte para a maior ficar no topo do barh

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.25)))
    ax.barh(top.index, top.values, color="steelblue")
    ax.set_title(f"LightGBM — Top {top_n} features (média dos folds)")
    ax.set_xlabel("Importância (ganho)")
    plt.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=120)
    plt.close(fig)
    print(f"Grafico de importancia salvo em: {FEATURE_IMPORTANCE_PLOT_PATH}")


def run():
    """
    Lê a ABT, treina o modelo e salva a submissão.
    Chamável diretamente (python train.py) ou via AirFlow PythonOperator.
    """
    print("=== Iniciando treinamento ===")
    df = pd.read_csv(ABT_DATA_PATH)

    # Rede de segurança: se alguma coluna de texto chegou até aqui, o encoding
    # falhou lá atrás (ver one_hot_encoder). Avisa em vez de convertê-la em NaN
    # silenciosamente — foi exatamente assim que as categóricas sumiram antes.
    # "object" + "string": funciona em pandas 2 e 3 ("str" quebra no pandas 2).
    leftover_text = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if leftover_text:
        print(f"AVISO: {len(leftover_text)} colunas de texto chegaram na ABT "
              f"(encoding falhou?): {leftover_text[:5]}")
        for col in leftover_text:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feat_importance = kfold_lightgbm(df)

    # Salva importância de features para uso no evaluation.ipynb
    feat_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"Importancia de features salva em: {FEATURE_IMPORTANCE_PATH}")

    plot_feature_importance(feat_importance)


if __name__ == "__main__":
    run()