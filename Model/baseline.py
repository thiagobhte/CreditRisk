"""
baseline.py — Treinamento de uma Regressão Logística com K-Fold Cross Validation.

Por que esse script existe:
    Serve como baseline de comparação para o LightGBM (train.py). Se o LightGBM
    não superar uma regressão logística simples por uma margem relevante, é sinal
    de que vale revisar as features antes de investir em tuning do modelo principal.

Diferença-chave em relação ao train.py:
    Regressão Logística NÃO lida nativamente com NaN nem com features em escalas
    muito diferentes (LightGBM lida com ambos). Por isso este script adiciona duas
    etapas extras que o train.py não precisa: imputação e padronização (scaling).

Compatível com AirFlow via PythonOperator:
    task = PythonOperator(task_id="baseline", python_callable=run)
"""

import gc
import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    ABT_DATA_PATH, BASELINE_SUBMISSION_PATH,
    NUM_FOLDS, STRATIFIED, RANDOM_STATE,
    NON_FEATURE_COLS, LOGREG_PARAMS,
    FEATURE_IMPORTANCE_BASELINE_PATH,
)


# ============================================================
# TREINAMENTO COM K-FOLD CROSS VALIDATION
# ============================================================

def kfold_logistic_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treina a Regressão Logística com K-Fold CV e retorna a importância de features.

    Estrutura idêntica ao kfold_lightgbm() do train.py — usa o mesmo esquema
    de OOF (Out-of-Fold) predictions para uma comparação justa de AUC.

    Retorna DataFrame com colunas: feature, importance, fold.
    """
    # Separa treino (TARGET preenchido) de teste (TARGET = NaN)
    train_df = df[df["TARGET"].notnull()].copy()
    test_df  = df[df["TARGET"].isnull()].copy()

    # Sanitiza nomes de colunas — mesmo motivo do train.py, por consistência
    def clean_cols(frame):
        frame.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in frame.columns]
        return frame

    train_df = clean_cols(train_df)
    test_df  = clean_cols(test_df)

    print(f"Treino: {train_df.shape} | Teste: {test_df.shape}")
    del df; gc.collect()

    # Seleciona features (exclui IDs e target)
    feats = [c for c in train_df.columns if c not in NON_FEATURE_COLS]

    # ---- Garante que só colunas numéricas entrem no pipeline ----
    # Em tese, todo o encoding já foi feito no data_sanitization.py — mas se a
    # ABT foi gerada a partir de um clean_data.csv antigo, ou se alguma coluna
    # de texto escapou do one-hot, ela chega aqui ainda como string.
    # LightGBM ignoraria/erraria de forma menos silenciosa; a Regressão Logística
    # e o SimpleImputer não aceitam 'object' de forma alguma.
    non_numeric = train_df[feats].select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        print(f"Aviso: {len(non_numeric)} colunas não numéricas encontradas e removidas: {non_numeric[:5]}{'...' if len(non_numeric) > 5 else ''}")
        feats = [c for c in feats if c not in non_numeric]

    # ---- Pré-processamento exclusivo da Regressão Logística ----
    # Algumas features derivadas no abt_transform.py são razões (ex: AMT_CREDIT /
    # AMT_ANNUITY). Quando o denominador é 0, o resultado é +-inf, não NaN.
    # SimpleImputer só trata NaN — precisamos converter inf para NaN primeiro,
    # senão o fit_transform falha com "Input X contains infinity".
    train_df[feats] = train_df[feats].replace([np.inf, -np.inf], np.nan)
    test_df[feats]  = test_df[feats].replace([np.inf, -np.inf], np.nan)

    # Remove colunas 100% vazias (todas NaN) — o SimpleImputer as descartaria
    # silenciosamente e devolveria um array com menos colunas do que `feats`,
    # quebrando o pd.DataFrame(..., columns=feats) com um shape mismatch.
    all_nan_cols = train_df[feats].columns[train_df[feats].isna().all()].tolist()
    if all_nan_cols:
        print(f"Aviso: {len(all_nan_cols)} colunas 100% vazias removidas: {all_nan_cols[:5]}{'...' if len(all_nan_cols) > 5 else ''}")
        feats = [c for c in feats if c not in all_nan_cols]

    # LightGBM lida com NaN nativamente; a Regressão Logística não.
    # Imputação pela mediana: robusta a outliers, mais segura que a média
    # quando há valores muito distantes (comum em features financeiras).
    imputer = SimpleImputer(strategy="median")
    train_x_all = pd.DataFrame(
        imputer.fit_transform(train_df[feats]), columns=feats, index=train_df.index
    )
    test_x_all = pd.DataFrame(
        imputer.transform(test_df[feats]), columns=feats, index=test_df.index
    )

    # Padronização: Regressão Logística é sensível à escala das variáveis
    # (AMT_CREDIT na casa dos milhões vs DAYS_BIRTH na casa dos milhares).
    # Sem isso, features com valores maiores dominam o coeficiente artificialmente.
    scaler = StandardScaler()
    train_x_all = pd.DataFrame(
        scaler.fit_transform(train_x_all), columns=feats, index=train_df.index
    )
    test_x_all = pd.DataFrame(
        scaler.transform(test_x_all), columns=feats, index=test_df.index
    )

    # Escolha da estratégia de fold — mesma lógica do train.py
    if STRATIFIED:
        folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    else:
        folds = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    # ---- Loop de validação cruzada ----
    for fold_n, (train_idx, valid_idx) in enumerate(
        folds.split(train_x_all, train_df["TARGET"])
    ):
        train_x = train_x_all.iloc[train_idx]
        train_y = train_df["TARGET"].iloc[train_idx]
        valid_x = train_x_all.iloc[valid_idx]
        valid_y = train_df["TARGET"].iloc[valid_idx]

        clf = LogisticRegression(**LOGREG_PARAMS)
        clf.fit(train_x, train_y)

        # Predição OOF — mesma lógica do train.py
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]

        # Acumula predições de teste (média entre os folds)
        sub_preds += clf.predict_proba(test_x_all)[:, 1] / folds.n_splits

        # "Importância" na Regressão Logística é o valor absoluto do coeficiente:
        # quanto maior |coef|, mais a feature move a predição (após padronização,
        # os coeficientes já estão na mesma escala e são comparáveis entre si).
        fold_imp = pd.DataFrame({
            "feature":    feats,
            "importance": np.abs(clf.coef_[0]),
            "fold":       fold_n + 1,
        })
        feature_importance_df = pd.concat([feature_importance_df, fold_imp], axis=0)

        fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])
        print(f"Fold {fold_n + 1:2d} | AUC: {fold_auc:.6f}")

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    # AUC final — comparável diretamente com o "AUC total (OOF)" do train.py
    full_auc = roc_auc_score(train_df["TARGET"], oof_preds)
    print(f"\nAUC total (OOF) — baseline: {full_auc:.6f}")

    # Salva submissão em arquivo separado para não sobrescrever a do LightGBM
    test_df["TARGET"] = sub_preds
    test_df[["SK_ID_CURR", "TARGET"]].to_csv(BASELINE_SUBMISSION_PATH, index=False)
    print(f"Submissão do baseline salva em: {BASELINE_SUBMISSION_PATH}")

    return feature_importance_df


# ============================================================
# PONTO DE ENTRADA (VS Code e AirFlow)
# ============================================================

def run():
    """
    Lê a ABT, treina o baseline (Regressão Logística) e salva a submissão.
    Chamável diretamente (python baseline.py) ou via AirFlow PythonOperator.

    Compare o "AUC total (OOF)" impresso aqui com o do train.py:
    a diferença entre os dois é o ganho real que o LightGBM está trazendo.
    """
    print("=== Iniciando treinamento do baseline (Regressão Logística) ===")
    df = pd.read_csv(ABT_DATA_PATH)

    # Converte colunas object/string remanescentes (mesma segurança do train.py)
    # Pandas 3 distingue "object" de "str" — incluímos os dois.
    # "object" + "string": funciona em pandas 2 e 3 ("str" quebra no pandas 2).
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    feat_importance = kfold_logistic_regression(df)

    # Salva em arquivo separado do feature_importance.csv do LightGBM
    feat_importance.to_csv(FEATURE_IMPORTANCE_BASELINE_PATH, index=False)
    print(f"Importancia de features salva em: {FEATURE_IMPORTANCE_BASELINE_PATH}")


if __name__ == "__main__":
    run()