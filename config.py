"""
config.py — Variáveis, parâmetros e metadados do pipeline de dados.

Centralizar configurações aqui evita hardcoding espalhado pelo código:
altere um valor aqui e todos os scripts herdam automaticamente.
"""

import os

# ============================================================
# CAMINHOS DE DADOS
# ============================================================
# Raiz do projeto: pasta onde este config.py está localizado.
# Serve de default portável — funciona em Windows, Linux e Mac sem edição.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# DATA_DIR aponta para onde ficam os CSVs (raw_data/) e as saídas geradas.
# Default: a pasta /Dados do projeto (layout do PDF). Sobrescreva com a variável
# de ambiente DATA_DIR se os dados estiverem em outro lugar (ex.: volume Docker).
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "Dados"))

RAW_DATA = {
    "application_train": os.path.join(DATA_DIR, "raw_data", "application_train.csv"),
    "application_test":  os.path.join(DATA_DIR, "raw_data", "application_test.csv"),
    "bureau":            os.path.join(DATA_DIR, "raw_data", "bureau.csv"),
    "bureau_balance":    os.path.join(DATA_DIR, "raw_data", "bureau_balance.csv"),
    "previous_app":      os.path.join(DATA_DIR, "raw_data", "previous_application.csv"),
    "pos_cash":          os.path.join(DATA_DIR, "raw_data", "POS_CASH_balance.csv"),
    "installments":      os.path.join(DATA_DIR, "raw_data", "installments_payments.csv"),
    "credit_card":       os.path.join(DATA_DIR, "raw_data", "credit_card_balance.csv"),
}

CLEAN_DATA_PATH  = os.path.join(DATA_DIR, "clean_data.csv")
ABT_DATA_PATH    = os.path.join(DATA_DIR, "abt.csv")

# ============================================================
# PARÂMETROS DO PIPELINE
# ============================================================
# num_rows=None carrega tudo; defina um inteiro (ex: 10000) para debug rápido.
# Pode ser sobrescrito pela variável de ambiente NUM_ROWS — usado na demo do
# Airflow para a DAG rodar em ~2 min (amostra) em vez de ~15 min (base completa).
NUM_ROWS = int(os.environ["NUM_ROWS"]) if os.environ.get("NUM_ROWS") else None
NAN_AS_CATEGORY = True  # Cria coluna "_nan" para ausências em variáveis categóricas

# Valor sentinela usado no dataset original para "sem data" / "sem emprego"
SENTINEL_VALUE = 365243

# ============================================================
# METADADOS DO PROJETO
# ============================================================
PROJECT_NAME    = "Home Credit Default Risk"
TARGET_COLUMN   = "TARGET"
ID_COLUMN       = "SK_ID_CURR"
RANDOM_STATE    = 1001

# ============================================================
# SAÍDA / SUBMISSÃO
# ============================================================
SUBMISSION_PATH = os.path.join(DATA_DIR, "submission.csv")

# ============================================================
# VALIDAÇÃO CRUZADA
# ============================================================
NUM_FOLDS  = 5
STRATIFIED = True   # usa StratifiedKFold (recomendado p/ target desbalanceado)

# ============================================================
# FEATURES
# ============================================================
# Colunas que NÃO são features (IDs, target, índices auxiliares)
NON_FEATURE_COLS = [ID_COLUMN, TARGET_COLUMN, "SK_ID_BUREAU", "SK_ID_PREV", "index"]

# ============================================================
# LIGHTGBM
# ============================================================
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 24,
    "max_depth": 8,
    "min_child_samples": 100,
    "subsample": 0.87,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.15,
    "reg_lambda": 0.20,
    "n_estimators": 10000,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

EARLY_STOPPING_ROUNDS = 200
LOG_PERIOD            = 100

# ============================================================
# REGRESSÃO LOGÍSTICA (BASELINE)
# ============================================================
# Usado por baseline.py para comparar o ganho real do LightGBM.
# max_iter alto porque, com centenas de features, a convergência é mais lenta.
LOGREG_PARAMS = {
    "max_iter": 1000,
    "C": 0.1,          # regularização forte — útil com tantas features (evita overfit)
    "random_state": RANDOM_STATE,
}

BASELINE_SUBMISSION_PATH = os.path.join(DATA_DIR, "submission_baseline.csv")

# ============================================================
# ARTEFATOS DO MODELO (persistência para o serviço de predição)
# ============================================================
# Ficam versionados dentro do repositório (Model/artifacts), não em DATA_DIR,
# porque o modelo treinado + a lista de features são parte do "deploy" e
# precisam acompanhar o código no git / na imagem Docker.
# MODEL_DIR é sobrescrevível por variável de ambiente para que execuções de
# demonstração (ex.: a DAG do Airflow rodando numa amostra) gravem num diretório
# separado, em vez de sobrescrever o modelo de produção treinado na base cheia.
MODEL_DIR           = os.environ.get("MODEL_DIR", os.path.join(PROJECT_ROOT, "Model", "artifacts"))
MODEL_PATH          = os.path.join(MODEL_DIR, "lgbm_model.joblib")     # modelo treinado (final, em todo o treino)
MODEL_FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.json")   # ordem exata das features esperadas
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")   # AUC, data, nº de features, thresholds

# Hiperparâmetros otimizados pelo tune.py. Fica na raiz do projeto (não em
# /Dados) porque é um artefato de configuração versionado junto ao código,
# não um dado. train.py o lê; tune.py o escreve.
BEST_PARAMS_PATH = os.path.join(PROJECT_ROOT, "best_params.json")

# Importância de features e gráfico. Caminhos ABSOLUTOS: antes o train.py
# gravava "feature_importance.csv" relativo ao diretório atual, então o arquivo
# ia parar em lugares diferentes dependendo de onde o script era chamado
# (e os notebooks não o encontravam).
# Predições out-of-fold: para cada cliente de treino, a probabilidade prevista
# por um modelo que NÃO o viu. É a única base honesta para calcular AUC/KS/Gini —
# métricas medidas sobre os dados de treino ficariam otimistas (o modelo decorou
# aqueles clientes) e enganariam quem lê o painel.
OOF_PREDICTIONS_PATH             = os.path.join(DATA_DIR, "oof_predictions.csv")

# Amostra de clientes servida pelo painel Streamlit (a ABT inteira passa de
# 500 MB e travaria o app). Gerada pelo train.py.
DEMO_CLIENTS_PATH                = os.path.join(DATA_DIR, "demo_clients.csv")

FEATURE_IMPORTANCE_PATH          = os.path.join(PROJECT_ROOT, "feature_importance.csv")
FEATURE_IMPORTANCE_BASELINE_PATH = os.path.join(PROJECT_ROOT, "feature_importance_baseline.csv")
FEATURE_IMPORTANCE_PLOT_PATH     = os.path.join(PROJECT_ROOT, "Model", "lgbm_importances.png")

# ============================================================
# POLÍTICA DE DECISÃO DE CRÉDITO
# ============================================================
# Traduz a probabilidade de default (PD) em uma DECISÃO de negócio.
# Estes cortes são de negócio, não estatísticos — ajuste conforme o apetite
# a risco da instituição. predict.py e a API usam estes valores.
#   PD < APPROVE_BELOW            → APROVAR automaticamente
#   APPROVE_BELOW <= PD <= REJECT → ANÁLISE MANUAL (zona cinzenta)
#   PD > REJECT_ABOVE             → RECUSAR automaticamente
DECISION_APPROVE_BELOW = 0.08
DECISION_REJECT_ABOVE  = 0.30

# Faixas de risco (rótulo, limite superior de PD). A última faixa cobre o resto.
RISK_BANDS = [
    ("BAIXO",     0.08),
    ("MODERADO",  0.20),
    ("ALTO",      0.40),
    ("MUITO_ALTO", 1.01),
]

# ============================================================
# CONFIGURAÇÕES: PREVIOUS APPLICATIONS
# ============================================================
PREV_APP_DATE_COLS = [
    "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION",
]

PREV_APP_NUM_AGG = {
    "AMT_ANNUITY":             ["min", "max", "mean"],
    "AMT_APPLICATION":         ["min", "max", "mean"],
    "AMT_CREDIT":              ["min", "max", "mean"],
    "APP_CREDIT_PERC":         ["min", "max", "mean", "var"],
    "AMT_DOWN_PAYMENT":        ["min", "max", "mean"],
    "AMT_GOODS_PRICE":         ["min", "max", "mean"],
    "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
    "RATE_DOWN_PAYMENT":       ["min", "max", "mean"],
    "DAYS_DECISION":           ["min", "max", "mean"],
    "CNT_PAYMENT":             ["mean", "sum"],
}

# ============================================================
# CONFIGURAÇÕES: BUREAU E BUREAU BALANCE
# ============================================================
BUREAU_SCORE_MAP = {
    "STATUS_0": 0, "STATUS_1": 1, "STATUS_2": 2,
    "STATUS_3": 3, "STATUS_4": 4, "STATUS_5": 5
}

BUREAU_BALANCE_AGG_SPEC = {
    "MONTHS_BALANCE":       ["min", "max", "size"],
    "STATUS_SCORE":         ["max", "mean"],
    "STATUS_SCORE_RECENT6": ["mean"]
}

BUREAU_NUM_AGG = {
    "DAYS_CREDIT":            ["min", "max", "mean", "var"],
    "DAYS_CREDIT_ENDDATE":    ["min", "max", "mean"],
    "DAYS_CREDIT_UPDATE":     ["mean"],
    "CREDIT_DAY_OVERDUE":     ["max", "mean"],
    "AMT_CREDIT_MAX_OVERDUE": ["mean"],
    "AMT_CREDIT_SUM":         ["max", "mean", "sum"],
    "AMT_CREDIT_SUM_DEBT":    ["max", "mean", "sum"],
    "AMT_CREDIT_SUM_OVERDUE": ["mean"],
    "AMT_CREDIT_SUM_LIMIT":   ["mean", "sum"],
    "AMT_ANNUITY":            ["max", "mean"],
    "CNT_CREDIT_PROLONG":     ["sum"],
    "MONTHS_BALANCE_MIN":     ["min"],
    "MONTHS_BALANCE_MAX":     ["max"],
    "MONTHS_BALANCE_SIZE":    ["mean", "sum"],
  # [ITEM 1] Propaga severidade/recência/tendência do nível crédito → cliente
    "STATUS_SCORE_MAX":       ["max", "mean"],
    "STATUS_SCORE_MEAN":      ["mean"],
    "STATUS_SCORE_RECENT6_MEAN": ["mean"],
    "STATUS_SCORE_TREND":     ["mean", "max"],
}

# ============================================================
# CONFIGURAÇÕES: POS CASH BALANCE
# ============================================================
POS_CASH_AGG_BASE = {
    "MONTHS_BALANCE": ["max", "mean", "size"],
    "SK_DPD":         ["max", "mean"],   # max = pior atraso; mean = comportamento médio
    "SK_DPD_DEF":     ["max", "mean"],   # versão "default" do atraso (threshold mais rígido) 
}

# ============================================================
# CONFIGURAÇÕES: INSTALLMENTS PAYMENTS
# ============================================================
INSTALLMENTS_EXPLICIT_COLS = {
    "NUM_INSTALMENT_VERSION", "DPD", "DBD", "PAYMENT_PERC",
    "PAYMENT_DIFF", "AMT_INSTALMENT", "AMT_PAYMENT", "DAYS_ENTRY_PAYMENT"
}

INSTALLMENTS_AGG_BASE = {
    "NUM_INSTALMENT_VERSION": ["nunique"], 
    "DPD":                    ["max", "mean", "sum"],
    "DBD":                    ["max", "mean", "sum"],
    "PAYMENT_PERC":           ["max", "mean", "sum", "var"],
    "PAYMENT_DIFF":           ["max", "mean", "sum", "var"],
    "AMT_INSTALMENT":         ["max", "mean", "sum"],
    "AMT_PAYMENT":            ["min", "max", "mean", "sum"],
    "DAYS_ENTRY_PAYMENT":     ["max", "mean", "sum"],
}

# ============================================================
# CONFIGURAÇÕES: CREDIT CARD BALANCE
# ============================================================
CC_AGG_FUNCS = ["min", "max", "mean", "sum", "var"]