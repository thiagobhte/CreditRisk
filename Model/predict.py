"""
predict.py — Serviço de predição (inferência) do modelo de risco de crédito.

Carrega o modelo persistido por train.py (Model/artifacts/) e expõe uma função
`predict()` reutilizável — usada tanto pela API FastAPI (app/) quanto pela CLI.

Responsabilidades:
    1. Carregar UMA vez o modelo + a lista de features + os metadados (lazy load).
    2. Alinhar qualquer entrada ao formato EXATO que o modelo espera:
       mesmos nomes de coluna, mesma ordem, colunas ausentes viram NaN
       (o LightGBM lida com NaN nativamente).
    3. Traduzir a probabilidade de default (PD) em uma DECISÃO de negócio
       (aprovar / analisar / recusar) e em uma faixa de risco.

A entrada esperada é uma linha (ou várias) no nível da ABT — ou seja, as
features já engenheiradas por abt_transform.py. A API recebe um dicionário
parcial dessas features; o que faltar é tratado como ausente.

Uso via CLI:
    python -m Model.predict --input abt.csv --output scores.csv
    python -m Model.predict --input cliente.json
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
import joblib

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MODEL_PATH, MODEL_FEATURES_PATH, MODEL_METADATA_PATH,
    DECISION_APPROVE_BELOW, DECISION_REJECT_ABOVE, RISK_BANDS,
    ID_COLUMN, TARGET_COLUMN,
)


# ============================================================
# CARREGAMENTO PREGUIÇOSO DOS ARTEFATOS (carrega 1x e reutiliza)
# ============================================================

_MODEL = None
_FEATURES = None
_METADATA = None


def _load_artifacts():
    """
    Carrega modelo, features e metadados na primeira chamada e mantém em cache.

    Na API (processo de vida longa) isso garante que o modelo — que é caro de
    desserializar — seja lido do disco uma única vez, não a cada requisição.
    """
    global _MODEL, _FEATURES, _METADATA
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modelo não encontrado em {MODEL_PATH}. "
                f"Treine primeiro com: python -m Model.train"
            )
        _MODEL = joblib.load(MODEL_PATH)
        with open(MODEL_FEATURES_PATH, "r", encoding="utf-8") as f:
            _FEATURES = json.load(f)
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            _METADATA = json.load(f)
    return _MODEL, _FEATURES, _METADATA


def model_metadata() -> dict:
    """Devolve os metadados do modelo (AUC, data, nº de features) — usado no /health."""
    _, _, meta = _load_artifacts()
    return meta


# ============================================================
# ALINHAMENTO DE FEATURES
# ============================================================

def _clean_cols(columns) -> list:
    """
    Sanitiza nomes de colunas com a MESMA regra do train.py.

    Precisa ser idêntico: se train.py transformou 'STATUS_[Approved]' em
    'STATUS__Approved__', a entrada da predição tem que sofrer a mesma
    transformação, senão a coluna não casa e vira NaN silenciosamente.
    """
    return [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in columns]


def _align(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    """
    Devolve um DataFrame com EXATAMENTE as colunas `feats`, na ordem certa.

    - Colunas presentes na entrada mas fora de `feats` são descartadas.
    - Colunas de `feats` ausentes na entrada são criadas como NaN.
    - inf/-inf (de razões com denominador 0) viram NaN.
    """
    df = df.copy()
    df.columns = _clean_cols(df.columns)

    # reindex garante ordem + presença de todas as features; ausentes = NaN
    aligned = df.reindex(columns=feats)
    aligned = aligned.replace([np.inf, -np.inf], np.nan)

    # Força tudo a numérico (uma string perdida quebraria o LightGBM)
    for c in feats:
        aligned[c] = pd.to_numeric(aligned[c], errors="coerce")

    return aligned


# ============================================================
# TRADUÇÃO PD → DECISÃO DE NEGÓCIO
# ============================================================

def _risk_band(pd_value: float) -> str:
    """Mapeia a probabilidade de default para uma faixa de risco legível."""
    for label, upper in RISK_BANDS:
        if pd_value < upper:
            return label
    return RISK_BANDS[-1][0]


def _decision(pd_value: float) -> str:
    """Aplica a política de crédito do config.py sobre a PD."""
    if pd_value < DECISION_APPROVE_BELOW:
        return "APROVAR"
    if pd_value > DECISION_REJECT_ABOVE:
        return "RECUSAR"
    return "ANALISE_MANUAL"


# ============================================================
# FUNÇÃO PRINCIPAL DE PREDIÇÃO
# ============================================================

def predict(records) -> list:
    """
    Faz a predição para um ou mais clientes.

    Args:
        records: dict (um cliente) ou lista de dicts (vários), cada um com as
                 features no nível da ABT. Pode conter também SK_ID_CURR (opcional,
                 apenas propagado para a saída) — não precisa conter todas as features.

    Returns:
        Lista de dicts, um por cliente, com:
            - SK_ID_CURR        (se fornecido)
            - probability_default (PD, entre 0 e 1)
            - risk_band          (BAIXO/MODERADO/ALTO/MUITO_ALTO)
            - decision           (APROVAR/ANALISE_MANUAL/RECUSAR)
    """
    model, feats, _ = _load_artifacts()

    if isinstance(records, dict):
        records = [records]
    df = pd.DataFrame(records)

    # Preserva IDs para devolver na resposta, mas não usa como feature
    ids = df[ID_COLUMN] if ID_COLUMN in df.columns else pd.Series([None] * len(df))

    X = _align(df, feats)
    proba = model.predict_proba(X)[:, 1]

    results = []
    for i, pd_value in enumerate(proba):
        pd_value = float(pd_value)
        results.append({
            ID_COLUMN:             None if ids.iloc[i] is None else _to_native(ids.iloc[i]),
            "probability_default": round(pd_value, 6),
            "risk_band":           _risk_band(pd_value),
            "decision":            _decision(pd_value),
        })
    return results


def _to_native(v):
    """Converte tipos numpy para tipos nativos Python (serializáveis em JSON)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ============================================================
# CLI: predição em lote a partir de arquivo
# ============================================================

def _run_cli():
    parser = argparse.ArgumentParser(description="Predição de risco de crédito em lote")
    parser.add_argument("--input", required=True, help="Arquivo .csv ou .json com features no nível da ABT")
    parser.add_argument("--output", default=None, help="CSV de saída (default: imprime no terminal)")
    args = parser.parse_args()

    # Lê CSV ou JSON
    if args.input.lower().endswith(".json"):
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
    else:
        df = pd.read_csv(args.input)
        # Remove o TARGET se vier junto (arquivo de teste da ABT não tem; treino tem)
        if TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])
        records = df.to_dict(orient="records")

    results = predict(records)
    out_df = pd.DataFrame(results)

    if args.output:
        out_df.to_csv(args.output, index=False)
        print(f"{len(out_df)} predições salvas em: {args.output}")
    else:
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    _run_cli()
