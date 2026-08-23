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
# SIMULAÇÃO: APLICAR ALTERAÇÕES DO ANALISTA
# ============================================================
# O analista altera poucas variáveis (renda, crédito, parcela, idade, scores
# externos). Mas várias features da ABT são DERIVADAS dessas — se mudarmos a
# renda sem recalcular a razão parcela/renda, o modelo recebe um cliente que
# não existe: renda nova com endividamento antigo. O resultado seria uma
# simulação sem sentido, e pior, silenciosa.
#
# Fórmulas idênticas às de DataPipeline/abt_transform.py — é o mesmo cálculo
# que gerou a ABT no treino, e precisa continuar sendo na simulação.

EPS = 1e-5  # mesmo epsilon do abt_transform: evita divisão por zero

# Variáveis que o analista pode alterar → features que precisam ser refeitas.
DERIVADAS_POR_VARIAVEL = {
    "AMT_INCOME_TOTAL": ["INCOME_CREDIT_PERC", "INCOME_PER_PERSON", "ANNUITY_INCOME_PERC",
                         "DEBT_INCOME_RATIO", "CC_UTILIZATION_INCOME"],
    "AMT_CREDIT":       ["INCOME_CREDIT_PERC", "PAYMENT_RATE", "CURR_PREV_CREDIT_RATIO",
                         "CREDIT_GOODS_PRICE_RATIO"],
    "AMT_ANNUITY":      ["ANNUITY_INCOME_PERC", "PAYMENT_RATE"],
    "DAYS_BIRTH":       ["DAYS_EMPLOYED_PERC"],
}


def _num(dados: dict, chave: str):
    """Lê um número do dicionário de features, tratando ausente/NaN como None."""
    valor = dados.get(chave)
    if valor is None:
        return None
    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(valor) or np.isinf(valor)) else valor


def _divide(numerador, denominador, eps: float = 0.0):
    """Divisão que devolve None quando não dá para calcular."""
    if numerador is None or denominador is None:
        return None
    divisor = denominador + eps
    if divisor == 0:
        return None
    resultado = numerador / divisor
    return None if (np.isnan(resultado) or np.isinf(resultado)) else resultado


def apply_overrides(features: dict, overrides: dict) -> dict:
    """
    Aplica as alterações do analista e recalcula as features derivadas afetadas.

    Args:
        features:  o cliente completo, como veio do banco (836 features)
        overrides: só o que o analista mudou (ex.: {"AMT_INCOME_TOTAL": 250000})

    Returns:
        Novo dicionário de features, coerente: as derivadas refletem os valores
        novos, e as que não dependem do que mudou permanecem intactas.
    """
    resultado = dict(features)
    resultado.update({k: v for k, v in overrides.items() if v is not None})

    # Só recalcula o que foi realmente afetado — recalcular tudo poderia
    # sobrescrever, com uma fórmula aproximada, valores que a ABT calculou
    # com mais informação do que temos aqui.
    afetadas = set()
    for variavel in overrides:
        afetadas.update(DERIVADAS_POR_VARIAVEL.get(variavel, []))
    if not afetadas:
        return resultado

    renda   = _num(resultado, "AMT_INCOME_TOTAL")
    credito = _num(resultado, "AMT_CREDIT")
    parcela = _num(resultado, "AMT_ANNUITY")

    formulas = {
        "INCOME_CREDIT_PERC":       lambda: _divide(renda, credito),
        "PAYMENT_RATE":             lambda: _divide(parcela, credito),
        "ANNUITY_INCOME_PERC":      lambda: _divide(parcela, renda),
        "INCOME_PER_PERSON":        lambda: _divide(renda, _num(resultado, "CNT_FAM_MEMBERS")),
        "DAYS_EMPLOYED_PERC":       lambda: _divide(_num(resultado, "DAYS_EMPLOYED"),
                                                    _num(resultado, "DAYS_BIRTH")),
        "DEBT_INCOME_RATIO":        lambda: _divide(_num(resultado, "BURO_AMT_CREDIT_SUM_DEBT_SUM"),
                                                    renda, EPS),
        "CURR_PREV_CREDIT_RATIO":   lambda: _divide(credito,
                                                    _num(resultado, "PREV_AMT_CREDIT_MEAN"), EPS),
        "CREDIT_GOODS_PRICE_RATIO": lambda: _divide(credito,
                                                    _num(resultado, "AMT_GOODS_PRICE"), EPS),
        "CC_UTILIZATION_INCOME":    lambda: _divide(_num(resultado, "CC_AMT_BALANCE_MEAN"),
                                                    renda, EPS),
    }

    for feature in afetadas:
        calculo = formulas.get(feature)
        if calculo is None:
            continue
        novo_valor = calculo()
        # Só grava se deu para calcular. Uma feature que já era ausente para
        # este cliente (ex.: nunca teve cartão) deve continuar ausente.
        if novo_valor is not None:
            resultado[feature] = novo_valor

    return resultado


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

    # Quantas features o modelo esperava e não recebeu. Antes isso passava em
    # silêncio: uma requisição com 3 features era pontuada como se as outras 833
    # fossem "desconhecidas", e devolvia uma PD com cara de confiável. Agora a
    # resposta carrega o número, e quem consome decide se confia.
    ausentes = X.isna().sum(axis=1).tolist()

    results = []
    for i, pd_value in enumerate(proba):
        pd_value = float(pd_value)
        results.append({
            ID_COLUMN:             None if ids.iloc[i] is None else _to_native(ids.iloc[i]),
            "probability_default": round(pd_value, 6),
            "risk_band":           _risk_band(pd_value),
            "decision":            _decision(pd_value),
            "n_features_expected": len(feats),
            "n_features_missing":  int(ausentes[i]),
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
