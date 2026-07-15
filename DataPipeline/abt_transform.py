"""
abt_transform.py — Transforma os dados limpos na ABT (Analytical Base Table).

Responsabilidade: construir features derivadas (razões, agregações por cliente)
a partir de todas as tabelas secundárias e unir tudo em abt.csv.

AirFlow via PythonOperator:
    task = PythonOperator(task_id="build_abt", python_callable=run)
"""

import gc
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CLEAN_DATA_PATH, ABT_DATA_PATH, INSTALLMENTS_AGG_BASE, INSTALLMENTS_EXPLICIT_COLS, NUM_ROWS,
    ID_COLUMN, POS_CASH_AGG_BASE, TARGET_COLUMN, PREV_APP_NUM_AGG, BUREAU_NUM_AGG
)
from DataPipeline.data_sanitization import (
    load_bureau, load_previous_applications,
    load_pos_cash, load_installments, load_credit_card,
    one_hot_encoder,
)

# ============================================================
# UTILITÁRIO: TIMER DE CONTEXTO
# ============================================================

@contextmanager
def timer(title: str):
    """
    Mede e imprime o tempo de cada etapa do pipeline.
    Uso: `with timer("Processando bureau"):`
    """
    t0 = time.time()
    yield
    print(f"{title} — concluído em {time.time() - t0:.0f}s")


# ============================================================
# ENGENHARIA DE FEATURES: TABELA PRINCIPAL
# ============================================================

def build_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria razões e percentuais a partir da tabela principal.

    Razões capturam relações relativas que valores absolutos não capturam:
    ex. uma renda de R$5.000 é "muito" ou "pouco" dependendo do tamanho
    do crédito solicitado e do número de dependentes.

    Usa pd.concat para adicionar todas as colunas de uma vez, evitando o
    PerformanceWarning de fragmentação que ocorre ao inserir coluna por coluna
    em DataFrames largos (centenas de dummies já presentes).
    """
    new_cols = {
        # % do tempo de vida empregado — normaliza idade vs tempo de trabalho
        "DAYS_EMPLOYED_PERC":  df["DAYS_EMPLOYED"]    / df["DAYS_BIRTH"],

        # Renda / crédito: quanto maior, mais folgado o cliente
        "INCOME_CREDIT_PERC":  df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"],

        # Renda per capita familiar — R$5.000 para 1 pessoa ≠ para 5 pessoas
        "INCOME_PER_PERSON":   df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"],

        # Peso da parcela na renda: quanto maior, maior o risco de default
        "ANNUITY_INCOME_PERC": df["AMT_ANNUITY"]      / df["AMT_INCOME_TOTAL"],

        # Taxa de pagamento: parcela / crédito total — quão rápido o cliente quita
        "PAYMENT_RATE":        df["AMT_ANNUITY"]      / df["AMT_CREDIT"],
    }
    # Concatena todas as novas colunas de uma vez — evita fragmentação do DataFrame
    new_df = pd.DataFrame(new_cols, index=df.index)

    # Divisões por zero (ex: AMT_ANNUITY / AMT_CREDIT quando o crédito é 0)
    # geram +-inf em vez de NaN. Modelos lineares (Regressão Logística, etc.)
    # quebram com inf; o LightGBM aceita silenciosamente mas trata mal.
    # Resolve na origem para que todo consumidor da ABT receba dados limpos.
    new_df = new_df.replace([np.inf, -np.inf], np.nan)

    return pd.concat([df, new_df], axis=1)


# ============================================================
# AGREGAÇÕES: BUREAU + BUREAU_BALANCE
# ============================================================

def build_bureau_features(num_rows=None) -> pd.DataFrame:
    """
    Agrega bureau e bureau_balance por cliente (SK_ID_CURR).

    Estratégia em 3 camadas:
    1. Agrega bureau_balance (mensal) → um resumo por crédito
    2. Une com bureau e agrega → um resumo por cliente
    3. Cria features separadas para créditos Ativos vs Encerrados
       (ativos = carga atual de dívida; encerrados = histórico de comportamento)

    [ITEM 1 — Opção A] Antes da agregação de bureau_balance, reconstrói um
    STATUS_SCORE de severidade a partir das dummies STATUS_* (que já vêm do
    one_hot_encoder em data_sanitization). O STATUS bruto não existe mais aqui,
    então o score é derivado das colunas STATUS_0..STATUS_5 (C/X/nan = 0, sem
    atraso). Sobre esse score extraímos 4 sinais por crédito:
        - STATUS_SCORE_MAX    : pior atraso já registrado (severidade)
        - STATUS_SCORE_MEAN   : nível médio de atraso ao longo do histórico
        - STATUS_SCORE_RECENT6: severidade média nos 6 meses mais recentes (recência)
        - STATUS_SCORE_TREND  : inclinação (slope) do score no tempo (tendência:
                                positivo = piorando, negativo = melhorando)
    """
    bureau, bb = load_bureau(num_rows)

    # ---- 0. [ITEM 1] Reconstrói STATUS_SCORE a partir das dummies STATUS_* ----
    # Mapa de severidade: STATUS_N (N meses em atraso) → N. C (quitado), X
    # (desconhecido) e nan não representam atraso → contribuem 0.
    score_map = {"STATUS_0": 0, "STATUS_1": 1, "STATUS_2": 2,
                 "STATUS_3": 3, "STATUS_4": 4, "STATUS_5": 5}
    present = [c for c in score_map if c in bb.columns]
    if present:
        # Soma ponderada das dummies → severidade por linha (mês). Como cada linha
        # tem exatamente um STATUS ativo, a soma resulta no score daquele mês.
        bb["STATUS_SCORE"] = sum(bb[c].astype("int8") * score_map[c] for c in present)
    else:
        # Fallback defensivo: se nenhuma dummy STATUS_* existir, score neutro.
        bb["STATUS_SCORE"] = 0

    # ---- 0b. [ITEM 1] Severidade recente: média do score nos 6 meses mais
    # recentes de cada crédito. MONTHS_BALANCE é 0 (mês atual) e negativo para o
    # passado, então os 6 mais recentes são MONTHS_BALANCE >= -5.
    bb["STATUS_SCORE_RECENT6"] = bb["STATUS_SCORE"].where(bb["MONTHS_BALANCE"] >= -5)

    # ---- 0c. [ITEM 1] Tendência (slope) do score por crédito ----
    # Regressão linear simples STATUS_SCORE ~ MONTHS_BALANCE por SK_ID_BUREAU.
    # slope > 0 → atraso crescendo com o tempo (piorando); < 0 → melhorando.
    def _status_trend(group: pd.DataFrame) -> float:
        x = group["MONTHS_BALANCE"].to_numpy(dtype="float64")
        y = group["STATUS_SCORE"].to_numpy(dtype="float64")
        if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return 0.0
        # slope = cov(x, y) / var(x)
        return float(np.polyfit(x, y, 1)[0])

    status_trend = (
        bb.groupby("SK_ID_BUREAU")
          .apply(_status_trend)
          .rename("STATUS_SCORE_TREND")
    )

    # ---- 1. Agrega bureau_balance por crédito (SK_ID_BUREAU) ----
    # Seleciona apenas colunas numéricas de bb para agregar com mean.
    # Pandas moderno pode gerar dtype bool ou uint8 no get_dummies — cobrimos os dois.
    # Excluímos as chaves (SK_ID_BUREAU) e a coluna base (MONTHS_BALANCE).
    # Excluímos também as colunas derivadas de score (têm agregação dedicada abaixo).
    score_cols = ["STATUS_SCORE", "STATUS_SCORE_RECENT6"]
    bb_numeric_cols = bb.select_dtypes(include=["number", "bool"]).columns.tolist()
    bb_cols_to_agg  = [c for c in bb_numeric_cols
                       if c not in ["SK_ID_BUREAU", "MONTHS_BALANCE"] + score_cols]

    bb_agg_spec = {"MONTHS_BALANCE": ["min", "max", "size"]}
    for col in bb_cols_to_agg:
        bb_agg_spec[col] = ["mean"]   # média de coluna 0/1 = proporção de meses naquele status

    # [ITEM 1] Agregações dedicadas do score de severidade
    bb_agg_spec["STATUS_SCORE"]         = ["max", "mean"]
    bb_agg_spec["STATUS_SCORE_RECENT6"] = ["mean"]

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_agg_spec)
    bb_agg.columns = pd.Index([f"{e[0]}_{e[1].upper()}" for e in bb_agg.columns])

    # [ITEM 1] Junta o slope de tendência (uma coluna por crédito)
    bb_agg = bb_agg.join(status_trend, how="left")
    del status_trend

    bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")
    bureau.drop(["SK_ID_BUREAU"], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Para as categóricas (dummies), filtra só colunas numéricas/bool existentes no bureau
    # e que não façam parte do num_agg — evita tentar agregar colunas string
    bureau_num_cols = bureau.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_agg = {
        col: ["mean"]
        for col in bureau_num_cols
        if col not in BUREAU_NUM_AGG and col != ID_COLUMN
    }

    bureau_agg = bureau.groupby(ID_COLUMN).agg({**BUREAU_NUM_AGG, **cat_agg})
    bureau_agg.columns = pd.Index([f"BURO_{e[0]}_{e[1].upper()}" for e in bureau_agg.columns])

    # ---- 3a. Features de créditos ATIVOS (carga atual de dívida) ----
    if "CREDIT_ACTIVE_Active" in bureau.columns:
        active     = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
        active_agg = active.groupby(ID_COLUMN).agg(BUREAU_NUM_AGG)
        active_agg.columns = pd.Index([f"ACTIVE_{e[0]}_{e[1].upper()}" for e in active_agg.columns])
        bureau_agg = bureau_agg.join(active_agg, how="left", on=ID_COLUMN)
        del active, active_agg

    # ---- 3b. Features de créditos ENCERRADOS (histórico de comportamento) ----
    if "CREDIT_ACTIVE_Closed" in bureau.columns:
        closed     = bureau[bureau["CREDIT_ACTIVE_Closed"] == 1]
        closed_agg = closed.groupby(ID_COLUMN).agg(BUREAU_NUM_AGG)
        closed_agg.columns = pd.Index([f"CLOSED_{e[0]}_{e[1].upper()}" for e in closed_agg.columns])
        bureau_agg = bureau_agg.join(closed_agg, how="left", on=ID_COLUMN)
        del closed, closed_agg

    del bureau
    gc.collect()
    return bureau_agg

# ============================================================
# AGREGAÇÕES: APLICAÇÕES ANTERIORES
# ============================================================

def build_previous_app_features(num_rows=None) -> pd.DataFrame:
    """
    Agrega previous_application.csv por cliente.

    Cria features separadas para aplicações Aprovadas vs Recusadas:
    - Aprovadas: confiança histórica da instituição no cliente
    - Recusadas: tentativas anteriores em situação de risco
    """
    prev = load_previous_applications(num_rows)

    # Razão pedido/aprovado: < 1 significa que o banco cortou o valor solicitado
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]

    prev_num_cols = prev.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_agg = {col: ["mean"] for col in prev_num_cols if col not in PREV_APP_NUM_AGG and col != ID_COLUMN}

    prev_agg = prev.groupby(ID_COLUMN).agg({**PREV_APP_NUM_AGG, **cat_agg})
    prev_agg.columns = pd.Index([f"PREV_{e[0]}_{e[1].upper()}" for e in prev_agg.columns])

    # Features de aplicações APROVADAS
    if "NAME_CONTRACT_STATUS_Approved" in prev.columns:
        approved     = prev[prev["NAME_CONTRACT_STATUS_Approved"] == 1]
        approved_agg = approved.groupby(ID_COLUMN).agg(PREV_APP_NUM_AGG)
        approved_agg.columns = pd.Index([f"APPROVED_{e[0]}_{e[1].upper()}" for e in approved_agg.columns])
        prev_agg = prev_agg.join(approved_agg, how="left", on=ID_COLUMN)
        del approved, approved_agg

    # Features de aplicações RECUSADAS
    if "NAME_CONTRACT_STATUS_Refused" in prev.columns:
        refused     = prev[prev["NAME_CONTRACT_STATUS_Refused"] == 1]
        refused_agg = refused.groupby(ID_COLUMN).agg(PREV_APP_NUM_AGG)
        refused_agg.columns = pd.Index([f"REFUSED_{e[0]}_{e[1].upper()}" for e in refused_agg.columns])
        prev_agg = prev_agg.join(refused_agg, how="left", on=ID_COLUMN)
        del refused, refused_agg

    del prev
    gc.collect()
    return prev_agg


# ============================================================
# AGREGAÇÕES: POS CASH BALANCE
# ============================================================

def build_pos_cash_features(num_rows=None) -> pd.DataFrame:
    """
    Agrega POS_CASH_balance por cliente.

    Foco: comportamento de atraso (SK_DPD = Days Past Due).
    SK_DPD > 0 indica que o cliente atrasou o pagamento naquele mês.
    """
    pos = load_pos_cash(num_rows)

    pos_num_cols = pos.select_dtypes(include=["number", "bool"]).columns.tolist()
    agg = {
        **POS_CASH_AGG_BASE,
        **{col: ["mean"] for col in pos_num_cols
           if col not in ["MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF", ID_COLUMN]},
    }

    pos_agg = pos.groupby(ID_COLUMN).agg(agg)
    pos_agg.columns = pd.Index([f"POS_{e[0]}_{e[1].upper()}" for e in pos_agg.columns])
    pos_agg["POS_COUNT"] = pos.groupby(ID_COLUMN).size()

    del pos
    gc.collect()
    return pos_agg


# ============================================================
# AGREGAÇÕES: INSTALLMENTS PAYMENTS
# ============================================================

def build_installments_features(num_rows=None) -> pd.DataFrame:
    """
    Agrega installments_payments por cliente.

    Calcula DPD (atraso real em dias) e DBD (dias pagos antes do vencimento)
    a partir das datas de pagamento e vencimento — uma das fontes mais ricas
    de sinal sobre disciplina financeira.
    """
    ins = load_installments(num_rows)

    # % pago em relação ao devido: < 1 = pagou parcialmente (risco)
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]

    # Diferença absoluta: valor que ficou em aberto
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]

    # Dias de atraso (positivo) — zera quando pagou antes do vencimento
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)

    # Dias antes do vencimento (positivo) — zera quando pagou depois
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

    ins_num_cols    = ins.select_dtypes(include=["number", "bool"]).columns.tolist()
  
    agg = {
        **INSTALLMENTS_AGG_BASE,
        **{col: ["mean"] for col in ins_num_cols
           if col not in INSTALLMENTS_EXPLICIT_COLS and col != ID_COLUMN},
    }

    ins_agg = ins.groupby(ID_COLUMN).agg(agg)
    ins_agg.columns = pd.Index([f"INSTAL_{e[0]}_{e[1].upper()}" for e in ins_agg.columns])
    ins_agg["INSTAL_COUNT"] = ins.groupby(ID_COLUMN).size()

    del ins
    gc.collect()
    return ins_agg


# ============================================================
# AGREGAÇÕES: CREDIT CARD BALANCE
# ============================================================

def build_credit_card_features(num_rows=None) -> pd.DataFrame:
    """
    Agrega credit_card_balance por cliente.

    Aplica min/max/mean/sum/var em todas as colunas numéricas — abordagem
    ampla para capturar padrões de uso do cartão sem selecionar features a priori.
    """
    cc = load_credit_card(num_rows)

# Seleciona apenas colunas numéricas/booleanas — evita aplicar mean/sum/var
    # em colunas de texto (ex.: NAME_CONTRACT_STATUS), que quebram o agg.
    cc_num = cc.select_dtypes(include=["number", "bool"])

    cc_agg = cc_num.groupby(cc[ID_COLUMN]).agg(["min", "max", "mean", "sum", "var"])
    cc_agg.columns = pd.Index([f"CC_{e[0]}_{e[1].upper()}" for e in cc_agg.columns])
    cc_agg["CC_COUNT"] = cc.groupby(ID_COLUMN).size()

    del cc, cc_num
    gc.collect()
    return cc_agg

def build_cross_features(df):
    """
    Cria 5 features que CRUZAM informacoes de tabelas diferentes
    (application + bureau + previous_application + POS_CASH + installments + credit_card).

    Diferente das agregacoes isoladas, estas razoes combinam sinais de fontes
    distintas - capturando relacoes que nenhuma agregacao sozinha enxerga.
    DEVE ser chamada DEPOIS de todos os merges (colunas BURO_/PREV_/POS_/INSTAL_/CC_).
    """
    eps = 1e-5  # evita divisao por zero sem distorcer a magnitude

    def col(name):
        # retorna a coluna se existir; senao, Series de NaN (robusto a ABTs parciais)
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    # 1. DEBT_INCOME_RATIO: divida total no bureau / renda
    debt_income  = col("BURO_AMT_CREDIT_SUM_DEBT_SUM") / (col("AMT_INCOME_TOTAL") + eps)

    # 2. CURR_PREV_CREDIT_RATIO: credito atual / media de creditos anteriores
    curr_prev    = col("AMT_CREDIT") / (col("PREV_AMT_CREDIT_MEAN") + eps)

    # 3. CREDIT_GOODS_PRICE_RATIO: credito / valor do bem (>1 = credito inflado)
    credit_goods = col("AMT_CREDIT") / (col("AMT_GOODS_PRICE") + eps)

    # 4. COMBINED_DPD_MEAN: atraso medio consolidado POS_CASH + installments
    #    media PONDERADA pelo numero de registros de cada produto
    pos_cnt, inst_cnt = col("POS_COUNT").fillna(0), col("INSTAL_COUNT").fillna(0)
    weighted   = (col("POS_SK_DPD_MEAN").fillna(0) * pos_cnt
                  + col("INSTAL_DPD_MEAN").fillna(0) * inst_cnt)
    total_cnt  = pos_cnt + inst_cnt
    combined_dpd = (weighted / (total_cnt + eps)).where(total_cnt > 0, np.nan)

    # 5. CC_UTILIZATION_INCOME: saldo medio do cartao / renda
    cc_util_income = col("CC_AMT_BALANCE_MEAN") / (col("AMT_INCOME_TOTAL") + eps)

    new_cols = {
        "DEBT_INCOME_RATIO":        debt_income,
        "CURR_PREV_CREDIT_RATIO":   curr_prev,
        "CREDIT_GOODS_PRICE_RATIO": credit_goods,
        "COMBINED_DPD_MEAN":        combined_dpd,
        "CC_UTILIZATION_INCOME":    cc_util_income,
    }

    # usa pd.concat (evita PerformanceWarning de fragmentacao) e limpa +-inf
    cross = pd.DataFrame(new_cols, index=df.index).replace([np.inf, -np.inf], np.nan)
    df = pd.concat([df, cross], axis=1)
    return df

# ============================================================
# PONTO DE ENTRADA (VS Code e AirFlow)
# ============================================================

def run():
    """
    Constrói a ABT completa e salva abt.csv.
    Chamável diretamente (python abt_transform.py) ou via AirFlow PythonOperator.
    """
    print("=== Iniciando construção da ABT ===")

    # Carrega dados limpos gerados pelo data_sanitization.py
    df = pd.read_csv(CLEAN_DATA_PATH)
    df = build_application_features(df)

    with timer("Bureau e bureau_balance"):
        bureau = build_bureau_features(NUM_ROWS)
        print(f"  Bureau shape: {bureau.shape}")
        df = df.join(bureau, how="left", on=ID_COLUMN)
        del bureau; gc.collect()

    with timer("Previous applications"):
        prev = build_previous_app_features(NUM_ROWS)
        print(f"  Previous apps shape: {prev.shape}")
        df = df.join(prev, how="left", on=ID_COLUMN)
        del prev; gc.collect()

    with timer("POS Cash balance"):
        pos = build_pos_cash_features(NUM_ROWS)
        print(f"  POS Cash shape: {pos.shape}")
        df = df.join(pos, how="left", on=ID_COLUMN)
        del pos; gc.collect()

    with timer("Installments payments"):
        ins = build_installments_features(NUM_ROWS)
        print(f"  Installments shape: {ins.shape}")
        df = df.join(ins, how="left", on=ID_COLUMN)
        del ins; gc.collect()

    with timer("Credit card balance"):
        cc = build_credit_card_features(NUM_ROWS)
        print(f"  Credit card shape: {cc.shape}")
        df = df.join(cc, how="left", on=ID_COLUMN)
        del cc; gc.collect()
    
    with timer("Cross Features"):
        df = build_cross_features(df)          

    os.makedirs(os.path.dirname(ABT_DATA_PATH), exist_ok=True)
    df.to_csv(ABT_DATA_PATH, index=False)
    print(f"ABT salva em {ABT_DATA_PATH} | shape: {df.shape}")


if __name__ == "__main__":
    run()