"""
data_sanitization.py — Limpeza e padronização dos dados brutos.

Responsabilidade: ler os CSVs brutos, aplicar limpezas básicas (valores
sentinela, tipos incorretos, registros inválidos) e salvar clean_data.csv.

AirFlow via PythonOperator:
    task = PythonOperator(task_id="sanitize", python_callable=run)
"""

import gc
import re

import numpy as np
import pandas as pd

# Importa configurações centralizadas
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import RAW_DATA, CLEAN_DATA_PATH, NUM_ROWS, SENTINEL_VALUE, PREV_APP_DATE_COLS


# ============================================================
# UTILITÁRIOS
# ============================================================

def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True):
    """
    Converte colunas categóricas (texto) em colunas binárias 0/1.

    Por que: LightGBM e a maioria dos modelos não aceitam strings.
    nan_as_category=True cria coluna extra '_nan' para ausências,
    pois o fato de um valor estar ausente pode ser informativo por si só.

    IMPORTANTE — detecção de colunas de texto (funciona em pandas 2 E 3):
        Não use `df[col].dtype == "object"`. A partir do pandas 3, colunas de
        texto têm dtype "str", não "object", e essa comparação retorna False
        para TODAS elas. O encoding vira um no-op silencioso: as categóricas
        sobrevivem como texto, e mais adiante o `to_numeric(errors="coerce")`
        do train.py as transforma em colunas 100% NaN — ou seja, o modelo perde
        escolaridade, ocupação, estado civil etc. sem nenhum erro aparecer.

        Também não passe "str" para o select_dtypes: no pandas 2 isso levanta
        TypeError("string dtypes are not allowed"). Os rótulos "object" e
        "string" cobrem as duas versões — no pandas 3 as colunas de texto são
        StringDtype, capturadas por "string".

        Isso importa de verdade: o container do Airflow roda pandas 2 enquanto
        o ambiente local roda pandas 3. O mesmo código precisa servir aos dois.

    Retorna: (df_transformado, lista_de_novas_colunas)
    """
    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]

    # get_dummies devolve colunas bool. Convertemos para int8 (0/1) porque:
    #   1) agregações min/max sobre bool viram True/False, que ao passar por
    #      CSV são lidos de volta como TEXTO ("True"/"False") — e aí o
    #      to_numeric os transforma em NaN, matando a coluna silenciosamente;
    #   2) int8 ocupa menos memória que bool em DataFrames largos.
    if new_columns:
        df[new_columns] = df[new_columns].astype("int8")

    return df, new_columns


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove caracteres especiais dos nomes de colunas.

    LightGBM não aceita [ ] { } nos nomes das features porque usa JSON
    internamente. Ocorre naturalmente depois do get_dummies (ex:
    'NAME_CONTRACT_STATUS_[Approved]'). Substitui tudo que não seja
    letra/número/underscore por underscore.
    """
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", col) for col in df.columns]
    return df


# ============================================================
# CARREGAMENTO E LIMPEZA DA TABELA PRINCIPAL
# ============================================================

def load_application(num_rows=None) -> pd.DataFrame:
    """
    Carrega application_train.csv + application_test.csv e aplica limpezas básicas.

    Estratégia de concatenação: une treino e teste antes do encoding para
    garantir que as mesmas colunas dummies existam nos dois conjuntos.
    """
    df      = pd.read_csv(RAW_DATA["application_train"], nrows=num_rows)
    test_df = pd.read_csv(RAW_DATA["application_test"],  nrows=num_rows)
    print(f"Train: {len(df)} linhas | Test: {len(test_df)} linhas")

    df = pd.concat([df, test_df], axis=0).reset_index(drop=True)

    # Remove 4 registros com CODE_GENDER = 'XNA' (valor inválido)
    df = df[df["CODE_GENDER"] != "XNA"]

    # Encoding binário para features com exatamente 2 categorias
    # pd.factorize → 0/1 | economiza colunas vs get_dummies que criaria 2
    for col in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[col], _ = pd.factorize(df[col])

    # One-hot nas demais categóricas
    df, _ = one_hot_encoder(df, nan_as_category=False)

    # Trata valor sentinela: 365243 em DAYS_EMPLOYED significa "sem emprego"
    # Manter causaria o modelo interpretar o cliente como empregado há ~1000 anos
    #
    # NOTA sobre Copy-on-Write (pandas 2.x/3 com CoW ativado):
    # df["col"].replace(..., inplace=True) opera numa cópia intermediária e
    # NUNCA atualiza o df original — falha silenciosamente (gera apenas o
    # ChainedAssignmentError como aviso). A forma correta é reatribuir a coluna
    # ou usar df.replace({"col": valor}, inplace=True).
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(SENTINEL_VALUE, np.nan)

    del test_df
    gc.collect()
    return df


# ============================================================
# LIMPEZA DAS TABELAS SECUNDÁRIAS
# ============================================================

def load_bureau(num_rows=None) -> pd.DataFrame:
    """
    Carrega bureau.csv e bureau_balance.csv e aplica encoding.
    Substitui sentinel em datas e remove SK_ID_BUREAU após o join.
    """
    bureau = pd.read_csv(RAW_DATA["bureau"],         nrows=num_rows)
    bb     = pd.read_csv(RAW_DATA["bureau_balance"], nrows=num_rows)

    bb,     _ = one_hot_encoder(bb,     nan_as_category=True)
    bureau, _ = one_hot_encoder(bureau, nan_as_category=True)

    return bureau, bb


def load_previous_applications(num_rows=None) -> pd.DataFrame:
    """
    Carrega previous_application.csv, aplica encoding e trata sentinelas de datas.
    Colunas de datas preenchidas com 365243 indicam que o evento não ocorreu.
    """
    prev, _ = one_hot_encoder(
        pd.read_csv(RAW_DATA["previous_app"], nrows=num_rows),
        nan_as_category=True,
    )

    for col in PREV_APP_DATE_COLS:
        # Mesma correção do load_application(): reatribuir em vez de inplace=True
        prev[col] = prev[col].replace(SENTINEL_VALUE, np.nan)

    return prev


def load_pos_cash(num_rows=None) -> pd.DataFrame:
    pos, _ = one_hot_encoder(
        pd.read_csv(RAW_DATA["pos_cash"], nrows=num_rows),
        nan_as_category=True,
    )
    return pos


def load_installments(num_rows=None) -> pd.DataFrame:
    ins, _ = one_hot_encoder(
        pd.read_csv(RAW_DATA["installments"], nrows=num_rows),
        nan_as_category=True,
    )
    return ins


def load_credit_card(num_rows=None) -> pd.DataFrame:
    cc, _ = one_hot_encoder(
        pd.read_csv(RAW_DATA["credit_card"], nrows=num_rows),
        nan_as_category=True,
    )
    # SK_ID_PREV é chave de contrato, não feature — remove antes de agregar
    cc.drop(["SK_ID_PREV"], axis=1, inplace=True)
    return cc


# ============================================================
# PONTO DE ENTRADA (VS Code e AirFlow)
# ============================================================

def run():
    """
    Executa a sanitização completa e salva clean_data.csv.
    Chamável diretamente (python data_sanitization.py) ou via AirFlow PythonOperator.
    """
    print("=== Iniciando sanitização dos dados ===")

    df = load_application(NUM_ROWS)
    df = sanitize_column_names(df)

    # Converte colunas object/string remanescentes para numérico.
    # Pandas 3 introduziu o dtype "str" como distinto de "object" — uma coluna
    # de texto pode ter qualquer um dos dois dependendo de como foi criada.
    # Incluímos ambos explicitamente para não deixar nenhuma coluna de texto
    # passar intacta para o clean_data.csv (o que quebraria modelos lineares
    # como a Regressão Logística mais adiante no pipeline).
    # "object" + "string" cobre pandas 2 e 3. Nunca use "str" aqui: o pandas 2
    # levanta TypeError("string dtypes are not allowed").
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"clean_data.csv salvo em {CLEAN_DATA_PATH} | shape: {df.shape}")
    gc.collect()


if __name__ == "__main__":
    run()