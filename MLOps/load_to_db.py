"""
load_to_db.py — Carga dos dados do pipeline para o PostgreSQL.

Fecha a lacuna entre o pipeline (que produz CSVs) e o serviço de predição (que
precisa ler UM cliente com baixa latência). Depois desta etapa, a API não
depende mais de receber as features prontas no corpo da requisição: ela busca o
cliente na `feature_store.abt`.

    Dados/clean_data.csv  ──►  staging.clean_data
    Dados/abt.csv         ──►  feature_store.abt      ← lida pela API

Três cuidados que o código toma, e o porquê:

1. NOMES DE COLUNA
   O train.py sanitiza os nomes ("STATUS_[Approved]" → "STATUS__Approved__")
   antes de treinar; 90 das 838 colunas da ABT mudam de nome nesse processo.
   A carga aplica exatamente a mesma regra — senão a chave gravada no banco não
   casaria com o que o modelo espera, e a feature viraria NaN silenciosamente.

2. AUSENTES NÃO SÃO GRAVADOS
   Em média, 221 das 836 features de um cliente são NaN (quem nunca teve
   crédito não tem histórico de bureau). Gravar `"FEATURE": null` ocuparia
   espaço para dizer "não sei". A chave simplesmente não existe no JSONB — e a
   ausência já significa NaN, que é como o LightGBM trata o dado faltante.

3. CARGA POR COPY, EM LOTES
   INSERT linha a linha levaria horas. Usamos COPY (o caminho mais rápido do
   Postgres) sobre lotes lidos do CSV em streaming — a ABT tem 1,3 GB e não
   cabe confortavelmente na memória de uma vez.

Uso:
    python -m MLOps.load_to_db --abt                      # amostra padrao
    python -m MLOps.load_to_db --abt --limit 50000
    python -m MLOps.load_to_db --abt --full               # base completa
    python -m MLOps.load_to_db --clean                    # camada staging
    python -m MLOps.load_to_db --abt --clean --truncate   # recarga do zero
"""

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text

from config import (
    ABT_DATA_PATH, CLEAN_DATA_PATH, DEMO_CLIENTS_PATH, MODEL_FEATURES_PATH,
    DB_CHUNK_SIZE, ID_COLUMN, TARGET_COLUMN, NUM_ROWS,
)
from MLOps.db import get_engine

# Limite padrão de uma carga INCREMENTAL (a que o Airflow dispara a cada
# execução da DAG). O banco é povoado uma vez com a base completa
# (`--full`: 356 mil clientes, ~3,9 GB, ~16 min); as execuções seguintes só
# precisam atualizar uma fatia, e como a carga é UPSERT, elas atualizam o que
# tocam sem apagar o resto.
DEFAULT_LIMIT = int(os.environ.get("ABT_LOAD_LIMIT", "20000"))


# ============================================================
# PREPARO DAS LINHAS
# ============================================================

def _sanitize(colunas) -> list:
    """Aplica a MESMA normalização de nomes usada por train.py e predict.py."""
    return [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in colunas]


def _to_json(valor):
    """
    Converte um valor do pandas para algo serializável em JSON.

    Devolve None para ausentes (NaN/NaT) e para infinitos — que aparecem em
    razões com denominador zero e não têm representação em JSON.
    """
    if valor is None:
        return None
    if isinstance(valor, float):
        if math.isnan(valor) or math.isinf(valor):
            return None
        return valor
    # numpy int64/float64 e afins expõem .item(), que devolve o tipo nativo
    item = getattr(valor, "item", None)
    if callable(item):
        try:
            return _to_json(item())
        except (ValueError, TypeError):
            return None
    return valor


def _linha_para_payload(registro: dict) -> str:
    """Monta o JSONB de um cliente, descartando as features ausentes."""
    limpo = {}
    for chave, valor in registro.items():
        convertido = _to_json(valor)
        if convertido is not None:
            limpo[chave] = convertido
    return json.dumps(limpo, ensure_ascii=False)


# ============================================================
# CARGA VIA COPY
# ============================================================

# Tipos das colunas de carga, usados para criar a tabela temporária do UPSERT.
TIPOS_COLUNA = {
    "sk_id_curr":  "bigint",
    "target":      "smallint",
    "is_train":    "boolean",
    "features":    "jsonb",
    "payload":     "jsonb",
    "abt_version": "text",
}


def _copy(engine, tabela: str, colunas: list, linhas: list, chave: str = None) -> int:
    """
    Envia um lote para o Postgres usando COPY.

    Monta um CSV em memória e o entrega ao servidor num único comando — uma
    ordem de grandeza mais rápido que INSERTs individuais.

    Com `chave`, faz UPSERT em vez de inserção simples: o lote vai primeiro
    para uma tabela temporária e de lá é fundido no destino (ON CONFLICT DO
    UPDATE).

    Por que isso importa: a task do Airflow recarrega uma AMOSTRA a cada
    execução. Se ela apagasse a tabela antes (TRUNCATE), acionar a DAG numa
    demonstração destruiria a base completa já carregada e deixaria só a
    amostra no lugar. Com UPSERT, a execução atualiza o que tocar e preserva
    todo o resto.
    """
    if not linhas:
        return 0

    buffer = io.StringIO()
    escritor = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    escritor.writerows(linhas)
    buffer.seek(0)

    # COPY é uma operação do driver (psycopg2), não do SQLAlchemy: precisamos da
    # conexão crua. O commit é explícito porque raw_connection não usa a
    # transação gerenciada pela engine.
    bruta = engine.raw_connection()
    try:
        with bruta.cursor() as cur:
            if chave is None:
                cur.copy_expert(
                    f"COPY {tabela} ({', '.join(colunas)}) FROM STDIN WITH (FORMAT csv)",
                    buffer,
                )
            else:
                # A temporária vive só nesta conexão e some no fim da transação.
                definicao = ", ".join(f"{c} {TIPOS_COLUNA[c]}" for c in colunas)
                cur.execute(f"CREATE TEMP TABLE lote ({definicao}) ON COMMIT DROP")
                cur.copy_expert(
                    f"COPY lote ({', '.join(colunas)}) FROM STDIN WITH (FORMAT csv)",
                    buffer,
                )
                atualiza = ", ".join(f"{c} = EXCLUDED.{c}" for c in colunas if c != chave)
                cur.execute(f"""
                    INSERT INTO {tabela} ({', '.join(colunas)})
                    SELECT {', '.join(colunas)} FROM lote
                    ON CONFLICT ({chave}) DO UPDATE SET {atualiza}
                """)
        bruta.commit()
    finally:
        bruta.close()

    return len(linhas)


def _truncar(engine, tabela: str) -> None:
    """Esvazia a tabela antes de uma recarga (RESTART IDENTITY zera as sequences)."""
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {tabela} RESTART IDENTITY CASCADE"))
    print(f"  {tabela} esvaziada")


# ============================================================
# CARGA DA ABT (feature store)
# ============================================================

def _ids_da_demo() -> set:
    """
    IDs dos clientes usados pelo painel de demonstração.

    A carga parcial pega as primeiras N linhas da ABT; sem este cuidado, um
    cliente que aparece no seletor do painel poderia não existir no banco e a
    demonstração quebraria ao vivo.
    """
    if not os.path.exists(DEMO_CLIENTS_PATH):
        return set()
    return set(pd.read_csv(DEMO_CLIENTS_PATH, usecols=[ID_COLUMN])[ID_COLUMN].tolist())


def load_abt(limite: int = None, truncate: bool = False,
             incluir_demo: bool = True, versao: str = None,
             permitir_amostra: bool = False) -> int:
    """
    Carrega a ABT em `feature_store.abt`.

    Grava apenas as 836 features que o modelo usa — o que sobra na ABT (IDs
    auxiliares, colunas descartadas no treino) não tem utilidade na inferência
    e só ocuparia espaço.

    TRAVA DE SEGURANÇA (`permitir_amostra`): a carga se recusa a publicar uma
    ABT construída sobre AMOSTRA. Isso não é zelo abstrato — aconteceu:

        uma execução de demonstração da DAG rodou com NUM_ROWS=30000, que corta
        CADA tabela de origem. As agregações de bureau e de aplicações
        anteriores saíram vazias, e o UPSERT gravou isso por cima dos valores
        completos. O cliente 100002 caiu de 658 para 243 features, e a sua PD
        pulou de 0,346 para 0,457 — sem que nada no sistema acusasse.

    Uma ABT de amostra serve para testar o pipeline, nunca para servir decisões.
    """
    if not os.path.exists(ABT_DATA_PATH):
        raise FileNotFoundError(
            f"ABT não encontrada em {ABT_DATA_PATH}. "
            f"Gere primeiro com: python -m DataPipeline.abt_transform"
        )

    if NUM_ROWS and not permitir_amostra:
        raise RuntimeError(
            f"Recusando publicar uma ABT de AMOSTRA na feature store.\n"
            f"  NUM_ROWS={NUM_ROWS} está definido, então esta ABT foi construída sobre um\n"
            f"  recorte de cada tabela de origem: os clientes sairiam sem o histórico de\n"
            f"  bureau, parcelas e aplicações anteriores, e o UPSERT sobrescreveria os\n"
            f"  valores completos que já estão no banco.\n"
            f"  Para publicar mesmo assim (ambiente de teste): --permitir-amostra"
        )
    with open(MODEL_FEATURES_PATH, "r", encoding="utf-8") as f:
        features = json.load(f)
    conjunto_features = set(features)

    engine = get_engine()
    if truncate:
        _truncar(engine, "feature_store.abt")

    versao = versao or time.strftime("%Y-%m-%dT%H:%M:%S")
    demo_ids = _ids_da_demo() if incluir_demo else set()
    colunas_destino = ["sk_id_curr", "target", "is_train", "features", "abt_version"]

    print("Carregando ABT  ->  feature_store.abt")
    print(f"  origem : {ABT_DATA_PATH}")
    print(f"  limite : {'base completa' if limite is None else f'{limite:,} clientes'}")
    print(f"  demo   : {len(demo_ids)} clientes do painel garantidos na carga")

    inicio = time.time()
    total = 0
    vistos = set()
    pendentes_demo = set(demo_ids)

    leitor = pd.read_csv(ABT_DATA_PATH, chunksize=DB_CHUNK_SIZE, low_memory=False)
    for bloco in leitor:
        bloco.columns = _sanitize(bloco.columns)

        # Depois de atingir o limite, seguimos lendo só para recolher os
        # clientes da demonstração que ainda não apareceram.
        if limite is not None and total >= limite:
            if not pendentes_demo:
                break
            bloco = bloco[bloco[ID_COLUMN].isin(pendentes_demo)]
            if bloco.empty:
                continue
        elif limite is not None:
            restante = limite - total
            if len(bloco) > restante:
                faltantes = bloco[bloco[ID_COLUMN].isin(pendentes_demo)]
                bloco = pd.concat([bloco.head(restante), faltantes]).drop_duplicates(
                    subset=[ID_COLUMN]
                )

        colunas_feature = [c for c in bloco.columns if c in conjunto_features]
        alvo = bloco[TARGET_COLUMN] if TARGET_COLUMN in bloco.columns else None
        ids = bloco[ID_COLUMN].tolist()
        registros = bloco[colunas_feature].to_dict(orient="records")

        linhas = []
        for i, sk_id in enumerate(ids):
            if sk_id in vistos:          # evita violar a PK em recargas parciais
                continue
            vistos.add(sk_id)
            pendentes_demo.discard(sk_id)

            valor_alvo = _to_json(alvo.iloc[i]) if alvo is not None else None
            linhas.append([
                int(sk_id),
                "" if valor_alvo is None else int(valor_alvo),   # "" → NULL no COPY
                "true" if valor_alvo is not None else "false",   # is_train
                _linha_para_payload(registros[i]),
                versao,
            ])

        total += _copy(engine, "feature_store.abt", colunas_destino, linhas,
                       chave="sk_id_curr")
        print(f"    {total:,} clientes carregados...", end="\r")

    duracao = time.time() - inicio
    print(f"\n  OK: {total:,} clientes em {duracao:.0f}s "
          f"({total / max(duracao, 1):.0f} linhas/s) · versao '{versao}'")
    return total


# ============================================================
# CARGA DO CLEAN_DATA (camada staging)
# ============================================================

def load_clean(limite: int = None, truncate: bool = False) -> int:
    """Carrega os dados limpos em `staging.clean_data` (camada silver)."""
    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(
            f"clean_data não encontrado em {CLEAN_DATA_PATH}. "
            f"Gere primeiro com: python -m DataPipeline.data_sanitization"
        )

    engine = get_engine()
    if truncate:
        _truncar(engine, "staging.clean_data")

    print("Carregando clean_data  ->  staging.clean_data")
    print(f"  limite : {'base completa' if limite is None else f'{limite:,} linhas'}")

    inicio, total, vistos = time.time(), 0, set()
    colunas_destino = ["sk_id_curr", "target", "payload"]

    for bloco in pd.read_csv(CLEAN_DATA_PATH, chunksize=DB_CHUNK_SIZE, low_memory=False):
        if limite is not None:
            if total >= limite:
                break
            bloco = bloco.head(limite - total)

        bloco.columns = _sanitize(bloco.columns)
        alvo = bloco[TARGET_COLUMN] if TARGET_COLUMN in bloco.columns else None
        ids = bloco[ID_COLUMN].tolist()
        demais = [c for c in bloco.columns if c not in (ID_COLUMN, TARGET_COLUMN)]
        registros = bloco[demais].to_dict(orient="records")

        linhas = []
        for i, sk_id in enumerate(ids):
            if sk_id in vistos:
                continue
            vistos.add(sk_id)
            valor_alvo = _to_json(alvo.iloc[i]) if alvo is not None else None
            linhas.append([
                int(sk_id),
                "" if valor_alvo is None else int(valor_alvo),
                _linha_para_payload(registros[i]),
            ])

        total += _copy(engine, "staging.clean_data", colunas_destino, linhas,
                         chave="sk_id_curr")
        print(f"    {total:,} linhas carregadas...", end="\r")

    print(f"\n  OK: {total:,} linhas em {time.time() - inicio:.0f}s")
    return total


# ============================================================
# RESUMO (usado no fim da carga e pelo Airflow)
# ============================================================

def resumo() -> dict:
    """Confere o que ficou no banco depois da carga."""
    with get_engine().connect() as conn:
        clientes = conn.execute(text("SELECT count(*) FROM feature_store.abt")).scalar_one()
        staging  = conn.execute(text("SELECT count(*) FROM staging.clean_data")).scalar_one()
        medias   = conn.execute(text("""
            SELECT round(avg(jsonb_array_length(
                       (SELECT jsonb_agg(k) FROM jsonb_object_keys(features) k))), 0) AS chaves
            FROM (SELECT features FROM feature_store.abt LIMIT 500) amostra
        """)).scalar()
    return {"clientes_abt": clientes, "linhas_staging": staging, "features_por_cliente": medias}


# ============================================================
# CLI
# ============================================================

def _run_cli() -> int:
    parser = argparse.ArgumentParser(description="Carrega os dados do pipeline no PostgreSQL")
    parser.add_argument("--abt",      action="store_true", help="carrega a ABT em feature_store.abt")
    parser.add_argument("--clean",    action="store_true", help="carrega clean_data em staging.clean_data")
    parser.add_argument("--truncate", action="store_true", help="esvazia a tabela antes de carregar")
    parser.add_argument("--full",     action="store_true", help="carrega a base completa (sem limite)")
    parser.add_argument("--limit",    type=int, default=DEFAULT_LIMIT,
                        help=f"quantos registros carregar (padrao: {DEFAULT_LIMIT:,})")
    parser.add_argument("--permitir-amostra", action="store_true",
                        help="publica mesmo com NUM_ROWS definido (ABT de amostra)")
    args = parser.parse_args()

    if not (args.abt or args.clean):
        parser.print_help()
        return 0

    limite = None if args.full else args.limit

    if args.clean:
        load_clean(limite=limite, truncate=args.truncate)
    if args.abt:
        load_abt(limite=limite, truncate=args.truncate,
                 permitir_amostra=args.permitir_amostra)

    print("\nEstado do banco:")
    for chave, valor in resumo().items():
        print(f"  {chave:<22} {valor:,}" if isinstance(valor, int) else f"  {chave:<22} {valor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
