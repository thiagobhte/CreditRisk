"""
db.py — Camada de acesso ao banco de dados (PostgreSQL).

Ponto único de conexão do projeto. Todos os módulos que falam com o banco
(carga da ABT, API, monitoramento, DAGs do Airflow) passam por aqui, em vez de
cada um montar sua própria string de conexão.

Por que um banco, e não os CSVs de antes:
    - a API precisa ler UM cliente por vez, com baixa latência — ler uma linha
      de um CSV de 1,3 GB não é viável em produção;
    - as decisões de crédito precisam ficar registradas para auditoria;
    - o monitoramento precisa de série histórica, não de um JSON sobrescrito.

Uso:
    python -m MLOps.db --check    # testa a conexão e mostra a versão do servidor
    python -m MLOps.db --init     # aplica o schema (idempotente)
    python -m MLOps.db --status   # lista schemas, tabelas e contagem de linhas

Na aplicação:
    from MLOps.db import get_engine
    with get_engine().connect() as conn:
        ...
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config import DATABASE_URL, SCHEMA_SQL_PATH


# ============================================================
# ENGINE (criada uma vez e reaproveitada)
# ============================================================
# A engine do SQLAlchemy mantém um POOL de conexões. Criá-la a cada chamada
# abriria uma conexão nova por requisição da API — caro e limitado pelo
# max_connections do Postgres. Por isso guardamos a instância no módulo.

_ENGINE: Engine | None = None


def get_engine() -> Engine:
    """
    Devolve a engine compartilhada, criando-a na primeira chamada.

    pool_pre_ping=True faz o SQLAlchemy testar a conexão antes de entregá-la:
    sem isso, uma conexão que morreu (container do banco reiniciado, timeout de
    rede) só é descoberta no meio de uma query, derrubando a requisição.
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            future=True,
        )
    return _ENGINE


def check_connection() -> dict:
    """
    Testa a conexão e devolve informações do servidor.

    Usado pelo /health da API e pelo healthcheck do docker-compose: um serviço
    que não alcança o banco está "vivo", mas não está "pronto".
    """
    with get_engine().connect() as conn:
        versao = conn.execute(text("SELECT version()")).scalar_one()
        base   = conn.execute(text("SELECT current_database()")).scalar_one()
    return {"ok": True, "database": base, "server_version": versao.split(",")[0]}


# ============================================================
# APLICAÇÃO DO SCHEMA
# ============================================================

def init_db(schema_path: str = None) -> None:
    """
    Aplica o schema.sql no banco.

    É idempotente (todo o DDL usa IF NOT EXISTS / OR REPLACE), então pode rodar
    quantas vezes for preciso. O Postgres também executa este mesmo arquivo
    sozinho na primeira subida do container, via /docker-entrypoint-initdb.d —
    este comando existe para os casos em que o volume já existe e o schema
    precisa ser (re)aplicado sem destruir os dados.
    """
    schema_path = schema_path or SCHEMA_SQL_PATH
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema não encontrado: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        ddl = f.read()

    # exec_driver_sql executa o arquivo inteiro de uma vez, preservando os
    # blocos $func$...$func$ da função plpgsql — que quebrariam se
    # tentássemos dividir o arquivo por ";".
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)

    print(f"Schema aplicado com sucesso a partir de: {schema_path}")


# ============================================================
# INSPEÇÃO (apoio à demonstração e ao diagnóstico)
# ============================================================

def status() -> list:
    """Lista as tabelas dos schemas do projeto com a contagem de linhas."""
    consulta = text("""
        SELECT schemaname AS schema, relname AS tabela, n_live_tup AS linhas_aprox
        FROM pg_stat_user_tables
        WHERE schemaname IN ('staging', 'feature_store', 'serving', 'mlops')
        ORDER BY schemaname, relname
    """)
    with get_engine().connect() as conn:
        return [dict(r._mapping) for r in conn.execute(consulta)]


def table_exists(schema: str, tabela: str) -> bool:
    """Informa se uma tabela existe — usado antes de cargas e leituras."""
    consulta = text("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = :schema AND table_name = :tabela
        )
    """)
    with get_engine().connect() as conn:
        return bool(conn.execute(consulta, {"schema": schema, "tabela": tabela}).scalar())


# ============================================================
# CLI
# ============================================================

def _run_cli() -> int:
    parser = argparse.ArgumentParser(description="Utilitários do banco de dados do projeto")
    parser.add_argument("--check",  action="store_true", help="testa a conexão com o banco")
    parser.add_argument("--init",   action="store_true", help="aplica o schema.sql (idempotente)")
    parser.add_argument("--status", action="store_true", help="lista tabelas e contagem de linhas")
    args = parser.parse_args()

    if not (args.check or args.init or args.status):
        parser.print_help()
        return 0

    # Mostra para onde estamos apontando, mas sem vazar a senha no log.
    print(f"Conexao: {DATABASE_URL.split('@')[-1]}")

    try:
        if args.check:
            info = check_connection()
            print(f"[OK] conectado ao banco '{info['database']}'")
            print(f"     {info['server_version']}")

        if args.init:
            init_db()

        if args.status:
            linhas = status()
            if not linhas:
                print("Nenhuma tabela encontrada — rode antes: python -m MLOps.db --init")
            else:
                print(f"\n{'SCHEMA':<15} {'TABELA':<22} {'LINHAS':>12}")
                print("-" * 51)
                for r in linhas:
                    print(f"{r['schema']:<15} {r['tabela']:<22} {r['linhas_aprox']:>12,}")

    except SQLAlchemyError as e:
        # Erro de banco é operacional (container no ar? senha certa?), não um bug
        # do código — por isso a mensagem é orientada à ação, e não um traceback.
        print(f"\n[FALHA] nao foi possivel falar com o banco:\n  {e}\n")
        print("Verifique se o container esta no ar:")
        print("  docker compose -f MLOps/docker-compose.yml up -d postgres")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
