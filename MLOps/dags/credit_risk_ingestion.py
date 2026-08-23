"""
DAG de INGESTÃO — dados brutos até a feature store.

    raw_data  →  sanitize  →  build_abt  →  load_feature_store  →  PostgreSQL

É a esteira que alimenta tudo: sem ela a feature store envelhece, e a API passa
a decidir sobre o histórico de ontem.

MODO DEMO: roda sobre uma amostra (NUM_ROWS) e grava os CSVs intermediários num
volume isolado (/demo). A última task publica no banco COMPARTILHADO — mas por
UPSERT, então uma execução de demonstração atualiza os clientes que tocar e
preserva os 356 mil já carregados.
"""

from airflow import DAG

from _comum import DEFAULT_ARGS, INICIO, tarefa

with DAG(
    dag_id="credit_risk_ingestion",
    description="raw_data -> clean_data -> ABT -> feature store (PostgreSQL)",
    default_args=DEFAULT_ARGS,
    schedule="@daily",
    start_date=INICIO,
    catchup=False,
    max_active_runs=1,          # duas ingestões simultâneas brigariam pelos mesmos arquivos
    tags=["credit-risk", "dados"],
) as dag:

    sanitize = tarefa(
        "sanitize",
        "python -m DataPipeline.data_sanitization",
    )

    build_abt = tarefa(
        "build_abt",
        "python -m DataPipeline.abt_transform",
    )

    load_feature_store = tarefa(
        "load_feature_store",
        "python -m MLOps.load_to_db --abt --clean",
    )

    sanitize >> build_abt >> load_feature_store
