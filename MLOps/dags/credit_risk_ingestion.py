"""
DAG de INGESTÃO — dados brutos até a feature store.

    raw_data  →  sanitize  →  build_abt  →  load_feature_store  →  PostgreSQL

É a esteira que alimenta tudo: sem ela a feature store envelhece, e a API passa
a decidir sobre o histórico de ontem.

MODO DEMO: as duas primeiras tasks rodam sobre uma amostra (NUM_ROWS) e gravam
os CSVs intermediários num volume isolado (/demo), para a demonstração levar
2 minutos em vez de 25.

POR QUE A CARGA USA A ABT DE PRODUÇÃO, E NÃO A QUE ACABOU DE SER CONSTRUÍDA:

    `NUM_ROWS` corta CADA tabela de origem, não só a principal. Uma ABT de
    amostra sai com as agregações de bureau, parcelas e aplicações anteriores
    vazias — o cliente perde o histórico, não uma parte dele.

    Isso não é hipótese. Numa execução de teste desta DAG, o UPSERT gravou
    features de amostra por cima das completas em 20.532 clientes: o cliente
    100002 caiu de 658 para 243 features e a sua PD pulou de 0,346 para 0,457,
    sem que nada acusasse.

    Em produção esta task carregaria exatamente a ABT que a task anterior
    construiu — porque lá ela é construída sobre a base inteira. No modo demo
    ela carrega a ABT de produção, e `load_to_db` tem uma trava que recusa
    publicar uma ABT de amostra mesmo que alguém tente.
"""

from airflow import DAG

from _comum import AMBIENTE_PRODUCAO, DEFAULT_ARGS, INICIO, tarefa

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
        ambiente=AMBIENTE_PRODUCAO,
    )

    sanitize >> build_abt >> load_feature_store
