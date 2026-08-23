"""
DAG de SCORING EM LOTE — pontua a carteira todo dia.

    batch_scoring  →  serving.predictions

É o segundo caminho de consumo do modelo, ao lado da API:

    API    → decisão individual, síncrona, no momento da proposta
    batch  → decisão em massa, agendada, sobre a fila de propostas

Usa o modelo e os dados de PRODUÇÃO (não a amostra do modo demo): pontuar a
carteira com um modelo de brinquedo geraria decisões que ninguém pode usar, e
poluiria o log que o monitoramento lê.
"""

from airflow import DAG

from _comum import AMBIENTE_PRODUCAO, DEFAULT_ARGS, INICIO, tarefa

with DAG(
    dag_id="credit_risk_batch_scoring",
    description="Pontua a carteira em lote e grava as decisoes para auditoria",
    default_args=DEFAULT_ARGS,
    schedule="@daily",
    start_date=INICIO,
    catchup=False,
    max_active_runs=1,
    tags=["credit-risk", "scoring"],
) as dag:

    batch_scoring = tarefa(
        "batch_scoring",
        "python -m MLOps.batch_scoring --limite 2000 --somente-novos",
        ambiente=AMBIENTE_PRODUCAO,
    )
