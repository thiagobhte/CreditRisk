"""
DAG de MONITORAMENTO — as três camadas, todo dia.

    data_drift  ─┐
    prediction_drift ─┼─►  (persistem em mlops.monitoring_runs)
    performance ─┘

As três rodam em PARALELO porque são independentes: a performance por safra
depende de desfechos que só amadurecem meses depois (label lag) e pode não ter
o que calcular. Se estivessem em série, essa falta bloquearia justamente o
drift — que é o sinal antecipado, o único disponível no curto prazo.

Usa dados e modelo de PRODUÇÃO: monitorar a amostra do modo demo não diria nada
sobre o modelo que está decidindo de verdade.
"""

from airflow import DAG

from _comum import AMBIENTE_PRODUCAO, DEFAULT_ARGS, INICIO, tarefa

with DAG(
    dag_id="credit_risk_monitoring",
    description="Data drift (PSI), prediction drift e performance por safra",
    default_args=DEFAULT_ARGS,
    schedule="@daily",
    start_date=INICIO,
    catchup=False,
    max_active_runs=1,
    tags=["credit-risk", "monitoramento"],
) as dag:

    data_drift = tarefa(
        "data_drift",
        "python -m MLOps.monitoring --data-drift --amostra 8000",
        ambiente=AMBIENTE_PRODUCAO,
    )

    prediction_drift = tarefa(
        "prediction_drift",
        "python -m MLOps.monitoring --prediction-drift --janela-dias 30",
        ambiente=AMBIENTE_PRODUCAO,
    )

    performance = tarefa(
        "performance",
        "python -m MLOps.monitoring --performance",
        ambiente=AMBIENTE_PRODUCAO,
    )

    [data_drift, prediction_drift, performance]
