"""
DAG de TREINO — treina o modelo e o registra para governança.

    train  →  register_model

Separada da ingestão de propósito: re-treinar é uma decisão, não uma rotina
diária. Esta DAG fica com `schedule=None` (só roda quando alguém aciona, ou
quando o monitoramento recomenda) — treinar todo dia sem motivo gastaria horas
de CPU e trocaria um modelo estável por outro sem ganho comprovado.

GOVERNANÇA: o modelo treinado aqui é registrado como **staging**, nunca
promovido a produção automaticamente. Promover é um ato deliberado, feito
depois de comparar o AUC novo com o vigente. Foi assim que o schema foi
desenhado: um índice único parcial impede dois modelos em produção ao mesmo
tempo.
"""

from airflow import DAG

from _comum import DEFAULT_ARGS, INICIO, tarefa

with DAG(
    dag_id="credit_risk_training",
    description="Treina o LightGBM e registra a versao (sem promover a producao)",
    default_args=DEFAULT_ARGS,
    schedule=None,               # acionada sob demanda
    start_date=INICIO,
    catchup=False,
    max_active_runs=1,
    tags=["credit-risk", "modelo"],
) as dag:

    train = tarefa(
        "train",
        "python -m Model.train",
    )

    # Lê os metadados do modelo recém-treinado e grava a versão no
    # mlops.model_registry. Sem este passo, uma decisão tomada por este modelo
    # não teria a que se referir no log de auditoria.
    register_model = tarefa(
        "register_model",
        'python -c "'
        "from Model.predict import model_metadata; "
        "from MLOps import store; "
        "print('versao registrada:', store.ensure_model_registered(model_metadata()))"
        '"',
    )

    train >> register_model
