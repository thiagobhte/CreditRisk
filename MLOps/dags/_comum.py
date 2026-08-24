"""
_comum.py — Definições compartilhadas pelas DAGs do projeto.

Concentra aqui o que se repetiria em cada arquivo: os defaults, os caminhos e o
helper que monta uma task. Assim cada DAG fica com o que interessa — o grafo.

DUAS DECISÕES QUE VALEM EXPLICAR:

1. AS TASKS SÃO COMANDOS, NÃO FUNÇÕES IMPORTADAS.
   Cada task roda exatamente o mesmo comando de terminal que usamos na mão
   (`python -m MLOps.monitoring --tudo`). A DAG não faz nada diferente do que
   você consegue reproduzir fora dela — o que torna qualquer falha depurável
   sem o Airflow no meio. Também evita que um import pesado (lightgbm, shap)
   aconteça no processo do scheduler a cada varredura de DAGs.

2. O AMBIENTE MUDA POR TASK, DE PROPÓSITO.
   O container do Airflow roda por padrão em modo DEMO: escreve num volume
   isolado (/demo) e sobre uma amostra, para uma execução de demonstração
   jamais sobrescrever a ABT e o modelo de produção. Mas as tasks de scoring e
   de monitoramento precisam do modelo REAL e dos dados REAIS — senão estariam
   monitorando um brinquedo. Elas recebem AMBIENTE_PRODUCAO explicitamente.
"""

from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator

# Raiz do projeto dentro do container (bind-mount do repositório)
PROJETO = "/project"

# Ambiente de PRODUÇÃO: dados e modelo reais.
#   /data              → bind-mount de ./Dados (somente leitura)
#   Model/artifacts    → o modelo treinado na base completa (AUC 0,7909)
#   NUM_ROWS vazio     → base inteira, sem recorte
#
# LIMPAR `NUM_ROWS` É OBRIGATÓRIO, NÃO COSMÉTICO.
#   `append_env=True` herda o ambiente do container, que roda em modo demo com
#   NUM_ROWS=30000. Sem apagá-la aqui, uma task de produção continuaria em modo
#   amostra: a carga era recusada pela trava do load_to_db e a DAG falhava todo
#   dia às 00h. `config.py` trata string vazia como ausente.
AMBIENTE_PRODUCAO = {
    "DATA_DIR":  "/data",
    "MODEL_DIR": f"{PROJETO}/Model/artifacts",
    "NUM_ROWS":  "",
}

DEFAULT_ARGS = {
    "owner": "labdata",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    # Sem e-mail configurado: melhor não fingir que existe alerta por e-mail.
    "email_on_failure": False,
}

INICIO = datetime(2026, 1, 1)


def tarefa(task_id: str, comando: str, ambiente: dict = None, **kwargs) -> BashOperator:
    """
    Cria uma task que roda um comando do projeto a partir da raiz.

    `ambiente` sobrescreve variáveis só para esta task — é como as tasks de
    produção pedem o modelo real sem que o container inteiro saia do modo demo.
    """
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJETO} && {comando}",
        env=ambiente,
        append_env=True,   # herda o ambiente do container e sobrescreve o que vier
        **kwargs,
    )
