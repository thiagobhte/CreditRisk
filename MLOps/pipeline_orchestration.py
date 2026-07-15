"""
pipeline_orchestration.py — Orquestração ponta-a-ponta do pipeline de ML.

Encadeia as etapas na ordem correta:

    raw_data ─► data_sanitization ─► abt_transform ─► (tune) ─► train ─► modelo

Dois modos de execução:

1. STANDALONE (default) — roda tudo em sequência num único processo.
   Ideal para rodar localmente, dentro de um container ou como um job agendado
   simples (cron). Cada etapa só roda se a anterior tiver sucesso.

       python -m MLOps.pipeline_orchestration
       python -m MLOps.pipeline_orchestration --with-tuning --trials 30

2. AIRFLOW — o mesmo grafo de dependências exposto como uma DAG. O Airflow
   dá agendamento, retries, backfill e observabilidade que o modo standalone
   não tem. A DAG só é definida se o pacote `airflow` estiver instalado, então
   este arquivo continua importável sem o Airflow.

Cada etapa é uma função `run()` já existente nos módulos do projeto — aqui só
as encadeamos, sem reimplementar nada.
"""

import argparse
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# DEFINIÇÃO DAS ETAPAS
# ============================================================
# Cada etapa é (nome, callable). As funções são importadas dentro do wrapper
# para não pagar o custo de import (pandas, lightgbm) se a etapa não rodar.

def step_sanitize():
    from DataPipeline.data_sanitization import run
    run()


def step_build_abt():
    from DataPipeline.abt_transform import run
    run()


def step_tune(trials: int = 50):
    from Model.tune import run
    run(n_trials=trials)


def step_train():
    from Model.train import run
    run()


# ============================================================
# ORQUESTRAÇÃO STANDALONE
# ============================================================

def run_pipeline(with_tuning: bool = False, trials: int = 50):
    """
    Executa o pipeline completo em sequência.

    Para no primeiro erro (fail-fast): não faz sentido construir a ABT se a
    sanitização falhou, nem treinar sobre uma ABT incompleta.
    """
    steps = [
        ("Sanitização dos dados", step_sanitize),
        ("Construção da ABT",     step_build_abt),
    ]
    if with_tuning:
        steps.append(("Tuning de hiperparâmetros", lambda: step_tune(trials)))
    steps.append(("Treino + persistência do modelo", step_train))

    # Só ASCII nas mensagens: o console do Windows usa cp1252 e estoura
    # UnicodeEncodeError em caracteres como "→", "✅" e "❌".
    print("=" * 60)
    print("PIPELINE DE ML - Home Credit Default Risk")
    print(f"Etapas: {' -> '.join(name for name, _ in steps)}")
    print("=" * 60)

    t_start = time.time()
    for i, (name, fn) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {name} ...")
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            print(f"\n[FALHA] etapa '{name}': {e}")
            raise
        print(f"[OK] '{name}' concluida em {time.time() - t0:.0f}s")

    print(f"\n{'=' * 60}")
    print(f"Pipeline concluído com sucesso em {time.time() - t_start:.0f}s")
    print("=" * 60)


# ============================================================
# DAG DO AIRFLOW (opcional — só se airflow estiver instalado)
# ============================================================
# Definir a DAG no nível do módulo é o padrão do Airflow: o scheduler importa
# este arquivo e encontra o objeto `dag`. Protegemos com try/except para que
# o modo standalone funcione em ambientes sem Airflow.

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from datetime import datetime, timedelta

    default_args = {
        "owner": "labdata",
        "retries": 2,                          # re-executa etapas transitoriamente falhas
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="credit_risk_pipeline",
        description="Sanitização → ABT → Treino do modelo de risco de crédito",
        default_args=default_args,
        schedule="@daily",                     # re-treina diariamente (ajuste conforme drift)
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["credit-risk", "ml"],
    ) as dag:

        t_sanitize = PythonOperator(task_id="sanitize",  python_callable=step_sanitize)
        t_abt      = PythonOperator(task_id="build_abt", python_callable=step_build_abt)
        t_train    = PythonOperator(task_id="train",     python_callable=step_train)

        # Encadeia as dependências: define o grafo (a "seta" do diagrama)
        t_sanitize >> t_abt >> t_train

except ImportError:
    # Airflow não instalado — modo standalone continua funcionando normalmente.
    dag = None


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestra o pipeline de ML de ponta a ponta")
    parser.add_argument("--with-tuning", action="store_true",
                        help="Inclui a etapa de tuning (Optuna) antes do treino")
    parser.add_argument("--trials", type=int, default=50,
                        help="Nº de trials do Optuna, se --with-tuning (padrão: 50)")
    args = parser.parse_args()

    run_pipeline(with_tuning=args.with_tuning, trials=args.trials)
