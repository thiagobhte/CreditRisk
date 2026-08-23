"""
pipeline_orchestration.py — Orquestração ponta-a-ponta do pipeline de ML.

Encadeia as etapas na ordem correta:

    raw_data ─► data_sanitization ─► abt_transform ─┬─► (tune) ─► train ─► modelo
                                                    └─► load_to_db ─► PostgreSQL

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


def step_load_db(limite="auto"):
    """
    Publica a ABT recém-construída no PostgreSQL (feature store).

    É o elo que faltava entre o pipeline e o serviço de predição: sem esta
    etapa, a API não teria de onde ler as features de um cliente. Roda depois
    da ABT e em paralelo ao treino — uma coisa não depende da outra.

    O default é "auto" (e não None) de propósito: o Airflow chama esta função
    sem argumentos, e um None seria interpretado como "carregue a base inteira"
    — 356 mil clientes, mais de 5 GB no banco, no meio de uma demonstração.
    Com "auto", vale o ABT_LOAD_LIMIT do ambiente. Passe None explicitamente
    para carregar tudo.
    """
    from MLOps.load_to_db import load_abt, load_clean, DEFAULT_LIMIT
    if limite == "auto":
        limite = DEFAULT_LIMIT
    # Sem truncate de proposito: a carga e um UPSERT. Uma execucao de
    # demonstracao atualiza a amostra e preserva a base completa ja carregada.
    load_clean(limite=limite)
    load_abt(limite=limite)


# ============================================================
# ORQUESTRAÇÃO STANDALONE
# ============================================================

def run_pipeline(with_tuning: bool = False, trials: int = 50, db_limit="auto"):
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
    steps.append(("Carga da ABT no PostgreSQL", lambda: step_load_db(db_limit)))

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
# E O AIRFLOW?
# ============================================================
# As DAGs vivem em MLOps/dags/, separadas desta lógica de propósito.
#
# Antes a DAG era declarada aqui dentro, protegida por try/except ImportError.
# Funcionava, mas misturava duas coisas: o QUE fazer (as etapas) e QUANDO/COMO
# orquestrar. Pior, o scheduler do Airflow importa todo arquivo da pasta de
# DAGs a cada varredura — e importar este módulo arrastava pandas e lightgbm
# junto, a cada poucos segundos.
#
# Agora as etapas continuam aqui, chamáveis pela CLI, e cada DAG em
# MLOps/dags/ apenas invoca o comando correspondente.

# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestra o pipeline de ML de ponta a ponta")
    parser.add_argument("--with-tuning", action="store_true",
                        help="Inclui a etapa de tuning (Optuna) antes do treino")
    parser.add_argument("--trials", type=int, default=50,
                        help="Nº de trials do Optuna, se --with-tuning (padrão: 50)")
    parser.add_argument("--db-limit", type=int, default=None,
                        help="Quantos clientes publicar no banco (padrão: ABT_LOAD_LIMIT)")
    parser.add_argument("--db-full", action="store_true",
                        help="Publica a base COMPLETA no banco (lento, +5 GB)")
    args = parser.parse_args()

    db_limit = None if args.db_full else (args.db_limit if args.db_limit is not None else "auto")
    run_pipeline(with_tuning=args.with_tuning, trials=args.trials, db_limit=db_limit)
