# ============================================================
# Imagem do Airflow para orquestrar o pipeline de risco de crédito.
#
# Parte da imagem oficial e acrescenta APENAS as dependências que as tasks da
# DAG precisam (pandas, lightgbm, ...). Não instalamos o requirements.txt
# inteiro do projeto de propósito: fastapi/uvicorn/optuna não são usados pela
# DAG e só aumentariam a chance de conflito com as dependências do Airflow.
# ============================================================
FROM apache/airflow:2.10.5-python3.11

# libgomp1: runtime OpenMP exigido pelo LightGBM (a imagem base não o traz).
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# /demo é onde a DAG grava clean_data, abt e o modelo da demonstração.
# Criamos o diretório JÁ com o dono correto: ao montar um volume nomeado vazio
# aqui, o Docker inicializa o volume a partir do conteúdo da imagem e preserva
# essa permissão. Sem isso o volume nasce como root e o usuário `airflow` leva
# "Permission denied" ao tentar escrever.
RUN mkdir -p /demo && chown -R airflow:root /demo && chmod -R 775 /demo

# pip deve rodar como o usuário airflow (a imagem oficial exige isso)
USER airflow

COPY MLOps/requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt
