#!/usr/bin/env bash
# Sobe o Airflow: espera o Postgres, migra o metastore, cria o admin e então
# levanta scheduler + webserver.
#
# Por que um script e não um "command" inline no compose:
#   encadear isso no YAML com && e & é frágil — o "&" acabava mandando o
#   webserver subir ANTES do "db migrate" terminar, e ele morria com
#   "You need to initialize the database".
set -e

echo ">>> Esperando o Postgres aceitar conexoes..."
# O healthcheck do compose garante que o SERVIÇO está de pé, mas o banco
# `airflow` é criado pelos scripts de init — que rodam depois. Esperamos a
# conexão que de fato vamos usar.
for tentativa in $(seq 1 30); do
  if python -c "
import sys, psycopg2
try:
    psycopg2.connect(host='postgres', dbname='airflow', user='airflow', password='airflow').close()
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    echo ">>> Postgres pronto (banco 'airflow' acessivel)."
    break
  fi
  if [ "$tentativa" -eq 30 ]; then
    echo "!!! O banco 'airflow' nao respondeu."
    echo "!!! Se o volume do Postgres ja existia antes desta versao, os scripts de"
    echo "!!! init nao rodam de novo. Crie o banco uma vez com:"
    echo "!!!   docker compose -f MLOps/docker-compose.yml exec postgres \\"
    echo "!!!     psql -U creditrisk -d creditrisk -f /docker-entrypoint-initdb.d/00_airflow_metastore.sql"
    exit 1
  fi
  sleep 2
done

echo ">>> Migrando o metastore do Airflow..."
airflow db migrate

echo ">>> Criando usuario admin (ignora se ja existir)..."
airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com || true

echo ">>> Subindo o scheduler em background..."
airflow scheduler &

echo ">>> Subindo o webserver (processo principal)..."
exec airflow webserver
