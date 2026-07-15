#!/usr/bin/env bash
# Sobe o Airflow em modo demo: migra o banco, cria o usuário admin e então
# levanta scheduler + webserver.
#
# Por que um script e não um "command" inline no compose:
#   encadear isso no YAML com && e & é frágil — o "&" acabava mandando o
#   webserver subir ANTES do "db migrate" terminar, e ele morria com
#   "You need to initialize the database".
set -e

echo ">>> Migrando o banco de metadados do Airflow..."
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
