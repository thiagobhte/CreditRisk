-- ============================================================
-- 00_airflow_metastore.sql — Banco de metadados do Airflow.
--
-- O prefixo "00_" garante que este arquivo rode ANTES do schema.sql: o
-- Postgres executa os scripts de /docker-entrypoint-initdb.d em ordem
-- alfabética, na primeira subida do container.
--
-- POR QUE O AIRFLOW PRECISA DE UM BANCO PRÓPRIO:
--   o Airflow guarda em banco todo o seu estado — DAGs registradas, execuções,
--   estado de cada task, logs, conexões, usuários. Rodando com SQLite (o
--   default do `airflow standalone`), esse estado vive num arquivo dentro do
--   container: ele some quando o container é recriado, e o SQLite só admite o
--   SequentialExecutor, que roda UMA task por vez — o grafo em leque da DAG
--   vira uma fila.
--
--   Com Postgres, o histórico sobrevive ao `docker compose down` e o
--   LocalExecutor pode rodar tasks independentes em paralelo.
--
-- ISOLAMENTO: é um BANCO separado (`airflow`) dentro da mesma instância. Os
-- dados da solução de crédito ficam no banco `creditrisk`; o Airflow não
-- enxerga um, nem o outro enxerga o dele. Compartilhar a instância é uma
-- escolha de custo, adequada a este porte — em produção seriam servidores
-- distintos, para que uma carga pesada de orquestração não dispute recursos
-- com as consultas do serviço de predição.
-- ============================================================

CREATE ROLE airflow WITH LOGIN PASSWORD 'airflow';

CREATE DATABASE airflow OWNER airflow;

COMMENT ON DATABASE airflow IS 'Metadados do Apache Airflow (execucoes, estado das tasks, usuarios)';
