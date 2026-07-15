# MLOps — Arquitetura da Solução de Risco de Crédito

Este documento descreve a **arquitetura funcional completa** da solução, do dado
bruto ao deploy do modelo como serviço de predição, além da estratégia de
**monitoramento** e das **ações automatizadas** disparadas pelas previsões.

---

## 1. Visão geral da arquitetura

```
   FONTES                    PIPELINE (orquestrado)                   SERVING
 ┌─────────┐        ┌──────────────────────────────────┐        ┌──────────────┐
 │ raw_data│        │ data_sanitization ─► abt_transform│        │  FastAPI     │
 │  (CSVs) │──────► │        │                    │     │        │  /predict    │
 └─────────┘        │        ▼                    ▼     │        │  /health     │
                    │   clean_data.csv         abt.csv  │──────► │              │
                    │                             │     │        │  modelo      │
                    │                    tune ─► train  │        │ (joblib)     │
                    │                             │     │        └──────┬───────┘
                    │                    Model/artifacts│               │
                    │                  (modelo + meta)  │               ▼
                    └──────────────────────────────────┘        decisão de crédito
                                   ▲                              (APROVAR/ANÁLISE/
                                   │                                  RECUSAR)
                          ┌────────┴────────┐
                          │  monitoring.py  │◄──── dados de produção
                          │  (drift / PSI)  │
                          └────────┬────────┘
                                   │ drift severo → gatilho de re-treino
                                   ▼
                            (volta ao train)
```

### Componentes

| Componente | Papel | Arquivo |
|---|---|---|
| **Ingestão / Sanitização** | Lê os CSVs brutos, limpa e padroniza | [`DataPipeline/data_sanitization.py`](../DataPipeline/data_sanitization.py) |
| **Feature Engineering (ABT)** | Agrega tabelas e cria features | [`DataPipeline/abt_transform.py`](../DataPipeline/abt_transform.py) |
| **Tuning** | Busca de hiperparâmetros (Optuna) | [`Model/tune.py`](../Model/tune.py) |
| **Treino** | Treina, avalia (K-Fold) e **persiste** o modelo | [`Model/train.py`](../Model/train.py) |
| **Predição (lógica)** | Alinha features e traduz PD → decisão | [`Model/predict.py`](../Model/predict.py) |
| **API (serving)** | Expõe o modelo via HTTP | [`app/main.py`](../app/main.py) |
| **Orquestração** | Encadeia o pipeline (standalone / Airflow) | [`pipeline_orchestration.py`](pipeline_orchestration.py) |
| **Monitoramento** | Detecta drift de dados (PSI) | [`monitoring.py`](monitoring.py) |
| **Infra** | Containeriza e sobe tudo | [`docker-compose.yml`](docker-compose.yml), [`../Dockerfile`](../Dockerfile) |

O modelo é persistido em `Model/artifacts/` (`lgbm_model.joblib`,
`model_features.json`, `model_metadata.json`) — esses artefatos são o contrato
entre o **treino** e o **serving**.

---

## 2. Como subir a infraestrutura (docker-compose)

Os dados são montados por **bind-mount** de `Dados/` para `/data` dentro dos
containers — basta ter os CSVs brutos em `Dados/raw_data/`.

```bash
# API de predição        →  http://localhost:8000/docs
docker compose -f MLOps/docker-compose.yml up -d --build api

# Painel Streamlit (SHAP) →  http://localhost:8501
docker compose -f MLOps/docker-compose.yml up -d --build streamlit

# Airflow (orquestração)  →  http://localhost:8081   (admin / admin)
docker compose -f MLOps/docker-compose.yml up -d --build airflow

# Pipeline sem Airflow (job único)
docker compose -f MLOps/docker-compose.yml run --rm pipeline

# Monitoramento de drift (job)
docker compose -f MLOps/docker-compose.yml run --rm monitoring

# Parar tudo
docker compose -f MLOps/docker-compose.yml down
```

### Orquestração sem Docker
```bash
python -m MLOps.pipeline_orchestration                 # pipeline completo
python -m MLOps.pipeline_orchestration --with-tuning   # incluindo tuning
```

### Se o build falhar por rede (ambiente corporativo)

Em algumas redes o host tem internet mas os **containers não** — nem o `ping` sai
(agentes de segurança que filtram o tráfego de máquinas virtuais). O `pip install`
do build quebra com timeout.

Saída: baixe os pacotes no host e construa offline.

```bash
# 1. No host (que tem internet), baixe os wheels Linux
pip download -r requirements.txt -d wheelhouse \
  --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 \
  --python-version 3.11 --only-binary=:all:

# 2. Construa sem acessar a internet de dentro do container
docker build -f MLOps/Dockerfile.offline -t credit-risk:latest .
```

Como diagnosticar se é esse o caso:
```bash
docker run --rm busybox ping -c 2 8.8.8.8   # se falhar, o container não tem rede
```

---

## 2.1. Airflow — orquestração

A DAG **`credit_risk_pipeline`** vive em [`pipeline_orchestration.py`](pipeline_orchestration.py)
e é carregada pelo Airflow direto da pasta do projeto
(`AIRFLOW__CORE__DAGS_FOLDER=/project/MLOps`).

**Grafo de tasks:**

```
sanitize  ──►  build_abt  ──►  train
```

| Task | O que faz |
|---|---|
| `sanitize` | Lê `Dados/raw_data/*.csv` → grava `clean_data.csv` |
| `build_abt` | Agrega as 7 tabelas → grava `abt.csv` |
| `train` | Treina o LightGBM (K-Fold) e persiste o modelo em `Model/artifacts/` |

**Como demonstrar:**
1. `docker compose -f MLOps/docker-compose.yml up -d --build airflow`
2. Abra `http://localhost:8081` e faça login (**admin / admin**)
3. Ative a DAG `credit_risk_pipeline` e clique em **Trigger**
4. Acompanhe as tasks ficando verdes na aba *Graph*

> **Modo demo:** a variável `NUM_ROWS=30000` no compose faz a DAG rodar sobre
> uma **amostra** (~2 min). Remova-a para processar a base completa (~15 min) —
> tempo demais para uma apresentação ao vivo.

**Sobre a arquitetura escolhida:** o container roda `airflow standalone`
(SQLite + SequentialExecutor), suficiente para demonstrar a orquestração. Em
produção, trocaríamos por **Postgres como metastore** e **CeleryExecutor ou
KubernetesExecutor** para paralelizar as tasks e escalar horizontalmente.

O mesmo pipeline também roda **sem Airflow**, em modo sequencial
(`python -m MLOps.pipeline_orchestration`) — útil em CI ou num cron simples.

---

## 3. Monitoramento em produção (item iii)

Um modelo de crédito degrada **sem ninguém mexer nele**, porque o mundo muda
(inflação, sazonalidade, novo público). Monitoramos três camadas:

### a) Saúde do serviço (infra)
- `/health` retorna a versão/AUC do modelo servido; usado pelo `healthcheck` do
  compose e por um orquestrador (Kubernetes) para *liveness/readiness*.
- Métricas de infra: latência p95 da API, taxa de erro HTTP, throughput.

### b) Drift de dados ([`monitoring.py`](monitoring.py))
- **PSI (Population Stability Index)** por feature, comparando produção contra o
  treino (referência): `PSI > 0.25` = drift severo.
- **Prediction drift**: acompanha a taxa média de default prevista ao longo do
  tempo. Um salto sinaliza problema mesmo **antes** de termos o label real.

### c) Performance do modelo (com *label lag*)
- No crédito, o TARGET real (inadimplência) só se materializa meses depois. Por
  isso monitoramos **proxies antecipados** (drift + prediction drift) e, quando
  os labels chegam, recalculamos AUC/KS sobre as safras já maduras.

**Gatilho de re-treino:** `monitoring_report()` devolve `retrain_recommended=True`
quando há drift severo → a DAG do Airflow (ou um cron) dispara o
`pipeline_orchestration` novamente.

---

## 4. Ações automatizadas + agentes de IA (item iv)

A previsão não é o fim — ela **aciona decisões de negócio**:

| Faixa (PD) | Decisão automática | Ação acionada |
|---|---|---|
| `< 0.08` (BAIXO) | **APROVAR** | Crédito liberado automaticamente; e-mail de boas-vindas |
| `0.08–0.30` (MODERADO/ALTO) | **ANÁLISE MANUAL** | Cria ticket para o analista com as features de maior peso (SHAP) |
| `> 0.30` (MUITO ALTO) | **RECUSAR** | Recusa automática + oferta alternativa (limite menor / garantia) |

### Agentes de IA no fluxo
- **Agente de explicação:** para casos de análise manual, um LLM recebe as
  top features (SHAP) e gera um resumo em linguagem natural do *porquê* do risco,
  acelerando a decisão do analista e apoiando a **explicabilidade regulatória**.
- **Agente de retenção:** para clientes recusados de baixo-moderado risco, propõe
  automaticamente um produto alternativo viável.
- **Agente de monitoramento:** interpreta o `drift_report.json`, resume em
  linguagem natural quais features driftaram e abre um incidente com a
  recomendação (re-treinar / investigar fonte de dados).

Todas as decisões automáticas são **logadas** (entrada, PD, decisão, versão do
modelo) para auditoria e conformidade (governança).
