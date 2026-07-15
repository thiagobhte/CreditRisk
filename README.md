# 🏦 Home Credit Default Risk

Modelo de classificação binária para prever **inadimplência de clientes de crédito**.
Usa **LightGBM** com **K-Fold Cross Validation** e centenas de features engenheiradas
a partir das **8 tabelas** do dataset Home Credit — indo da limpeza dos dados brutos
até o **deploy do modelo como serviço de predição** (API + painel + orquestração).

> **Resultado:** AUC-ROC de **0,79** (out-of-fold), superando o baseline de Regressão
> Logística (0,77). O modelo, além de prever, **explica cada decisão** (SHAP) e a
> traduz em uma recomendação de negócio: **aprovar / analisar / recusar**.

---

## 📋 Descrição do projeto

Este projeto simula o ciclo real de um projeto de Machine Learning dentro de uma
instituição financeira, seguindo o método **CRISP-DM**. Ele cobre a jornada completa:
da **limpeza dos dados brutos**, passando pela **construção da ABT (Analytical Base
Table)**, **modelagem** (baseline → tuning → treino final) e **avaliação**, até o
**deploy** do modelo como um serviço de predição com API, painel visual e
orquestração via Airflow.

A ideia não é só "treinar um modelo", mas conectar **negócio + dados + tecnologia +
decisão** — do CSV bruto à decisão de crédito automatizada.

---

## 🎯 Objetivo de negócio

Identificar, no momento da concessão, os clientes com maior probabilidade de **não
pagar um empréstimo** (`TARGET = 1`). Isso permite que a instituição:

- **reduza a inadimplência** recusando (ou revisando) os casos de alto risco;
- **preserve receita e inclusão** aprovando bons pagadores que seriam recusados por regras manuais;
- **decida com transparência** — cada decisão vem acompanhada do *porquê* (explicabilidade), exigência de conformidade em crédito.

O grande desafio do problema é o **desbalanceamento**: apenas **~8%** dos clientes
são inadimplentes. Prever "todo mundo paga" acerta 92% — e é inútil. Por isso a
métrica-guia é o **AUC-ROC**, não a acurácia.

---

## 🧭 Metodologia (o passo a passo, em linguagem de gente)

1. **Sanitização** (`DataPipeline/data_sanitization.py`) — a "faxina" dos dados.
   Lê os CSVs brutos, remove registros inválidos (ex.: gênero `XNA`), aplica encoding
   binário e one-hot nas variáveis de texto, trata valores sentinela (o famoso
   `DAYS_EMPLOYED = 365243`, que significa "sem emprego", e não "empregado há mil anos")
   e salva o `clean_data.csv`.

2. **Construção da ABT** (`DataPipeline/abt_transform.py`) — de 8 tabelas a 1 visão por cliente.
   Agrega as tabelas secundárias (bureau, aplicações anteriores, cartão, parcelas...)
   ao nível do cliente com estatísticas (min, max, mean, sum, var) e cria features
   derivadas de negócio: razões renda/crédito, taxa de pagamento, dias de atraso,
   severidade e tendência de inadimplência no bureau. Resultado: a **`abt.csv` com 838 features**.

3. **Modelagem** (`Model/`) — do simples ao campeão.
   - `baseline.py`: treina uma **Regressão Logística** (com imputação e padronização)
     como régua de comparação. Se o modelo complexo não superar isto, não vale a pena.
   - `tune.py`: busca os melhores hiperparâmetros do LightGBM com **Optuna** e salva em `best_params.json`.
   - `train.py`: treina o **LightGBM final** com **K-Fold estratificado**, mede a
     performance sem vazamento (out-of-fold), **persiste o modelo** em `Model/artifacts/`
     e gera as predições e a importância das features.

4. **Avaliação e análise** (notebooks) — a hora de olhar o modelo com olhar crítico e responder, com números na mão, à pergunta que todo mundo faz: *"dá mesmo para confiar nele?"*.
   - `DataPipeline/exp_analysis.ipynb`: análise exploratória dos dados limpos.
   - `Model/evaluation.ipynb` e `evaluation_part2.ipynb`: performance do modelo (AUC por fold, ROC, calibração, importância).
   - `Model/kpi_analysis.ipynb`: traduz as predições em **R$** (inadimplência evitada, resultado líquido, corte ótimo).

5. **Deploy e MLOps** (`Model/predict.py`, `app/`, `MLOps/`) — colocando de pé.
   O modelo treinado é servido por uma **API (FastAPI)** e um **painel (Streamlit)**,
   orquestrado pelo **Airflow** e vigiado por um **monitoramento de drift (PSI)**.

---

## 📁 Estrutura do projeto

```
CreditRisk/
├── Dados/
│   ├── README.md                  → estrutura da camada de dados (CSVs não versionados)
│   ├── raw_data/                  → CSVs brutos do dataset (Kaggle)
│   ├── clean_data.csv             → gerado por data_sanitization.py
│   └── abt.csv                    → gerado por abt_transform.py
│
├── DataPipeline/
│   ├── data_sanitization.py       → limpeza e padronização dos dados brutos
│   ├── abt_transform.py           → construção da ABT com features agregadas
│   └── exp_analysis.ipynb         → análise exploratória dos dados limpos
│
├── Model/
│   ├── baseline.py                → baseline de comparação (Regressão Logística)
│   ├── tune.py                    → otimização de hiperparâmetros (Optuna)
│   ├── train.py                   → treino final (K-Fold) + persistência do modelo
│   ├── predict.py                 → serviço de predição (inferência) + CLI
│   ├── explain.py                 → explicabilidade com SHAP (global e por cliente)
│   ├── evaluation.ipynb           → avaliação do modelo (AUC, ROC, calibração)
│   ├── evaluation_part2.ipynb     → avaliação complementar / interpretabilidade
│   ├── kpi_analysis.ipynb         → análise de KPIs de negócio
│   └── artifacts/                 → modelo treinado + features + metadados (gerado por train.py)
│
├── app/
│   ├── main.py                    → API REST de predição (FastAPI)
│   └── streamlit_app.py           → painel de decisão com explicabilidade (SHAP)
│
├── MLOps/
│   ├── README.md                  → arquitetura, monitoramento e ações automatizadas
│   ├── docker-compose.yml         → infraestrutura (airflow + API + Streamlit + pipeline + monitoramento)
│   ├── Dockerfile.offline         → build sem internet (redes corporativas)
│   ├── airflow.Dockerfile         → imagem do Airflow com as dependências de ML
│   ├── requirements-airflow.txt   → dependências das tasks da DAG
│   ├── pipeline_orchestration.py  → DAG do Airflow + orquestração standalone
│   └── monitoring.py              → monitoramento de drift de dados (PSI)
│
├── Dockerfile                     → imagem única (API + pipeline)
├── config.py                      → variáveis, caminhos e parâmetros globais do projeto
├── requirements.txt               → dependências do projeto
├── best_params.json               → hiperparâmetros otimizados (gerado por tune.py)
└── README.md
```

> **Nota sobre o `config.py`:** o projeto usa **um único** arquivo de configuração
> compartilhado na raiz (importado por `DataPipeline/` e `Model/`), em vez de
> duplicá-lo em cada pasta — evita divergência entre cópias e mantém uma única
> fonte da verdade.

---

## ⚙️ Instalação

```bash
# Clone o repositório e entre na pasta do projeto
cd CreditRisk

# Crie e ative um ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Instale as dependências
pip install -r requirements.txt
```

> Os CSVs brutos do Kaggle devem ficar em `Dados/raw_data/`. Por padrão o projeto já
> aponta para lá (`DATA_DIR = Dados/`), então não é preciso configurar nada.

---

## 🚀 Como treinar o modelo (do zero ao modelo salvo)

> ⚠️ **Ponto de atenção:** rode **sempre a partir da raiz do projeto** e com o modo
> módulo (`python -m ...`). O `config.py` está na raiz, e é assim que os imports funcionam.

### 1️⃣ Sanitizar os dados brutos
```bash
python -m DataPipeline.data_sanitization
```
Faz a faxina inicial e gera o `Dados/clean_data.csv` (~40s).

### 2️⃣ Construir a ABT (Analytical Base Table)
```bash
python -m DataPipeline.abt_transform
```
Junta as 8 tabelas numa visão por cliente e gera o `Dados/abt.csv` — a entrada da
modelagem (~3 min).

### 3️⃣ Treinar o baseline
```bash
python -m Model.baseline
```
Roda a **Regressão Logística** de comparação. Guarde o AUC dela: é a régua que o
LightGBM precisa superar.

### 4️⃣ Otimizar hiperparâmetros
```bash
python -m Model.tune
```
Usa o **Optuna** para procurar a melhor combinação e atualiza o `best_params.json`.
É a etapa mais longa, mas é ela que garante os melhores hiperparâmetros para o treino final.

### 5️⃣ Treinar o modelo final
```bash
python -m Model.train
```
Lê a `abt.csv` + `best_params.json`, treina o **LightGBM com K-Fold** e entrega:
- `Dados/submission.csv` — predições do conjunto de teste;
- `Dados/oof_predictions.csv` — predições out-of-fold (a base honesta para as métricas);
- `Model/artifacts/` — o **modelo salvo** (`.joblib`) + features + metadados, prontos para o serviço de predição;
- `feature_importance.csv` e `lgbm_importances.png` — a importância das features.

### 6️⃣ Avaliar o modelo
Abra os notebooks (em `Model/`):
```bash
jupyter notebook Model/evaluation.ipynb
jupyter notebook Model/kpi_analysis.ipynb
```

---

## 📊 Resultado do modelo

| Métrica | Valor |
|---|---|
| AUC-ROC (out-of-fold) | **0,7909** |
| Baseline — Regressão Logística | 0,7730 → LightGBM ganha **+0,018** |
| Consistência entre folds | 0,786 – 0,794 (sem overfitting) |
| Features na ABT | 838 |
| Validação | K-Fold estratificado (5 folds) + early stopping |

---

## 🔮 Serviço de predição (inferência)

Depois do `python -m Model.train`, o modelo fica salvo em `Model/artifacts/` e pode
ser servido de três formas.

### CLI — predição em lote
```bash
# A partir de um CSV no formato da ABT (features já engenheiradas)
python -m Model.predict --input Dados/abt.csv --output scores.csv

# Ou de um único cliente em JSON
python -m Model.predict --input cliente.json
```
A saída traz, por cliente: `probability_default` (PD), `risk_band`
(BAIXO/MODERADO/ALTO/MUITO_ALTO) e `decision` (APROVAR/ANALISE_MANUAL/RECUSAR).

### Painel visual (Streamlit + SHAP)
```bash
streamlit run app/streamlit_app.py     # → http://localhost:8501
```
Mostra, para o cliente escolhido, a probabilidade de inadimplência, a decisão
recomendada e — o ponto central — **por que** o modelo decidiu assim, variável a
variável (valores SHAP). Dá até para simular alterações (renda, crédito, scores
externos) e ver a decisão mudar na hora.

> **Explicabilidade:** em crédito, recusar um cliente sem saber justificar é um
> problema de conformidade, não só de modelagem. O SHAP responde à pergunta que a
> importância média do LightGBM não responde: *"por que **este** cliente?"*.

### API REST (FastAPI)
```bash
uvicorn app.main:app --reload --port 8000
```
- Documentação interativa (Swagger): http://localhost:8000/docs
- `GET /health` — status do serviço + metadados do modelo
- `POST /predict` — 1 cliente
- `POST /predict/batch` — vários clientes

Exemplo de requisição:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"SK_ID_CURR": 100002, "AMT_CREDIT": 406597.5, "AMT_INCOME_TOTAL": 202500.0, "PAYMENT_RATE": 0.06}'
```
> Não precisa enviar todas as features: as ausentes viram NaN (o LightGBM lida com
> isso nativamente). Campos vazios no JSON devem ir como `null`.

---

## 🐳 Infraestrutura e MLOps (Docker + Airflow)

O jeito recomendado de rodar tudo junto é via **docker-compose**:

```bash
# API de predição                →  http://localhost:8000/docs
docker compose -f MLOps/docker-compose.yml up -d --build api

# Painel Streamlit (SHAP)        →  http://localhost:8501
docker compose -f MLOps/docker-compose.yml up -d --build streamlit

# Airflow — orquestração com UI  →  http://localhost:8081  (admin / admin)
docker compose -f MLOps/docker-compose.yml up -d --build airflow

# Pipeline sem Airflow (job único)
docker compose -f MLOps/docker-compose.yml run --rm pipeline

# Monitoramento de drift de dados (PSI)
docker compose -f MLOps/docker-compose.yml run --rm monitoring

# Parar tudo
docker compose -f MLOps/docker-compose.yml down
```

### Airflow — a orquestração
A DAG **`credit_risk_pipeline`** (`sanitize → build_abt → train`) é definida em
[MLOps/pipeline_orchestration.py](MLOps/pipeline_orchestration.py) e carregada
automaticamente pelo Airflow. Acesse **http://localhost:8081** (login `admin/admin`),
ative a DAG e clique em **Trigger** para executá-la, acompanhando as tasks ficarem
verdes na aba *Graph*.

> A variável `NUM_ROWS=30000` no compose faz a DAG rodar sobre uma **amostra**
> (~2-3 min, ideal para demonstração) e num volume isolado, **sem sobrescrever** o
> modelo de produção. Remova-a para processar a base completa.

### Orquestração sem Docker
```bash
python -m MLOps.pipeline_orchestration                 # pipeline completo
python -m MLOps.pipeline_orchestration --with-tuning   # incluindo tuning
```

> Rede corporativa que bloqueia o container? Há um build offline com wheels locais
> em [MLOps/Dockerfile.offline](MLOps/Dockerfile.offline).

A **arquitetura completa da solução**, a estratégia de **monitoramento** e as
**ações automatizadas + agentes de IA** estão detalhadas em
[MLOps/README.md](MLOps/README.md).

---

## 📦 Principais arquivos gerados

| Arquivo | Descrição |
|---|---|
| `clean_data.csv` | Dados limpos e padronizados da tabela principal |
| `abt.csv` | Tabela analítica completa (todas as features) |
| `best_params.json` | Hiperparâmetros otimizados do LightGBM |
| `submission.csv` | Predições do modelo final para o conjunto de teste |
| `oof_predictions.csv` | Predições out-of-fold (base honesta das métricas) |
| `feature_importance.csv` | Importância de features do modelo final |
| `Model/artifacts/lgbm_model.joblib` | Modelo final treinado (usado pelo serviço de predição) |
| `Model/artifacts/model_features.json` | Ordem exata das features esperadas pelo modelo |
| `Model/artifacts/model_metadata.json` | AUC, data do treino, nº de features e política de decisão |

---

## 🔗 Repositório

Github: https://github.com/thiagobhte/CreditRisk
