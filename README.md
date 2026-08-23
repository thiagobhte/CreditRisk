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

O escopo não se limita ao treinamento do modelo: integra **negócio, dados, tecnologia
e tomada de decisão**, cobrindo o fluxo do dado bruto à decisão de crédito automatizada.

---

## 🎯 Objetivo de negócio

Identificar, no momento da concessão, os clientes com maior probabilidade de **não
pagar um empréstimo** (`TARGET = 1`). Isso permite que a instituição:

- **reduza a inadimplência** recusando (ou revisando) os casos de alto risco;
- **preserve receita e inclusão** aprovando bons pagadores que seriam recusados por regras manuais;
- **decida com transparência** — cada decisão vem acompanhada do *porquê* (explicabilidade), exigência de conformidade em crédito.

O principal desafio do problema é o **desbalanceamento**: apenas **~8%** dos clientes
são inadimplentes. Nesse cenário a acurácia é enganosa (um modelo que aprova todos
acerta 92% e não agrega valor), por isso a métrica-guia é o **AUC-ROC**.

---

## 🧭 Metodologia

1. **Sanitização** (`DataPipeline/data_sanitization.py`) — limpeza e padronização dos
   dados brutos. Carrega os CSVs, remove registros inválidos (ex.: gênero `XNA`), aplica
   encoding binário e one-hot nas variáveis categóricas, trata valores sentinela
   (ex.: `DAYS_EMPLOYED = 365243`, que representa "sem emprego") e gera o `clean_data.csv`.

2. **Construção da ABT** (`DataPipeline/abt_transform.py`) — consolidação das 8 tabelas
   em uma visão por cliente. Agrega as tabelas secundárias (bureau, aplicações anteriores,
   cartão, parcelas) ao nível do cliente com estatísticas (min, max, mean, sum, var) e cria
   features derivadas: razões renda/crédito, taxa de pagamento, dias de atraso, severidade
   e tendência de inadimplência no bureau. Resultado: a `abt.csv` com **838 features**.

3. **Modelagem** (`Model/`).
   - `baseline.py`: treina uma **Regressão Logística** (com imputação e padronização) como referência de comparação.
   - `tune.py`: otimiza os hiperparâmetros do LightGBM com **Optuna** e salva em `best_params.json`.
   - `train.py`: treina o **LightGBM final** com **K-Fold estratificado**, mede a
     performance out-of-fold (sem vazamento), persiste o modelo em `Model/artifacts/`
     e gera as predições e a importância das features.

4. **Avaliação e análise** (notebooks).
   - `DataPipeline/exp_analysis.ipynb`: análise exploratória dos dados limpos.
   - `Model/evaluation.ipynb` e `evaluation_part2.ipynb`: performance do modelo (AUC por fold, ROC, calibração, importância).
   - `Model/kpi_analysis.ipynb`: tradução das predições em indicadores de negócio (inadimplência evitada, resultado líquido, corte ótimo).

5. **Deploy e MLOps** (`Model/predict.py`, `app/`, `MLOps/`).
   A ABT é publicada numa **feature store em PostgreSQL**, de onde a **API
   (FastAPI)** lê as features de cada cliente na hora da decisão. O **painel
   (Streamlit)** consome a API, o **Airflow** orquestra as quatro DAGs
   (ingestão, treino, scoring em lote e monitoramento) e cada decisão fica
   registrada para auditoria. O monitoramento cobre três camadas: drift de
   dados (PSI), drift de predições e performance por safra (AUC/KS/Gini).

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
│   ├── streamlit_app.py           → painel de decisão com explicabilidade (SHAP)
│   └── pages/1_Monitoramento.py   → painel de monitoramento (drift e performance)
│
├── MLOps/
│   ├── README.md                  → arquitetura (diagramas), monitoramento e ações automatizadas
│   ├── sql/schema.sql             → modelo de dados: 4 schemas, 7 tabelas, 2 visões
│   ├── db.py                      → conexão única com o PostgreSQL
│   ├── store.py                   → leitura da feature store e log de decisões
│   ├── load_to_db.py              → publica a ABT no banco (COPY + UPSERT)
│   ├── batch_scoring.py           → pontuação da carteira em lote
│   ├── simulate_production.py     → gera tráfego para o monitoramento observar
│   ├── data_dictionary.py         → gera o dicionário de dados a partir do banco
│   ├── dags/                      → as 4 DAGs do Airflow
│   ├── docker-compose.yml         → infraestrutura (postgres + airflow + API + Streamlit + jobs)
│   ├── Dockerfile.offline         → build sem internet (redes corporativas)
│   ├── airflow.Dockerfile         → imagem do Airflow com as dependências de ML
│   ├── requirements-airflow.txt   → dependências das tasks da DAG
│   ├── pipeline_orchestration.py  → orquestração standalone (sem Airflow)
│   └── monitoring.py              → drift de dados, drift de predições e performance
│
├── DICIONARIO_DE_DADOS.md         → dicionário gerado a partir do catálogo do banco
├── Dockerfile                     → imagem única (API + pipeline)
├── config.py                      → variáveis, caminhos e parâmetros globais do projeto
├── requirements.txt               → dependências do projeto
├── best_params.json               → hiperparâmetros otimizados (gerado por tune.py)
└── README.md
```

> **Nota sobre o `config.py`:** o projeto usa **um único** arquivo de configuração
> na raiz, importado por `DataPipeline/` e `Model/`.

---

## ⚙️ Instalação

```bash
# Clone o repositório e entre na pasta do projeto
cd CreditRisk

# Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Instale as dependências
pip install -r requirements.txt
```

> Os CSVs brutos do Kaggle devem ficar em `Dados/raw_data/`. Por padrão o projeto já
> aponta para lá (`DATA_DIR = Dados/`), então não é preciso configurar nada.

---

## 🚀 Como treinar o modelo

> ⚠️ **Ponto de atenção:** rode **sempre a partir da raiz do projeto** e com o modo
> módulo (`python -m ...`). O `config.py` está na raiz, e é assim que os imports funcionam.

### 1️⃣ Sanitizar os dados brutos
```bash
python -m DataPipeline.data_sanitization
```
Gera o `Dados/clean_data.csv` (~40s).

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
Treina a **Regressão Logística** de referência. O AUC dela é a base de comparação
que o LightGBM precisa superar.

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

| Endpoint | O que faz |
|---|---|
| `GET /health` | Saúde do serviço, do modelo e do banco |
| `GET /clients` | Lista IDs válidos na feature store |
| `GET /clients/{id}` | O que o banco sabe sobre o cliente |
| **`POST /predict/{id}`** | **Busca as features no banco, aplica simulações e pontua** |
| `POST /predict` | Payload completo (compatibilidade) |
| `POST /predict/batch` | Vários clientes |
| `GET /predictions/recent` | Últimas decisões registradas (auditoria) |

**De onde a API tira as features.** Um cliente tem 836 features, quase todas
agregações do histórico (bureau, aplicações anteriores, parcelas, cartão). Quem
consome a API é o sistema de originação, que conhece a proposta — renda, valor
pedido, prazo — e **não** o histórico consolidado. Por isso a API busca o
cliente na `feature_store.abt` a partir do ID:

```bash
# Pontua o cliente com o histórico que está no banco
curl -X POST http://localhost:8000/predict/100002

# Mesmo cliente, simulando um score externo melhor
curl -X POST http://localhost:8000/predict/100002 \
  -H "Content-Type: application/json" \
  -d '{"EXT_SOURCE_2": 0.85}'
```

A resposta declara a origem de cada coisa: `features_from_store` (quantas vieram
do banco), `features_overridden` (o que você alterou) e `derived_recalculated`
(as features derivadas que foram refeitas — alterar a renda sem recalcular a
razão parcela/renda entregaria ao modelo um cliente que não existe).

> **Por que isso importa.** O cliente 100002, enviado com 3 features na mão,
> recebe `PD 0,046 → APROVAR`. Buscado no banco, com as 658 features que ele
> realmente tem, recebe `PD 0,346 → RECUSAR` — e o desfecho real dele é
> inadimplência (`TARGET = 1`). O campo `n_features_missing` na resposta existe
> para tornar essa diferença visível em vez de silenciosa.

---

## 🐳 Infraestrutura e MLOps (Docker + Airflow)

A solução é containerizada e orquestrada. Subida rápida dos serviços via **docker-compose**:

```bash
docker compose -f MLOps/docker-compose.yml up -d --build api        # API     → http://localhost:8000/docs
docker compose -f MLOps/docker-compose.yml up -d --build streamlit  # Painel  → http://localhost:8501
docker compose -f MLOps/docker-compose.yml up -d --build airflow    # Airflow → http://localhost:8081  (admin/admin)
```

```bash
docker compose -f MLOps/docker-compose.yml up -d postgres        # banco da solução
```

👉 A **arquitetura completa** está em **[MLOps/README.md](MLOps/README.md)**, com
três diagramas: **componentes** (quem fala com quem), **sequência** (o que o
código faz do clique à decisão) e **modelo de dados** (as 7 tabelas). Lá também
estão a orquestração via **Airflow**, o **monitoramento** em três camadas e as
**ações automatizadas + agentes de IA**.

O **dicionário de dados** — cada tabela, coluna, chave e índice — está em
**[DICIONARIO_DE_DADOS.md](DICIONARIO_DE_DADOS.md)**, gerado a partir do
catálogo do banco:

```bash
python -m MLOps.data_dictionary --output DICIONARIO_DE_DADOS.md
```

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
