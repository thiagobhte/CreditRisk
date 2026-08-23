# Dicionário de Dados — Solução de Risco de Crédito

> **Documento gerado**, não escrito à mão. Sai do catálogo do PostgreSQL
> (tipos, chaves, índices e os `COMMENT ON` do `MLOps/sql/schema.sql`).
> Para atualizar: `python -m MLOps.data_dictionary --output DICIONARIO_DE_DADOS.md`

Gerado em 23/08/2026 18:45.

---

## Visão geral

| Schema | Camada | Papel |
|---|---|---|
| `staging` | Camada silver | Dados limpos e padronizados, ainda no nível da tabela de origem. |
| `feature_store` | Camada gold | A ABT: uma linha por cliente, pronta para o modelo. É daqui que a API lê as features na hora da decisão. |
| `serving` | Serviço | O que o modelo produziu em produção: toda decisão de crédito, com a versão do modelo que a tomou. |
| `mlops` | Governança | Registro de modelos e histórico de monitoramento — o que permite auditar e detectar degradação. |

### Objetos

| Schema | Objeto | Tipo | Linhas (aprox.) | Tamanho | Descrição |
|---|---|---|---:|---:|---|
| `feature_store` | `abt` | tabela | 356.158 | 3111 MB | ABT: uma linha por cliente. Fonte das features lidas pela API na inferencia |
| `mlops` | `drift_metrics` | tabela | 93 | 104 kB | PSI por feature em cada execucao de monitoramento |
| `mlops` | `model_registry` | tabela | 1 | 64 kB | Versoes do modelo treinadas e seu status de promocao |
| `mlops` | `monitoring_runs` | tabela | 10 | 104 kB | Uma linha por execucao de monitoramento — da a serie historica que faltava |
| `mlops` | `performance_metrics` | tabela | 16 | 48 kB | AUC/KS/Gini por safra — so calculavel quando os labels amadurecem (label lag) |
| `mlops` | `vw_modelo_producao` | visao | 0 | 0 bytes | Modelo atualmente servindo em producao |
| `serving` | `predictions` | tabela | 5.600 | 2088 kB | Log auditavel de toda decisao de credito produzida pelo modelo |
| `serving` | `vw_decisoes_diarias` | visao | 0 | 0 bytes | Volume e PD media por dia e por decisao |
| `staging` | `clean_data` | tabela | 356.251 | 1152 MB | Saida de DataPipeline/data_sanitization.py |

---

## Schema `feature_store` — Camada gold

A ABT: uma linha por cliente, pronta para o modelo. É daqui que a API lê as features na hora da decisão.


### `feature_store.abt`

ABT: uma linha por cliente. Fonte das features lidas pela API na inferencia

*~356.158 linhas · 3111 MB*

| Coluna | Tipo | Obrigatória | Gerada | Descrição |
|---|---|---|---|---|
| `sk_id_curr` | `bigint` | sim | — | — |
| `target` | `smallint` | — | — | — |
| `is_train` | `boolean` | sim | — | — |
| `features` | `jsonb` | sim | — | As 836 features do modelo em JSONB (chave = nome da feature) |
| `abt_version` | `text` | — | — | — |
| `created_at` | `timestamp with time zone` | sim | — | — |
| `amt_income_total` | `numeric` | — | sim | — |
| `amt_credit` | `numeric` | — | sim | Coluna GERADA a partir de features->>AMT_CREDIT (nao editar diretamente) |
| `amt_annuity` | `numeric` | — | sim | — |
| `days_birth` | `numeric` | — | sim | — |
| `ext_source_1` | `numeric` | — | sim | — |
| `ext_source_2` | `numeric` | — | sim | — |
| `ext_source_3` | `numeric` | — | sim | — |
| `payment_rate` | `numeric` | — | sim | — |

**Chaves e regras de validação**

- **PK** `abt_pkey` — `PRIMARY KEY (sk_id_curr)`

**Índices**

- `ix_abt_ext_source_2` — btree (ext_source_2)
- `ix_abt_is_train` — btree (is_train)
- `ix_abt_target` — btree (target)

---

## Schema `mlops` — Governança

Registro de modelos e histórico de monitoramento — o que permite auditar e detectar degradação.


### `mlops.drift_metrics`

PSI por feature em cada execucao de monitoramento

*~93 linhas · 104 kB*

| Coluna | Tipo | Obrigatória | Gerada | Descrição |
|---|---|---|---|---|
| `metric_id` | `bigint` | sim | — | — |
| `run_id` | `bigint` | sim | — | — |
| `feature` | `text` | sim | — | — |
| `psi` | `numeric(12,6)` | — | — | — |
| `status` | `text` | — | — | — |

**Chaves e regras de validação**

- **CHECK** `drift_metrics_status_check` — `CHECK ((status = ANY (ARRAY['OK'::text, 'MODERADO'::text, 'SEVERO'::text])))`
- **FK** `drift_metrics_run_id_fkey` — `FOREIGN KEY (run_id) REFERENCES mlops.monitoring_runs(run_id) ON DELETE CASCADE`
- **PK** `drift_metrics_pkey` — `PRIMARY KEY (metric_id)`

**Índices**

- `ix_drift_feature` — btree (feature)
- `ix_drift_run` — btree (run_id)

### `mlops.model_registry`

Versoes do modelo treinadas e seu status de promocao

*~1 linhas · 64 kB*

| Coluna | Tipo | Obrigatória | Gerada | Descrição |
|---|---|---|---|---|
| `model_id` | `integer` | sim | — | — |
| `model_name` | `text` | sim | — | — |
| `model_version` | `text` | sim | — | — |
| `model_type` | `text` | sim | — | — |
| `trained_at` | `timestamp with time zone` | sim | — | — |
| `oof_auc` | `numeric(8,6)` | — | — | — |
| `n_features` | `integer` | — | — | — |
| `n_estimators` | `integer` | — | — | — |
| `params` | `jsonb` | — | — | — |
| `decision_policy` | `jsonb` | — | — | — |
| `artifact_path` | `text` | — | — | — |
| `status` | `text` | sim | — | — |
| `promoted_at` | `timestamp with time zone` | — | — | — |
| `created_at` | `timestamp with time zone` | sim | — | — |

**Chaves e regras de validação**

- **CHECK** `model_registry_status_check` — `CHECK ((status = ANY (ARRAY['staging'::text, 'production'::text, 'archived'::text])))`
- **PK** `model_registry_pkey` — `PRIMARY KEY (model_id)`
- **UNIQUE** `model_registry_model_version_key` — `UNIQUE (model_version)`

**Índices**

- `model_registry_model_version_key` — btree (model_version)
- `ux_model_registry_producao` — btree (model_name) WHERE (status = 'production'::text)

### `mlops.monitoring_runs`

Uma linha por execucao de monitoramento — da a serie historica que faltava

*~10 linhas · 104 kB*

| Coluna | Tipo | Obrigatória | Gerada | Descrição |
|---|---|---|---|---|
| `run_id` | `bigint` | sim | — | — |
| `run_type` | `text` | sim | — | — |
| `reference_label` | `text` | — | — | — |
| `current_label` | `text` | — | — | — |
| `model_version` | `text` | — | — | — |
| `n_features_evaluated` | `integer` | — | — | — |
| `n_severe` | `integer` | — | — | — |
| `n_moderate` | `integer` | — | — | — |
| `retrain_recommended` | `boolean` | sim | — | — |
| `summary` | `jsonb` | — | — | — |
| `created_at` | `timestamp with time zone` | sim | — | — |

**Chaves e regras de validação**

- **CHECK** `monitoring_runs_run_type_check` — `CHECK ((run_type = ANY (ARRAY['data_drift'::text, 'prediction_drift'::text, 'performance'::text])))`
- **PK** `monitoring_runs_pkey` — `PRIMARY KEY (run_id)`

**Índices**

- `ix_monruns_created` — btree (created_at DESC)
- `ix_monruns_type` — btree (run_type)

### `mlops.performance_metrics`

AUC/KS/Gini por safra — so calculavel quando os labels amadurecem (label lag)

*~16 linhas · 48 kB*

| Coluna | Tipo | Obrigatória | Gerada | Descrição |
|---|---|---|---|---|
| `metric_id` | `bigint` | sim | — | — |
| `run_id` | `bigint` | — | — | — |
| `model_version` | `text` | — | — | — |
| `cohort` | `text` | — | — | — |
| `n_obs` | `integer` | — | — | — |
| `auc` | `numeric(8,6)` | — | — | — |
| `ks` | `numeric(8,6)` | — | — | — |
| `gini` | `numeric(8,6)` | — | — | — |
| `default_rate` | `numeric(8,6)` | — | — | — |
| `created_at` | `timestamp with time zone` | sim | — | — |

**Chaves e regras de validação**

- **FK** `performance_metrics_run_id_fkey` — `FOREIGN KEY (run_id) REFERENCES mlops.monitoring_runs(run_id) ON DELETE CASCADE`
- **PK** `performance_metrics_pkey` — `PRIMARY KEY (metric_id)`

**Índices**

- `ix_perf_cohort` — btree (cohort)

### `mlops.vw_modelo_producao`

Modelo atualmente servindo em producao

| Coluna | Tipo | Obrigatória | Gerada | Descrição |
|---|---|---|---|---|
| `model_name` | `text` | — | — | — |
| `model_version` | `text` | — | — | — |
| `model_type` | `text` | — | — | — |
| `trained_at` | `timestamp with time zone` | — | — | — |
| `oof_auc` | `numeric(8,6)` | — | — | — |
| `n_features` | `integer` | — | — | — |
| `decision_policy` | `jsonb` | — | — | — |
| `promoted_at` | `timestamp with time zone` | — | — | — |

---

## Schema `serving` — Serviço

O que o modelo produziu em produção: toda decisão de crédito, com a versão do modelo que a tomou.


### `serving.predictions`

Log auditavel de toda decisao de credito produzida pelo modelo

*~5.600 linhas · 2088 kB*

| Coluna | Tipo | Obrigatória | Gerada | Descrição |
|---|---|---|---|---|
| `prediction_id` | `bigint` | sim | — | — |
| `sk_id_curr` | `bigint` | — | — | — |
| `model_version` | `text` | — | — | — |
| `probability_default` | `numeric(9,6)` | sim | — | — |
| `risk_band` | `text` | sim | — | — |
| `decision` | `text` | sim | — | — |
| `source` | `text` | sim | — | — |
| `n_features_expected` | `integer` | — | — | — |
| `n_features_missing` | `integer` | — | — | Features esperadas que chegaram nulas — alto = entrada incompleta, decisao suspeita |
| `request_payload` | `jsonb` | — | — | — |
| `latency_ms` | `numeric(10,2)` | — | — | — |
| `created_at` | `timestamp with time zone` | sim | — | — |

**Chaves e regras de validação**

- **CHECK** `predictions_decision_check` — `CHECK ((decision = ANY (ARRAY['APROVAR'::text, 'ANALISE_MANUAL'::text, 'RECUSAR'::text])))`
- **CHECK** `predictions_source_check` — `CHECK ((source = ANY (ARRAY['api'::text, 'batch'::text, 'streamlit'::text, 'cli'::text])))`
- **FK** `predictions_model_version_fkey` — `FOREIGN KEY (model_version) REFERENCES mlops.model_registry(model_version)`
- **PK** `predictions_pkey` — `PRIMARY KEY (prediction_id)`

**Índices**

- `ix_pred_created_at` — btree (created_at DESC)
- `ix_pred_decision` — btree (decision)
- `ix_pred_model_ver` — btree (model_version)
- `ix_pred_sk_id` — btree (sk_id_curr)

### `serving.vw_decisoes_diarias`

Volume e PD media por dia e por decisao

| Coluna | Tipo | Obrigatória | Gerada | Descrição |
|---|---|---|---|---|
| `dia` | `date` | — | — | — |
| `decisao` | `text` | — | — | — |
| `n` | `bigint` | — | — | — |
| `pd_media` | `numeric` | — | — | — |

---

## Schema `staging` — Camada silver

Dados limpos e padronizados, ainda no nível da tabela de origem.


### `staging.clean_data`

Saida de DataPipeline/data_sanitization.py

*~356.251 linhas · 1152 MB*

| Coluna | Tipo | Obrigatória | Gerada | Descrição |
|---|---|---|---|---|
| `sk_id_curr` | `bigint` | sim | — | — |
| `target` | `smallint` | — | — | TARGET=1 inadimplente, 0 adimplente, NULL para o conjunto de teste |
| `payload` | `jsonb` | sim | — | — |
| `ingested_at` | `timestamp with time zone` | sim | — | — |

**Chaves e regras de validação**

- **PK** `clean_data_pkey` — `PRIMARY KEY (sk_id_curr)`
