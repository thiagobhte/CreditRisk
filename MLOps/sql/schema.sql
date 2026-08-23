-- ============================================================
-- schema.sql — Modelo de dados da solução de risco de crédito.
--
-- Executado automaticamente na PRIMEIRA subida do container Postgres
-- (montado em /docker-entrypoint-initdb.d) e também sob demanda por
-- `python -m MLOps.db --init`, que é idempotente (IF NOT EXISTS).
--
-- Organização em camadas, espelhando o fluxo do pipeline:
--
--   staging       → dado limpo, ainda no nível da tabela de origem
--   feature_store → a ABT: uma linha por cliente, pronta para o modelo.
--                   É DAQUI que a API lê as features de um cliente.
--   serving       → o que o modelo produziu em produção (log de decisões)
--   mlops         → governança: versões do modelo e histórico de monitoramento
-- ============================================================

CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS feature_store;
CREATE SCHEMA IF NOT EXISTS serving;
CREATE SCHEMA IF NOT EXISTS mlops;

COMMENT ON SCHEMA staging       IS 'Camada silver: dados limpos por data_sanitization.py';
COMMENT ON SCHEMA feature_store IS 'Camada gold: ABT por cliente, consumida pela API e pelo treino';
COMMENT ON SCHEMA serving       IS 'Saidas do modelo em producao (predicoes e decisoes)';
COMMENT ON SCHEMA mlops         IS 'Governanca: registro de modelos e monitoramento';


-- ============================================================
-- FUNÇÃO AUXILIAR: leitura numérica segura de dentro do JSONB
-- ============================================================
-- A ABT tem NaN/inf em abundância (razões com denominador zero, clientes sem
-- histórico). Um cast direto `(features->>'X')::numeric` derrubaria a consulta
-- inteira no primeiro valor inválido. Esta função devolve NULL nesses casos.
--
-- Precisa ser IMMUTABLE para poder ser usada em coluna GERADA (mais abaixo).
--
-- POR QUE SQL PURO, E NÃO plpgsql COM `EXCEPTION`:
--   a versão com bloco EXCEPTION parecia mais natural, mas capturar exceção em
--   plpgsql abre uma SUBTRANSAÇÃO — e o Postgres proíbe subtransação dentro de
--   consulta PARALELA. O monitoramento lê 30 features de uma amostra grande, o
--   planejador paraleliza, e a consulta quebrava com
--   "cannot start subtransactions during a parallel operation".
--   Validar o texto ANTES de converter resolve: sem exceção, sem subtransação,
--   função inlineável pelo planejador e de fato PARALLEL SAFE.
CREATE OR REPLACE FUNCTION feature_store.jsonb_num(payload jsonb, chave text)
RETURNS numeric
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $func$
    SELECT CASE
        WHEN payload ->> chave ~ '^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$'
        THEN (payload ->> chave)::numeric
        ELSE NULL          -- cobre NaN, Infinity, texto e ausência
    END;
$func$;

COMMENT ON FUNCTION feature_store.jsonb_num(jsonb, text)
    IS 'Extrai um numero do JSONB tolerando NaN/inf/texto invalido (devolve NULL)';


-- ============================================================
-- STAGING — dado limpo (saída de data_sanitization.py)
-- ============================================================
CREATE TABLE IF NOT EXISTS staging.clean_data (
    sk_id_curr   BIGINT      PRIMARY KEY,
    target       SMALLINT,                    -- NULL nas linhas de teste
    payload      JSONB       NOT NULL,        -- demais colunas da tabela limpa
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  staging.clean_data IS 'Saida de DataPipeline/data_sanitization.py';
COMMENT ON COLUMN staging.clean_data.target
    IS 'TARGET=1 inadimplente, 0 adimplente, NULL para o conjunto de teste';


-- ============================================================
-- FEATURE STORE — a ABT (uma linha por cliente)
-- ============================================================
-- DECISÃO DE MODELAGEM: as 836 features do modelo ficam num JSONB, e não em
-- 836 colunas.
--   * o Postgres suporta ~1600 colunas, mas um DDL com 836 colunas seria
--     impossível de manter — a cada feature nova, um ALTER TABLE;
--   * a API lê o cliente INTEIRO de uma vez (SELECT features WHERE sk_id_curr),
--     que é exatamente o padrão de acesso de um feature store online;
--   * as variáveis que o negócio consulta e filtra viram COLUNAS GERADAS a
--     partir do próprio JSONB — indexáveis, com tipo forte, e sem duplicar a
--     fonte da verdade (são derivadas, não copiadas).
CREATE TABLE IF NOT EXISTS feature_store.abt (
    sk_id_curr   BIGINT      PRIMARY KEY,
    target       SMALLINT,                    -- NULL = cliente do conjunto de teste
    is_train     BOOLEAN     NOT NULL DEFAULT true,
    features     JSONB       NOT NULL,        -- as 836 features do modelo
    abt_version  TEXT,                        -- rastreabilidade da carga
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Colunas de negócio materializadas (derivadas do JSONB acima).
    -- São as mesmas variáveis que o analista simula no painel.
    amt_income_total NUMERIC GENERATED ALWAYS AS (feature_store.jsonb_num(features, 'AMT_INCOME_TOTAL')) STORED,
    amt_credit       NUMERIC GENERATED ALWAYS AS (feature_store.jsonb_num(features, 'AMT_CREDIT'))       STORED,
    amt_annuity      NUMERIC GENERATED ALWAYS AS (feature_store.jsonb_num(features, 'AMT_ANNUITY'))      STORED,
    days_birth       NUMERIC GENERATED ALWAYS AS (feature_store.jsonb_num(features, 'DAYS_BIRTH'))       STORED,
    ext_source_1     NUMERIC GENERATED ALWAYS AS (feature_store.jsonb_num(features, 'EXT_SOURCE_1'))     STORED,
    ext_source_2     NUMERIC GENERATED ALWAYS AS (feature_store.jsonb_num(features, 'EXT_SOURCE_2'))     STORED,
    ext_source_3     NUMERIC GENERATED ALWAYS AS (feature_store.jsonb_num(features, 'EXT_SOURCE_3'))     STORED,
    payment_rate     NUMERIC GENERATED ALWAYS AS (feature_store.jsonb_num(features, 'PAYMENT_RATE'))     STORED
);

COMMENT ON TABLE  feature_store.abt
    IS 'ABT: uma linha por cliente. Fonte das features lidas pela API na inferencia';
COMMENT ON COLUMN feature_store.abt.features
    IS 'As 836 features do modelo em JSONB (chave = nome da feature)';
COMMENT ON COLUMN feature_store.abt.amt_credit
    IS 'Coluna GERADA a partir de features->>AMT_CREDIT (nao editar diretamente)';

CREATE INDEX IF NOT EXISTS ix_abt_target       ON feature_store.abt (target);
CREATE INDEX IF NOT EXISTS ix_abt_is_train     ON feature_store.abt (is_train);
CREATE INDEX IF NOT EXISTS ix_abt_ext_source_2 ON feature_store.abt (ext_source_2);


-- ============================================================
-- MLOPS — registro de modelos (model registry)
-- ============================================================
-- Responde "qual modelo estava servindo quando esta decisão foi tomada?" —
-- pergunta de auditoria que o projeto não conseguia responder, porque o modelo
-- era apenas um arquivo .joblib sobrescrito a cada treino.
CREATE TABLE IF NOT EXISTS mlops.model_registry (
    model_id        SERIAL      PRIMARY KEY,
    model_name      TEXT        NOT NULL DEFAULT 'credit_risk_lgbm',
    model_version   TEXT        NOT NULL UNIQUE,   -- ex.: '2026-07-13T09:54:50'
    model_type      TEXT        NOT NULL,          -- ex.: 'LGBMClassifier'
    trained_at      TIMESTAMPTZ NOT NULL,
    oof_auc         NUMERIC(8, 6),
    n_features      INTEGER,
    n_estimators    INTEGER,
    params          JSONB,
    decision_policy JSONB,                         -- cortes de aprovacao/recusa vigentes
    artifact_path   TEXT,                          -- caminho do .joblib
    status          TEXT        NOT NULL DEFAULT 'staging'
                    CHECK (status IN ('staging', 'production', 'archived')),
    promoted_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE mlops.model_registry IS 'Versoes do modelo treinadas e seu status de promocao';

-- Garante no máximo UM modelo em produção por nome (regra de governança).
CREATE UNIQUE INDEX IF NOT EXISTS ux_model_registry_producao
    ON mlops.model_registry (model_name)
    WHERE status = 'production';


-- ============================================================
-- SERVING — log de predições (auditoria e base do monitoramento)
-- ============================================================
-- Toda decisão de crédito tomada pelo modelo é gravada aqui. Serve a três
-- propósitos: auditoria regulatória, cálculo de performance quando os labels
-- amadurecem (label lag) e detecção de prediction drift.
CREATE TABLE IF NOT EXISTS serving.predictions (
    prediction_id       BIGSERIAL     PRIMARY KEY,
    sk_id_curr          BIGINT,                    -- sem FK: pode pontuar cliente novo
    model_version       TEXT          REFERENCES mlops.model_registry (model_version),
    probability_default NUMERIC(9, 6) NOT NULL,
    risk_band           TEXT          NOT NULL,
    decision            TEXT          NOT NULL
                        CHECK (decision IN ('APROVAR', 'ANALISE_MANUAL', 'RECUSAR')),
    source              TEXT          NOT NULL DEFAULT 'api'
                        CHECK (source IN ('api', 'batch', 'streamlit', 'cli')),
    n_features_expected INTEGER,
    n_features_missing  INTEGER,                   -- quantas chegaram nulas (qualidade)
    request_payload     JSONB,                     -- o que o chamador enviou/sobrescreveu
    latency_ms          NUMERIC(10, 2),
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT now()
);

COMMENT ON TABLE  serving.predictions IS 'Log auditavel de toda decisao de credito produzida pelo modelo';
COMMENT ON COLUMN serving.predictions.n_features_missing
    IS 'Features esperadas que chegaram nulas — alto = entrada incompleta, decisao suspeita';

CREATE INDEX IF NOT EXISTS ix_pred_created_at ON serving.predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_pred_sk_id      ON serving.predictions (sk_id_curr);
CREATE INDEX IF NOT EXISTS ix_pred_decision   ON serving.predictions (decision);
CREATE INDEX IF NOT EXISTS ix_pred_model_ver  ON serving.predictions (model_version);


-- ============================================================
-- MLOPS — monitoramento (execuções, drift e performance)
-- ============================================================
CREATE TABLE IF NOT EXISTS mlops.monitoring_runs (
    run_id               BIGSERIAL   PRIMARY KEY,
    run_type             TEXT        NOT NULL
                         CHECK (run_type IN ('data_drift', 'prediction_drift', 'performance')),
    reference_label      TEXT,                     -- o que foi usado como baseline
    current_label        TEXT,                     -- o que foi comparado contra o baseline
    model_version        TEXT,
    n_features_evaluated INTEGER,
    n_severe             INTEGER,
    n_moderate           INTEGER,
    retrain_recommended  BOOLEAN     NOT NULL DEFAULT false,
    summary              JSONB,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE mlops.monitoring_runs
    IS 'Uma linha por execucao de monitoramento — da a serie historica que faltava';

CREATE INDEX IF NOT EXISTS ix_monruns_created ON mlops.monitoring_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_monruns_type    ON mlops.monitoring_runs (run_type);


CREATE TABLE IF NOT EXISTS mlops.drift_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    run_id    BIGINT  NOT NULL REFERENCES mlops.monitoring_runs (run_id) ON DELETE CASCADE,
    feature   TEXT    NOT NULL,
    psi       NUMERIC(12, 6),
    status    TEXT    CHECK (status IN ('OK', 'MODERADO', 'SEVERO'))
);

COMMENT ON TABLE mlops.drift_metrics IS 'PSI por feature em cada execucao de monitoramento';

CREATE INDEX IF NOT EXISTS ix_drift_run     ON mlops.drift_metrics (run_id);
CREATE INDEX IF NOT EXISTS ix_drift_feature ON mlops.drift_metrics (feature);


CREATE TABLE IF NOT EXISTS mlops.performance_metrics (
    metric_id     BIGSERIAL PRIMARY KEY,
    run_id        BIGINT REFERENCES mlops.monitoring_runs (run_id) ON DELETE CASCADE,
    model_version TEXT,
    cohort        TEXT,                 -- safra avaliada (ex.: '2026-07')
    n_obs         INTEGER,
    auc           NUMERIC(8, 6),
    ks            NUMERIC(8, 6),
    gini          NUMERIC(8, 6),
    default_rate  NUMERIC(8, 6),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE mlops.performance_metrics
    IS 'AUC/KS/Gini por safra — so calculavel quando os labels amadurecem (label lag)';

CREATE INDEX IF NOT EXISTS ix_perf_cohort ON mlops.performance_metrics (cohort);


-- ============================================================
-- VISÕES DE APOIO (usadas pelo painel e pela demonstração)
-- ============================================================
CREATE OR REPLACE VIEW serving.vw_decisoes_diarias AS
SELECT
    date_trunc('day', created_at)::date AS dia,
    decision                            AS decisao,
    count(*)                            AS n,
    round(avg(probability_default), 4)  AS pd_media
FROM serving.predictions
GROUP BY 1, 2
ORDER BY 1 DESC, 2;

COMMENT ON VIEW serving.vw_decisoes_diarias IS 'Volume e PD media por dia e por decisao';


CREATE OR REPLACE VIEW mlops.vw_modelo_producao AS
SELECT model_name, model_version, model_type, trained_at, oof_auc,
       n_features, decision_policy, promoted_at
FROM mlops.model_registry
WHERE status = 'production';

COMMENT ON VIEW mlops.vw_modelo_producao IS 'Modelo atualmente servindo em producao';
