"""
store.py — Leitura e escrita das tabelas da aplicação em produção.

Enquanto o `db.py` cuida da conexão, este módulo cuida do que a aplicação faz
com o banco no dia a dia:

    LER   → as features de um cliente na feature store (o que a API precisa
            para pontuar alguém sem exigir 836 campos no corpo da requisição)
    ESCREVER → o registro de cada decisão de crédito (auditoria) e a versão do
            modelo que a produziu (governança)

É consumido pela API (`app/main.py`), pelo painel e, mais adiante, pelo
monitoramento.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from MLOps.db import get_engine


# ============================================================
# LEITURA — FEATURE STORE
# ============================================================

def get_client_features(sk_id_curr: int) -> dict | None:
    """
    Devolve as features de um cliente, ou None se ele não existe.

    Esta é A consulta do serviço de predição: uma leitura por chave primária,
    que o Postgres resolve em ~1 ms. É o que substitui a exigência antiga de o
    chamador enviar as 836 features prontas no corpo da requisição.
    """
    consulta = text("""
        SELECT features, target, is_train, abt_version
        FROM feature_store.abt
        WHERE sk_id_curr = :sk_id
    """)
    with get_engine().connect() as conn:
        linha = conn.execute(consulta, {"sk_id": sk_id_curr}).mappings().first()

    if linha is None:
        return None

    return {
        "features":    dict(linha["features"]),
        "target":      linha["target"],
        "is_train":    linha["is_train"],
        "abt_version": linha["abt_version"],
    }


def list_client_ids(limite: int = 100, apenas_rotulados: bool = False) -> list:
    """
    Lista IDs de clientes disponíveis — alimenta o seletor do painel e ajuda
    quem for testar a API a descobrir um ID válido.
    """
    filtro = "WHERE target IS NOT NULL" if apenas_rotulados else ""
    consulta = text(f"""
        SELECT sk_id_curr FROM feature_store.abt
        {filtro}
        ORDER BY sk_id_curr
        LIMIT :limite
    """)
    with get_engine().connect() as conn:
        return [r[0] for r in conn.execute(consulta, {"limite": limite})]


def count_clients() -> int:
    """Quantos clientes existem na feature store (usado no /health)."""
    with get_engine().connect() as conn:
        return conn.execute(text("SELECT count(*) FROM feature_store.abt")).scalar_one()


# ============================================================
# GOVERNANÇA — REGISTRO DO MODELO
# ============================================================

def ensure_model_registered(metadados: dict, artifact_path: str = None) -> str:
    """
    Garante que o modelo em uso está no `mlops.model_registry`.

    Por que é obrigatório e não opcional: cada predição gravada aponta para a
    versão do modelo que a gerou (chave estrangeira). Sem o modelo registrado,
    o log de decisões não teria como ser escrito — e é justamente esse vínculo
    que responde, numa auditoria, "qual modelo recusou este cliente?".

    A versão é o instante do treino (`trained_at`), que já vem nos metadados e
    é único por execução do train.py.
    """
    versao = str(metadados.get("trained_at"))

    registrar = text("""
        INSERT INTO mlops.model_registry
            (model_version, model_type, trained_at, oof_auc, n_features,
             n_estimators, params, decision_policy, artifact_path)
        VALUES
            (:versao, :tipo, :treinado_em, :auc, :n_features,
             :n_estimators, :params, :politica, :caminho)
        ON CONFLICT (model_version) DO UPDATE
            SET oof_auc         = EXCLUDED.oof_auc,
                n_features      = EXCLUDED.n_features,
                decision_policy = EXCLUDED.decision_policy,
                artifact_path   = EXCLUDED.artifact_path
    """)

    # Promove a produção apenas se ainda não houver nenhum modelo promovido.
    # O índice único parcial do schema impede dois em produção ao mesmo tempo;
    # trocar o modelo de produção é uma decisão deliberada, não um efeito
    # colateral de alguém ter subido a API.
    promover = text("""
        UPDATE mlops.model_registry
        SET status = 'production', promoted_at = now()
        WHERE model_version = :versao
          AND NOT EXISTS (
              SELECT 1 FROM mlops.model_registry
              WHERE status = 'production' AND model_name = 'credit_risk_lgbm'
          )
    """)

    with get_engine().begin() as conn:
        conn.execute(registrar, {
            "versao":       versao,
            "tipo":         metadados.get("model_type", "desconhecido"),
            "treinado_em":  metadados.get("trained_at"),
            "auc":          metadados.get("oof_auc"),
            "n_features":   metadados.get("n_features"),
            "n_estimators": metadados.get("n_estimators"),
            "params":       json.dumps(metadados.get("params", {})),
            "politica":     json.dumps(metadados.get("decision_policy", {})),
            "caminho":      artifact_path,
        })
        conn.execute(promover, {"versao": versao})

    return versao


def get_production_model() -> dict | None:
    """Qual modelo está servindo agora, segundo o registro."""
    with get_engine().connect() as conn:
        linha = conn.execute(text("SELECT * FROM mlops.vw_modelo_producao")).mappings().first()
    return dict(linha) if linha else None


# ============================================================
# ESCRITA — LOG DE DECISÕES
# ============================================================

def log_prediction(resultado: dict, model_version: str, origem: str = "api",
                   payload: dict = None, latencia_ms: float = None) -> int | None:
    """
    Grava uma decisão de crédito em `serving.predictions`.

    Devolve o id gerado, ou None se a gravação falhou.

    DECISÃO DE PROJETO — o log é "melhor esforço": se o banco estiver fora do
    ar, a API ainda responde a predição, e a falha do log vira um aviso. O
    raciocínio é que um analista esperando uma decisão de crédito não deve ser
    bloqueado por um problema na trilha de auditoria. Em produção de verdade,
    o certo seria enfileirar o registro (Kafka/SQS) e reprocessar depois, para
    não perder o evento — aqui não há fila, então assumimos essa limitação
    conscientemente em vez de fingir que ela não existe.
    """
    inserir = text("""
        INSERT INTO serving.predictions
            (sk_id_curr, model_version, probability_default, risk_band, decision,
             source, n_features_expected, n_features_missing, request_payload, latency_ms)
        VALUES
            (:sk_id, :versao, :pd, :faixa, :decisao,
             :origem, :esperadas, :ausentes, :payload, :latencia)
        RETURNING prediction_id
    """)
    try:
        with get_engine().begin() as conn:
            return conn.execute(inserir, {
                "sk_id":     resultado.get("SK_ID_CURR"),
                "versao":    model_version,
                "pd":        resultado["probability_default"],
                "faixa":     resultado["risk_band"],
                "decisao":   resultado["decision"],
                "origem":    origem,
                "esperadas": resultado.get("n_features_expected"),
                "ausentes":  resultado.get("n_features_missing"),
                "payload":   json.dumps(payload or {}, ensure_ascii=False, default=str),
                "latencia":  latencia_ms,
            }).scalar_one()
    except Exception as erro:
        print(f"[AVISO] nao foi possivel registrar a predicao no banco: {erro}")
        return None


def recent_predictions(limite: int = 20) -> list:
    """Últimas decisões registradas — usado no painel e na demonstração."""
    consulta = text("""
        SELECT prediction_id, sk_id_curr, probability_default, risk_band,
               decision, source, n_features_missing, latency_ms, created_at
        FROM serving.predictions
        ORDER BY created_at DESC
        LIMIT :limite
    """)
    with get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(consulta, {"limite": limite}).mappings()]
