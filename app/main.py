"""
app/main.py — API REST de predição de risco de crédito (FastAPI).

Expõe o modelo treinado como um serviço HTTP. É a camada de "deploy do modelo
como serviço de predição" pedida na etapa individual.

DE ONDE VÊM AS FEATURES
    A ABT tem 836 features por cliente, quase todas agregações do histórico
    (bureau, aplicações anteriores, parcelas, cartão, POS). Exigir que o
    chamador enviasse tudo isso no corpo da requisição não é realista: quem
    consome a API é o sistema de originação de crédito, que conhece a proposta
    (renda, valor pedido, prazo) e NÃO o histórico consolidado do cliente.

    Por isso a API busca as features na `feature_store.abt` (PostgreSQL) a
    partir do ID do cliente, e aceita "overrides" apenas para as variáveis que
    fazem sentido simular. É o caminho principal — `POST /predict/{id}`.

Endpoints:
    GET  /health              → saúde do serviço, do modelo e do banco
    GET  /clients/{id}        → features do cliente na feature store
    GET  /clients             → alguns IDs válidos, para descobrir o que testar
    POST /predict/{id}        → busca no banco + aplica overrides + pontua
    POST /predict             → payload completo (compatibilidade)
    POST /predict/batch       → vários clientes de uma vez
    GET  /predictions/recent  → últimas decisões registradas (auditoria)

Rodar localmente:
    uvicorn app.main:app --reload --port 8000
Documentação interativa (Swagger) em: http://localhost:8000/docs
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import MODEL_PATH
from Model.predict import predict, model_metadata, apply_overrides
from MLOps import store

# Versão do modelo servida por este processo. Preenchida na subida e usada em
# cada registro de predição, para o log de auditoria dizer QUAL modelo decidiu.
MODEL_VERSION: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Registra no banco o modelo que este processo está servindo.

    Roda uma vez, quando a API sobe. Se o banco não estiver disponível, o
    serviço sobe assim mesmo: predições continuam funcionando (o modelo está em
    disco), só o log de auditoria fica indisponível — e o /health denuncia isso
    em vez de esconder.
    """
    global MODEL_VERSION
    try:
        MODEL_VERSION = store.ensure_model_registered(model_metadata(), artifact_path=MODEL_PATH)
        print(f"[startup] modelo registrado no banco: versao {MODEL_VERSION}")
    except FileNotFoundError:
        print("[startup] modelo nao encontrado em disco - treine com: python -m Model.train")
    except Exception as erro:
        print(f"[startup] banco indisponivel, seguindo sem registro de modelo: {erro}")
    yield


app = FastAPI(
    title="Home Credit Default Risk — Serviço de Predição",
    description=(
        "Retorna a probabilidade de inadimplência (PD) e a decisão de crédito. "
        "As features do cliente são lidas da feature store (PostgreSQL); o "
        "chamador informa apenas o ID e, opcionalmente, as variáveis que quer simular."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# SCHEMAS DE ENTRADA/SAÍDA
# ============================================================

class ClientFeatures(BaseModel):
    """
    Features de UM cliente no nível da ABT (endpoint legado `/predict`).

    Aceita campos arbitrários (extra='allow') porque a ABT tem centenas de
    features e listá-las todas aqui seria impraticável e frágil. O predict.py
    alinha o que chegar ao formato do modelo; o que faltar vira NaN — e a
    resposta informa quantas faltaram.
    """
    model_config = {"extra": "allow"}

    SK_ID_CURR: Optional[int] = Field(
        default=None, description="ID do cliente (opcional, apenas ecoado na resposta)"
    )


class SimulationRequest(BaseModel):
    """
    Alterações a aplicar sobre um cliente que já existe na feature store.

    Todos os campos são opcionais: o que não vier, fica como está no banco.
    Ao alterar renda/crédito/parcela/idade, as features derivadas (razão
    parcela/renda, taxa de pagamento, endividamento sobre renda...) são
    recalculadas — senão o modelo receberia um cliente incoerente.
    """
    model_config = {"extra": "forbid"}

    AMT_INCOME_TOTAL: Optional[float] = Field(default=None, description="Renda anual", gt=0)
    AMT_CREDIT:       Optional[float] = Field(default=None, description="Crédito solicitado", gt=0)
    AMT_ANNUITY:      Optional[float] = Field(default=None, description="Parcela anual", ge=0)
    DAYS_BIRTH:       Optional[float] = Field(default=None,
                                              description="Idade em dias, negativo (ex.: -12000)",
                                              lt=0)
    EXT_SOURCE_1:     Optional[float] = Field(default=None, description="Score externo 1", ge=0, le=1)
    EXT_SOURCE_2:     Optional[float] = Field(default=None, description="Score externo 2", ge=0, le=1)
    EXT_SOURCE_3:     Optional[float] = Field(default=None, description="Score externo 3", ge=0, le=1)


class BatchRequest(BaseModel):
    """Lote de clientes para o endpoint /predict/batch."""
    clients: List[Dict[str, Any]] = Field(..., description="Lista de clientes (features da ABT)")


class PredictionResponse(BaseModel):
    SK_ID_CURR: Optional[int] = None
    probability_default: float = Field(..., description="Probabilidade de default (0 a 1)")
    risk_band: str = Field(..., description="BAIXO / MODERADO / ALTO / MUITO_ALTO")
    decision: str = Field(..., description="APROVAR / ANALISE_MANUAL / RECUSAR")
    n_features_expected: Optional[int] = Field(
        default=None, description="Quantas features o modelo espera")
    n_features_missing: Optional[int] = Field(
        default=None,
        description="Quantas chegaram vazias. Valor alto = entrada incompleta, "
                    "decisão pouco confiável.")


class SimulationResponse(PredictionResponse):
    """Resposta da simulação, com a origem das features declarada."""
    features_from_store: int = Field(..., description="Features carregadas do banco")
    features_overridden: List[str] = Field(..., description="Variáveis alteradas pelo chamador")
    derived_recalculated: List[str] = Field(..., description="Features derivadas recalculadas")
    real_outcome: Optional[int] = Field(
        default=None, description="Desfecho real do cliente (TARGET), quando conhecido")


# ============================================================
# INFRA
# ============================================================

@app.get("/health", tags=["infra"])
def health() -> dict:
    """
    Verifica se o serviço está de pé, se o modelo carregou e se o banco responde.

    Usado por orquestradores (docker-compose healthcheck, Kubernetes) e pelo
    monitoramento. Devolve os metadados do modelo para rastreabilidade
    (qual versão/AUC está servindo agora).
    """
    try:
        meta = model_metadata()
    except FileNotFoundError as e:
        # Modelo ainda não treinado → serviço "vivo" mas não "pronto"
        raise HTTPException(status_code=503, detail=str(e))

    # O banco fora do ar não derruba o /health: a API ainda pontua clientes com
    # payload completo. Mas o estado precisa ficar visível, não escondido.
    try:
        banco = {"status": "ok", "clientes_na_feature_store": store.count_clients()}
    except Exception as erro:
        banco = {"status": "indisponivel", "erro": str(erro)[:200]}

    return {
        "status": "ok",
        "model": meta,
        "model_version": MODEL_VERSION,
        "database": banco,
    }


# ============================================================
# CONSULTA À FEATURE STORE
# ============================================================

@app.get("/clients", tags=["feature store"])
def listar_clientes(
    limite: int = Query(20, ge=1, le=200, description="Quantos IDs devolver"),
    apenas_rotulados: bool = Query(False, description="Só clientes com desfecho conhecido"),
) -> dict:
    """Lista alguns IDs válidos — atalho para descobrir o que testar na API."""
    try:
        ids = store.list_client_ids(limite=limite, apenas_rotulados=apenas_rotulados)
        return {"total_na_feature_store": store.count_clients(), "sk_id_curr": ids}
    except Exception as erro:
        raise HTTPException(status_code=503, detail=f"Feature store indisponivel: {erro}")


@app.get("/clients/{sk_id_curr}", tags=["feature store"])
def obter_cliente(
    sk_id_curr: int = Path(..., description="ID do cliente"),
    incluir_features: bool = Query(False, description="Devolver as 836 features"),
) -> dict:
    """
    Mostra o que a feature store sabe sobre um cliente.

    Por padrão devolve só o resumo — as variáveis de negócio e a contagem de
    features. É este endpoint que torna visível a resposta para "de onde vêm as
    outras variáveis": elas estão aqui, vindas do histórico do cliente.
    """
    try:
        cliente = store.get_client_features(sk_id_curr)
    except Exception as erro:
        raise HTTPException(status_code=503, detail=f"Feature store indisponivel: {erro}")

    if cliente is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente {sk_id_curr} nao encontrado na feature store. "
                   f"Consulte /clients para ver IDs validos.",
        )

    features = cliente["features"]
    resposta = {
        "SK_ID_CURR": sk_id_curr,
        "n_features_disponiveis": len(features),
        "abt_version": cliente["abt_version"],
        "desfecho_real": cliente["target"],
        "variaveis_de_negocio": {
            chave: features.get(chave)
            for chave in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH",
                          "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "PAYMENT_RATE")
        },
    }
    if incluir_features:
        resposta["features"] = features
    return resposta


# ============================================================
# PREDIÇÃO
# ============================================================

@app.post("/predict/{sk_id_curr}", response_model=SimulationResponse, tags=["predição"])
def prever_cliente(
    sk_id_curr: int = Path(..., description="ID do cliente na feature store"),
    simulacao: Optional[SimulationRequest] = None,
) -> dict:
    """
    Prediz o risco de um cliente **buscando as features no banco**.

    É o endpoint principal. O chamador envia o ID e, se quiser, as variáveis a
    simular; a API carrega o restante do histórico da feature store, recalcula
    as features derivadas afetadas e pontua.

    Exemplo — o mesmo cliente com renda simulada em R$ 250 mil:
        POST /predict/100002
        {"AMT_INCOME_TOTAL": 250000}
    """
    inicio = time.perf_counter()

    try:
        cliente = store.get_client_features(sk_id_curr)
    except Exception as erro:
        raise HTTPException(status_code=503, detail=f"Feature store indisponivel: {erro}")

    if cliente is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente {sk_id_curr} nao encontrado na feature store. "
                   f"Consulte /clients para ver IDs validos.",
        )

    do_banco = cliente["features"]
    overrides = simulacao.model_dump(exclude_none=True) if simulacao else {}

    features = apply_overrides(do_banco, overrides) if overrides else dict(do_banco)

    # Quais derivadas o override afetou — informação que o painel exibe e que
    # deixa claro que a simulação não é uma troca ingênua de um número só.
    from Model.predict import DERIVADAS_POR_VARIAVEL
    derivadas = sorted({d for v in overrides for d in DERIVADAS_POR_VARIAVEL.get(v, [])})

    try:
        resultado = predict({"SK_ID_CURR": sk_id_curr, **features})[0]
    except Exception as erro:
        raise HTTPException(status_code=400, detail=f"Erro na predição: {erro}")

    latencia = (time.perf_counter() - inicio) * 1000
    store.log_prediction(resultado, MODEL_VERSION, origem="api",
                         payload=overrides, latencia_ms=latencia)

    return {
        **resultado,
        "features_from_store":  len(do_banco),
        "features_overridden":  sorted(overrides.keys()),
        "derived_recalculated": derivadas,
        "real_outcome":         cliente["target"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["predição"])
def predict_one(client: ClientFeatures) -> dict:
    """
    Prediz o risco de UM cliente a partir de um payload completo.

    Mantido para compatibilidade e para o caso de um cliente que ainda não está
    na feature store. Atenção ao `n_features_missing` da resposta: enviar
    poucas features não gera erro, gera uma PD sem lastro.
    """
    inicio = time.perf_counter()
    try:
        dados = client.model_dump()  # inclui os campos extras (features da ABT)
        resultado = predict(dados)[0]
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na predição: {e}")

    latencia = (time.perf_counter() - inicio) * 1000
    store.log_prediction(resultado, MODEL_VERSION, origem="api",
                         payload={"n_campos_enviados": len(dados)}, latencia_ms=latencia)
    return resultado


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["predição"])
def predict_batch(req: BatchRequest) -> list:
    """Prediz o risco de vários clientes de uma vez."""
    if not req.clients:
        raise HTTPException(status_code=400, detail="Lista 'clients' vazia.")
    inicio = time.perf_counter()
    try:
        resultados = predict(req.clients)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na predição: {e}")

    # Latência rateada: o custo do lote dividido entre os clientes, para a
    # métrica no banco continuar comparável com a das predições individuais.
    latencia = (time.perf_counter() - inicio) * 1000 / len(resultados)
    for resultado in resultados:
        store.log_prediction(resultado, MODEL_VERSION, origem="batch", latencia_ms=latencia)
    return resultados


# ============================================================
# EXPLICABILIDADE
# ============================================================
# Ficam na API, e não no painel, de propósito: explicar uma decisão de crédito
# é uma capacidade do SERVIÇO, não de uma tela. Assim qualquer consumidor —
# o painel do analista, um sistema de ouvidoria, um relatório regulatório —
# obtém a mesma justificativa, calculada pelo mesmo modelo.

# Origem de cada feature, deduzida do prefixo que o abt_transform usa ao
# agregar cada tabela. É o que responde "de onde vêm as outras variáveis?".
ORIGEM_POR_PREFIXO = (
    ("BURO_",     "Bureau de crédito"),
    ("BUREAU_",   "Bureau de crédito"),
    ("ACTIVE_",   "Bureau de crédito"),
    ("CLOSED_",   "Bureau de crédito"),
    ("PREV_",     "Aplicações anteriores"),
    ("APPROVED_", "Aplicações anteriores"),
    ("REFUSED_",  "Aplicações anteriores"),
    ("POS_",      "POS / crédito no ponto de venda"),
    ("INSTAL_",   "Histórico de parcelas"),
    ("CC_",       "Cartão de crédito"),
)


def origem_da_feature(nome: str) -> str:
    """Diz de qual tabela de origem uma feature veio."""
    for prefixo, origem in ORIGEM_POR_PREFIXO:
        if nome.startswith(prefixo):
            return origem
    return "Cadastro e proposta"


@app.get("/features/origins", tags=["explicabilidade"])
def origens_das_features() -> dict:
    """
    De onde vêm as 836 features do modelo, agrupadas por tabela de origem.

    Só 249 delas vêm do cadastro e da proposta — o que um formulário de crédito
    coleta. As outras 587 são agregações do histórico do cliente, e é por isso
    que a API precisa lê-las do banco em vez de esperá-las na requisição.
    """
    from Model.predict import _load_artifacts
    _, feats, _ = _load_artifacts()

    grupos: Dict[str, int] = {}
    for nome in feats:
        origem = origem_da_feature(nome)
        grupos[origem] = grupos.get(origem, 0) + 1

    return {
        "total": len(feats),
        "por_origem": dict(sorted(grupos.items(), key=lambda kv: -kv[1])),
        "simulaveis": sorted(SimulationRequest.model_fields.keys()),
    }


@app.post("/explain/{sk_id_curr}", tags=["explicabilidade"])
def explicar_cliente(
    sk_id_curr: int = Path(..., description="ID do cliente"),
    simulacao: Optional[SimulationRequest] = None,
    top_n: int = Query(12, ge=1, le=30, description="Quantas features retornar"),
) -> dict:
    """
    Explica a decisão de um cliente (valores SHAP), com a mesma simulação
    aceita pelo /predict/{id}.

    Responde "por que ESTE cliente foi recusado?" — a explicação individual
    exigida em crédito, que a importância média do modelo não dá.
    """
    from Model.explain import explain_client

    try:
        cliente = store.get_client_features(sk_id_curr)
    except Exception as erro:
        raise HTTPException(status_code=503, detail=f"Feature store indisponivel: {erro}")
    if cliente is None:
        raise HTTPException(status_code=404, detail=f"Cliente {sk_id_curr} nao encontrado.")

    overrides = simulacao.model_dump(exclude_none=True) if simulacao else {}
    features = apply_overrides(cliente["features"], overrides) if overrides else cliente["features"]

    try:
        explicacao = explain_client(features, top_n=top_n)
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro ao explicar: {erro}")

    # Anexa a origem de cada feature: o analista vê não só QUAL variável pesou,
    # mas de onde ela veio (bureau, parcelas, cadastro...).
    for contribuicao in explicacao["contributions"]:
        contribuicao["origem"] = origem_da_feature(contribuicao["feature"])

    return {"SK_ID_CURR": sk_id_curr, **explicacao}


@app.get("/model/metrics", tags=["explicabilidade"])
def metricas_do_modelo() -> dict:
    """
    Métricas de discriminação do modelo, medidas out-of-fold.

    Por que out-of-fold e não sobre a base do painel: todos os clientes
    rotulados foram usados no treino. Reprevê-los daria um AUC otimista (~0,86
    em vez de 0,79) — um número que o modelo não entrega na vida real. As
    predições out-of-fold vêm de modelos que NÃO viram aquele cliente.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve

    from config import OOF_PREDICTIONS_PATH, TARGET_COLUMN

    meta = model_metadata()
    resposta = {
        "auc": meta["oof_auc"],
        "gini": round(2 * meta["oof_auc"] - 1, 6),
        "ks": None,
        "n_obs": None,
        "n_features": meta["n_features"],
        "fonte": "model_metadata.json",
    }

    if os.path.exists(OOF_PREDICTIONS_PATH):
        oof = pd.read_csv(OOF_PREDICTIONS_PATH)
        y, p = oof[TARGET_COLUMN].to_numpy(), oof["PD"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        auc = float(roc_auc_score(y, p))
        resposta.update({
            "auc":   round(auc, 6),
            "gini":  round(2 * auc - 1, 6),
            "ks":    round(float(np.max(tpr - fpr)), 6),  # maior separação entre as curvas
            "n_obs": int(len(y)),
            "fonte": "oof_predictions.csv",
        })
    return resposta


# ============================================================
# MONITORAMENTO
# ============================================================
# Expor o monitoramento pela API (e não deixar o painel ler o banco direto)
# mantém a mesma regra do resto: um único lugar concentra o acesso aos dados da
# solução, e qualquer consumidor — painel, alertas, um agente de IA — enxerga
# exatamente os mesmos números.

@app.get("/monitoring/runs", tags=["monitoramento"])
def execucoes_de_monitoramento(
    limite: int = Query(20, ge=1, le=100),
    tipo: Optional[str] = Query(None, description="data_drift, prediction_drift ou performance"),
) -> dict:
    """Histórico de execuções de monitoramento — a série que permite ver tendência."""
    from MLOps.monitoring import historico
    try:
        return {"execucoes": historico(limite=limite, tipo=tipo)}
    except Exception as erro:
        raise HTTPException(status_code=503, detail=f"Banco indisponivel: {erro}")


@app.get("/monitoring/runs/{run_id}/drift", tags=["monitoramento"])
def drift_de_uma_execucao(run_id: int = Path(..., description="ID da execução")) -> dict:
    """PSI de cada feature naquela execução."""
    from MLOps.monitoring import drift_da_execucao
    try:
        linhas = drift_da_execucao(run_id)
    except Exception as erro:
        raise HTTPException(status_code=503, detail=f"Banco indisponivel: {erro}")
    if not linhas:
        raise HTTPException(status_code=404, detail=f"Execucao {run_id} sem metricas de drift.")
    return {"run_id": run_id, "metricas": linhas}


@app.get("/monitoring/performance", tags=["monitoramento"])
def performance_por_safra(limite: int = Query(50, ge=1, le=200)) -> dict:
    """
    AUC/KS/Gini por safra, com a referência do treino.

    É a resposta para "o modelo ainda funciona?" — e a única camada que mede a
    perda de verdade, em vez de inferi-la de um proxy.
    """
    from MLOps.monitoring import serie_performance, AUC_REFERENCIA, QUEDA_TOLERADA
    try:
        return {
            "referencia_treino": AUC_REFERENCIA,
            "limite_de_alerta": round(AUC_REFERENCIA * (1 - QUEDA_TOLERADA), 6),
            "safras": serie_performance(limite=limite),
        }
    except Exception as erro:
        raise HTTPException(status_code=503, detail=f"Banco indisponivel: {erro}")


# ============================================================
# AUDITORIA
# ============================================================

@app.get("/predictions/recent", tags=["auditoria"])
def predicoes_recentes(limite: int = Query(20, ge=1, le=200)) -> dict:
    """
    Últimas decisões registradas.

    Toda predição servida por esta API fica gravada com a versão do modelo, a
    entrada recebida e a latência — é o que permite responder, meses depois,
    por que um cliente foi recusado.
    """
    try:
        return {"predicoes": store.recent_predictions(limite=limite)}
    except Exception as erro:
        raise HTTPException(status_code=503, detail=f"Banco indisponivel: {erro}")
