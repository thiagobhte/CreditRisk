"""
app/main.py — API REST de predição de risco de crédito (FastAPI).

Expõe o modelo treinado como um serviço HTTP. É a camada de "deploy do modelo
como serviço de predição" pedida na etapa individual.

Endpoints:
    GET  /health         → liveness + metadados do modelo (AUC, data, nº features)
    POST /predict        → 1 cliente  (JSON de features no nível da ABT)
    POST /predict/batch  → N clientes ({"clients": [ {...}, {...} ]})

Rodar localmente:
    uvicorn app.main:app --reload --port 8000
Documentação interativa (Swagger) em: http://localhost:8000/docs
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Importa a lógica de inferência já pronta (carrega o modelo 1x, em cache).
# O import é do pacote Model na raiz do projeto (WORKDIR do container).
from Model.predict import predict, model_metadata

app = FastAPI(
    title="Home Credit Default Risk — Serviço de Predição",
    description="Retorna a probabilidade de inadimplência (PD) e a decisão de crédito.",
    version="1.0.0",
)


# ============================================================
# SCHEMAS DE ENTRADA/SAÍDA
# ============================================================

class ClientFeatures(BaseModel):
    """
    Features de UM cliente no nível da ABT.

    Aceita campos arbitrários (extra='allow') porque a ABT tem centenas de
    features e listá-las todas aqui seria impraticável e frágil. O predict.py
    alinha o que chegar ao formato do modelo; o que faltar vira NaN.
    """
    model_config = {"extra": "allow"}

    SK_ID_CURR: Optional[int] = Field(
        default=None, description="ID do cliente (opcional, apenas ecoado na resposta)"
    )


class BatchRequest(BaseModel):
    """Lote de clientes para o endpoint /predict/batch."""
    clients: List[Dict[str, Any]] = Field(..., description="Lista de clientes (features da ABT)")


class PredictionResponse(BaseModel):
    SK_ID_CURR: Optional[int] = None
    probability_default: float = Field(..., description="Probabilidade de default (0 a 1)")
    risk_band: str = Field(..., description="BAIXO / MODERADO / ALTO / MUITO_ALTO")
    decision: str = Field(..., description="APROVAR / ANALISE_MANUAL / RECUSAR")


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", tags=["infra"])
def health() -> dict:
    """
    Verifica se o serviço está de pé e se o modelo carregou.

    Usado por orquestradores (docker-compose healthcheck, Kubernetes) e pelo
    monitoramento. Devolve os metadados do modelo para rastreabilidade
    (qual versão/AUC está servindo agora).
    """
    try:
        meta = model_metadata()
        return {"status": "ok", "model": meta}
    except FileNotFoundError as e:
        # Modelo ainda não treinado → serviço "vivo" mas não "pronto"
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["predição"])
def predict_one(client: ClientFeatures) -> dict:
    """Prediz o risco de UM cliente."""
    try:
        # model_dump inclui os campos extras (as features da ABT)
        result = predict(client.model_dump())
        return result[0]
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na predição: {e}")


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["predição"])
def predict_batch(req: BatchRequest) -> list:
    """Prediz o risco de vários clientes de uma vez."""
    if not req.clients:
        raise HTTPException(status_code=400, detail="Lista 'clients' vazia.")
    try:
        return predict(req.clients)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na predição: {e}")
