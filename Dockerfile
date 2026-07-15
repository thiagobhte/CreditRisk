# ============================================================
# Imagem única do projeto — usada tanto pela API (app/) quanto
# pelo pipeline de dados/treino (DataPipeline/, Model/).
# O docker-compose escolhe o comando (uvicorn / python -m ...) por serviço.
# ============================================================
FROM python:3.11-slim

# Não gera .pyc e força stdout/stderr sem buffer (logs aparecem na hora)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /project

# Dependência de sistema do LightGBM: libgomp1 (runtime OpenMP).
# A imagem slim não a inclui, e sem ela "import lightgbm" quebra no container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências primeiro (camada cacheável — só reinstala se mudar).
# timeout/retries generosos: o shap arrasta llvmlite e numba (dezenas de MB) e o
# build quebrava com ReadTimeoutError no default do pip (15s).
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

# Copia o restante do código
COPY . .

# DATA_DIR aponta para um volume montado no compose (dados ficam fora da imagem)
ENV DATA_DIR=/data

EXPOSE 8000

# Default: sobe a API. O pipeline sobrescreve o command no docker-compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
