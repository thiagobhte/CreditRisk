"""
streamlit_app.py — Painel de decisão de crédito com explicabilidade (SHAP).

É a face visual do modelo: o analista escolhe um cliente, vê a probabilidade de
inadimplência, a decisão recomendada e — o ponto central — *por que* o modelo
decidiu assim, feature a feature.

Assim como a API, este app consome direto os artefatos do modelo
(Model/artifacts/), espelhando o diagrama da arquitetura: o Model alimenta em
paralelo a API (integração máquina-a-máquina) e o Streamlit (analista humano).

Rodar:
    streamlit run app/streamlit_app.py
    # ou: docker compose -f MLOps/docker-compose.yml up -d streamlit  → :8501
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Raiz do projeto no path (o Streamlit roda a partir de onde foi invocado)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    ID_COLUMN, TARGET_COLUMN, OOF_PREDICTIONS_PATH, DEMO_CLIENTS_PATH,
    DECISION_APPROVE_BELOW, DECISION_REJECT_ABOVE,
)
from Model.predict import predict, model_metadata
from Model.explain import explain_client, global_importance

st.set_page_config(page_title="Credit Risk — Decisão e Explicabilidade",
                   page_icon="🏦", layout="wide")

# Cores semânticas (risco/proteção), separadas do resto da identidade visual
COR_RISCO    = "#C0392B"
COR_PROTECAO = "#1E8567"


# ============================================================
# CARREGAMENTO (em cache — não recarrega a cada interação)
# ============================================================

@st.cache_data(show_spinner="Carregando clientes...")
def load_clients() -> pd.DataFrame:
    """Amostra de clientes da ABT para a demonstração (gerada de abt.csv)."""
    return pd.read_csv(DEMO_CLIENTS_PATH)


@st.cache_data(show_spinner="Calculando importância global (SHAP)...")
def load_global_importance(n: int = 300) -> pd.DataFrame:
    """Importância global por SHAP, medida numa amostra de clientes."""
    df = load_clients().head(n).drop(columns=[TARGET_COLUMN], errors="ignore")
    return global_importance(df, top_n=15)


@st.cache_data(show_spinner="Calculando métricas out-of-fold...")
def oof_metrics() -> dict:
    """
    Métricas a partir das predições OUT-OF-FOLD salvas pelo train.py.

    Por que não medir sobre a amostra de clientes deste painel:
        todos os clientes rotulados foram usados no treino. Reprevê-los daria um
        AUC otimista (~0,86 em vez de 0,79) — um número que o modelo não entrega
        na vida real. As predições out-of-fold vêm de modelos que NÃO viram
        aquele cliente, e por isso são a única base honesta para KS e Gini.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    if not os.path.exists(OOF_PREDICTIONS_PATH):
        return {}

    oof = pd.read_csv(OOF_PREDICTIONS_PATH)
    y, p = oof[TARGET_COLUMN].to_numpy(), oof["PD"].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    return {
        "auc":  float(auc),
        "gini": float(2 * auc - 1),
        "ks":   float(np.max(tpr - fpr)),   # KS = maior separação entre as curvas
        "n":    int(len(y)),
    }


# ============================================================
# COMPONENTES VISUAIS
# ============================================================

def gauge(pd_value: float) -> go.Figure:
    """Velocímetro da probabilidade de inadimplência, com as faixas de decisão."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pd_value * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#2C3E50", "thickness": 0.28},
            # As bandas espelham a política de crédito do config.py
            "steps": [
                {"range": [0, DECISION_APPROVE_BELOW * 100],   "color": "#D6EAE2"},
                {"range": [DECISION_APPROVE_BELOW * 100,
                           DECISION_REJECT_ABOVE * 100],       "color": "#F7E7C8"},
                {"range": [DECISION_REJECT_ABOVE * 100, 100],  "color": "#F2D3CE"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def waterfall(contribs: list) -> go.Figure:
    """
    Barras horizontais das features que mais moveram a decisão DESTE cliente.

    Barras para a direita (vermelho) aumentaram o risco; para a esquerda (verde)
    protegeram o cliente. É a resposta à pergunta "por que ele foi recusado?".
    """
    df = pd.DataFrame(contribs).iloc[::-1]   # maior impacto no topo do gráfico
    fig = go.Figure(go.Bar(
        x=df["shap"],
        y=df["feature"],
        orientation="h",
        marker_color=[COR_RISCO if s > 0 else COR_PROTECAO for s in df["shap"]],
        hovertemplate="<b>%{y}</b><br>impacto: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title="← protege o cliente   |   aumenta o risco →",
        yaxis_title=None,
        showlegend=False,
    )
    return fig


# ============================================================
# APP
# ============================================================

st.title("🏦 Credit Risk — Decisão e Explicabilidade")
st.caption("Home Credit Default Risk · Projeto Final — MBA Big Data e Analytics (FIA LABDATA)")

if not os.path.exists(DEMO_CLIENTS_PATH):
    st.error(
        f"Arquivo de demonstração não encontrado: `{DEMO_CLIENTS_PATH}`.\n\n"
        "Ele é gerado pelo treino — rode `python -m Model.train`."
    )
    st.stop()

try:
    meta = model_metadata()
except FileNotFoundError:
    st.error("Modelo não encontrado. Treine primeiro com `python -m Model.train`.")
    st.stop()

clients = load_clients()

# ---------- Faixa de métricas do modelo ----------
# Todas as métricas vêm das predições OUT-OF-FOLD: são o desempenho que o modelo
# entrega em clientes que ele nunca viu, não o desempenho inflado sobre o treino.
m = oof_metrics()
auc_oof = m.get("auc", meta["oof_auc"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC-ROC", f"{auc_oof:.3f}",
          help="Medido em validação cruzada de 5 folds (out-of-fold): cada cliente "
               "foi pontuado por um modelo que não o viu no treino.")
c2.metric("Gini", f"{m.get('gini', 2 * auc_oof - 1):.3f}",
          help="Gini = 2 × AUC − 1.")
if "ks" in m:
    c3.metric("KS", f"{m['ks']:.3f}",
              help=f"Máxima separação entre as distribuições de bons e maus pagadores, "
                   f"sobre {m['n']:,} clientes out-of-fold.".replace(",", "."))
else:
    c3.metric("KS", "—", help="Rode `python -m Model.train` para gerar as predições out-of-fold.")
c4.metric("Features do modelo", f"{meta['n_features']:,}".replace(",", "."))

st.divider()

# ---------- Barra lateral: seleção e ajuste do cliente ----------
with st.sidebar:
    st.header("Cliente")

    client_id = st.selectbox(
        "ID do cliente (SK_ID_CURR)",
        options=clients[ID_COLUMN].tolist(),
        help="Clientes retirados da ABT. A amostra mistura adimplentes e inadimplentes.",
    )
    row = clients[clients[ID_COLUMN] == client_id].iloc[0]
    record = row.drop(labels=[TARGET_COLUMN], errors="ignore").to_dict()

    st.divider()
    st.subheader("Simular alterações")
    st.caption("Ajuste as variáveis-chave e veja a decisão mudar em tempo real.")

    # Só as variáveis que um analista de crédito realmente conhece/negocia.
    # (O modelo usa 836 features — as demais vêm do histórico do cliente.)
    def num(label, key, fmt="%.0f", step=None, help=None):
        atual = float(record.get(key, 0) or 0)
        return st.number_input(label, value=atual, format=fmt, step=step, help=help)

    record["AMT_INCOME_TOTAL"] = num("Renda anual (R$)", "AMT_INCOME_TOTAL", step=5000.0)
    record["AMT_CREDIT"]       = num("Crédito solicitado (R$)", "AMT_CREDIT", step=10000.0)
    record["AMT_ANNUITY"]      = num("Parcela anual (R$)", "AMT_ANNUITY", step=1000.0)

    idade_atual = int(abs(record.get("DAYS_BIRTH", -12000)) / 365.25)
    idade = st.slider("Idade (anos)", 18, 80, idade_atual)
    record["DAYS_BIRTH"] = -idade * 365.25

    # EXT_SOURCE: scores de bureaus externos, entre 0 e 1. São as features mais
    # fortes do modelo — por isso valem um controle dedicado.
    for i in (1, 2, 3):
        k = f"EXT_SOURCE_{i}"
        v = record.get(k)
        v = 0.5 if (v is None or pd.isna(v)) else float(v)
        record[k] = st.slider(f"EXT_SOURCE_{i} (score externo)", 0.0, 1.0, v, 0.01)

    # Features derivadas dependem das acima: se o usuário mexeu na renda ou no
    # crédito e não recalculássemos, o modelo receberia um cliente incoerente.
    credito = record["AMT_CREDIT"] or np.nan
    renda   = record["AMT_INCOME_TOTAL"] or np.nan
    record["PAYMENT_RATE"]        = record["AMT_ANNUITY"] / credito
    record["ANNUITY_INCOME_PERC"] = record["AMT_ANNUITY"] / renda
    record["INCOME_CREDIT_PERC"]  = renda / credito

# ---------- Predição ----------
resultado = predict(record)[0]
pd_value = resultado["probability_default"]
decisao  = resultado["decision"]

col_esq, col_dir = st.columns([1, 1.35])

with col_esq:
    st.subheader("Probabilidade de inadimplência")
    st.plotly_chart(gauge(pd_value), use_container_width=True)

    cores = {"APROVAR": "🟢", "ANALISE_MANUAL": "🟡", "RECUSAR": "🔴"}
    rotulos = {"APROVAR": "APROVAR", "ANALISE_MANUAL": "ANÁLISE MANUAL", "RECUSAR": "RECUSAR"}
    st.markdown(f"### {cores[decisao]} Decisão: **{rotulos[decisao]}**")
    st.caption(
        f"Faixa de risco: **{resultado['risk_band']}** · Política: aprova abaixo de "
        f"{DECISION_APPROVE_BELOW:.0%}, recusa acima de {DECISION_REJECT_ABOVE:.0%}."
    )

    # Se o cliente tem rótulo real, mostramos — é honesto e ajuda a demonstrar
    # que o modelo acerta (ou erra) em casos concretos.
    real = row.get(TARGET_COLUMN)
    if pd.notna(real):
        st.info(
            f"**Desfecho real deste cliente:** "
            f"{'inadimplente (TARGET = 1)' if real == 1 else 'pagou o empréstimo (TARGET = 0)'}"
        )

with col_dir:
    st.subheader("Por que o modelo decidiu assim?")
    st.caption(
        "Contribuição de cada variável para ESTA decisão (valores SHAP). "
        "É a explicação individual exigida em crédito — não a importância média do modelo."
    )
    exp = explain_client(record, top_n=12)
    st.plotly_chart(waterfall(exp["contributions"]), use_container_width=True)

st.divider()

# ---------- Visão global ----------
st.subheader("Importância global das variáveis (SHAP)")
st.caption(
    "Impacto médio de cada variável sobre as predições, medido numa amostra de clientes. "
    "Diferente da importância nativa do LightGBM (que conta divisões de árvore), "
    "esta mede o efeito real na probabilidade."
)

gi = load_global_importance()
fig = go.Figure(go.Bar(
    x=gi["importance"][::-1],
    y=gi["feature"][::-1],
    orientation="h",
    marker_color="#2C6E9B",
    hovertemplate="<b>%{y}</b><br>impacto médio: %{x:.4f}<extra></extra>",
))
fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=30),
                  xaxis_title="Impacto médio absoluto (SHAP)", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Sobre o modelo"):
    st.json({
        "modelo":              meta["model_type"],
        "AUC (out-of-fold)":   meta["oof_auc"],
        "features":            meta["n_features"],
        "árvores":             meta["n_estimators"],
        "treinado em":         meta["trained_at"],
        "política de decisão": meta["decision_policy"],
    })
