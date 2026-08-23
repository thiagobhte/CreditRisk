"""
1_Monitoramento.py — Página de monitoramento do modelo em produção.

Responde à pergunta que a decisão individual não responde: **o modelo continua
funcionando?** Mostra as três camadas, da mais rápida de detectar para a mais
lenta e definitiva:

    1. Data drift (PSI por feature)   → sinal antecipado
    2. Prediction drift (PSI da PD)   → pega quebra de pipeline
    3. Performance por safra (AUC/KS) → a medida real, com label lag

Como o painel principal, esta página é um cliente HTTP da API — não lê o banco
por conta própria.
"""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
TIMEOUT = 60

st.set_page_config(page_title="Monitoramento — Credit Risk", page_icon="📈", layout="wide")

COR_OK       = "#1E8567"
COR_MODERADO = "#D68910"
COR_SEVERO   = "#C0392B"
CORES_STATUS = {"OK": COR_OK, "MODERADO": COR_MODERADO, "SEVERO": COR_SEVERO}


def api_get(caminho: str, **params):
    try:
        resposta = requests.get(f"{API_URL}{caminho}", params=params, timeout=TIMEOUT)
    except requests.RequestException as erro:
        return None, f"não consegui falar com a API em {API_URL}: {erro}"
    if resposta.status_code >= 400:
        return None, f"HTTP {resposta.status_code}: {resposta.text[:200]}"
    return resposta.json(), None


@st.cache_data(ttl=30, show_spinner="Carregando monitoramento...")
def carregar_execucoes(limite=40):
    return api_get("/monitoring/runs", limite=limite)


@st.cache_data(ttl=30, show_spinner="Carregando performance...")
def carregar_performance():
    return api_get("/monitoring/performance")


@st.cache_data(ttl=30)
def carregar_drift(run_id: int):
    return api_get(f"/monitoring/runs/{run_id}/drift")


st.title("📈 Monitoramento do modelo em produção")
st.caption(
    "Um modelo de crédito degrada sem ninguém mexer nele — o mundo muda. "
    "Estas três camadas existem para detectar isso antes do prejuízo."
)

dados, erro = carregar_execucoes()
if erro:
    st.error(
        f"**A API não respondeu.**\n\n{erro}\n\n"
        "```bash\ndocker compose -f MLOps/docker-compose.yml up -d postgres api\n```"
    )
    st.stop()

execucoes = dados["execucoes"]
if not execucoes:
    st.warning(
        "Nenhuma execução de monitoramento registrada ainda.\n\n"
        "Gere o histórico de decisões e rode as três camadas:\n\n"
        "```bash\n"
        "python -m MLOps.simulate_production\n"
        "python -m MLOps.monitoring --tudo\n"
        "```"
    )
    st.stop()

# ---------- Situação atual ----------
ultima = execucoes[0]
alertas = [e for e in execucoes[:6] if e["retrain_recommended"]]

if alertas:
    tipos = ", ".join(sorted({e["run_type"] for e in alertas}))
    st.error(
        f"### 🔴 Re-treino recomendado\n"
        f"Disparado por: **{tipos}**. "
        f"Última verificação: {ultima['created_at'][:16].replace('T', ' ')}."
    )
else:
    st.success(f"### 🟢 Modelo estável\nÚltima verificação: "
               f"{ultima['created_at'][:16].replace('T', ' ')}.")

st.divider()

aba_perf, aba_dados, aba_pred, aba_hist = st.tabs(
    ["Performance (AUC/KS)", "Drift de dados (PSI)", "Drift de predições", "Histórico"]
)

# ============================================================
# 1. PERFORMANCE POR SAFRA
# ============================================================
with aba_perf:
    st.subheader("Performance por safra")
    st.caption(
        "A medida definitiva — e a mais lenta. No crédito o desfecho só se "
        "materializa meses depois da decisão (label lag), então esta curva sempre "
        "olha para trás. As outras duas abas são os proxies que seguram a decisão "
        "até este número chegar."
    )

    perf, erro_perf = carregar_performance()
    safras = (perf or {}).get("safras", [])

    if not safras:
        st.info("Nenhuma safra com desfecho conhecido ainda. "
                "Rode `python -m MLOps.monitoring --performance`.")
    else:
        df = pd.DataFrame(safras).sort_values("safra")
        df = df.drop_duplicates(subset=["safra"], keep="first")
        referencia = perf["referencia_treino"]
        limite = perf["limite_de_alerta"]

        recente = df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC da safra mais recente", f"{recente['auc']:.3f}",
                  delta=f"{recente['auc'] - referencia:+.3f} vs treino")
        c2.metric("KS", f"{recente['ks']:.3f}")
        c3.metric("Inadimplência observada", f"{recente['default_rate']:.2%}")
        c4.metric("Decisões na safra", f"{int(recente['n_obs']):,}".replace(",", "."))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["safra"], y=df["auc"], mode="lines+markers+text", name="AUC da safra",
            line=dict(color="#2C6E9B", width=3), marker=dict(size=10),
            text=[f"{v:.3f}" for v in df["auc"]], textposition="top center",
        ))
        fig.add_hline(y=referencia, line_dash="dash", line_color=COR_OK,
                      annotation_text=f"AUC do treino ({referencia:.3f})",
                      annotation_position="top left")
        fig.add_hline(y=limite, line_dash="dot", line_color=COR_SEVERO,
                      annotation_text=f"limite de alerta ({limite:.3f})",
                      annotation_position="bottom left")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_title="AUC-ROC", xaxis_title="safra")
        st.plotly_chart(fig, use_container_width=True)

        if recente["auc"] < limite:
            st.error(
                f"**A safra {recente['safra']} caiu para AUC {recente['auc']:.3f}**, abaixo do "
                f"limite de {limite:.3f} (5% abaixo do treino). O modelo perdeu poder de "
                "discriminação nesta população — é o gatilho de re-treino."
            )

        st.dataframe(
            df[["safra", "n_obs", "auc", "ks", "gini", "default_rate"]]
              .rename(columns={"safra": "Safra", "n_obs": "Decisões", "auc": "AUC",
                               "ks": "KS", "gini": "Gini", "default_rate": "Inadimplência"}),
            use_container_width=True, hide_index=True,
        )

# ============================================================
# 2. DATA DRIFT
# ============================================================
with aba_dados:
    st.subheader("Drift de dados (PSI por feature)")
    st.caption(
        "Compara a distribuição de cada variável em produção contra a população de "
        "treino. É o sinal mais antecipado: aparece assim que a entrada muda, muito "
        "antes de qualquer perda medida. PSI > 0,25 = severo."
    )

    de_dados = [e for e in execucoes if e["run_type"] == "data_drift"]
    if not de_dados:
        st.info("Nenhuma execução de drift de dados. Rode `python -m MLOps.monitoring --data-drift`.")
    else:
        rotulos = {
            f"#{e['run_id']} · {e['created_at'][:16].replace('T', ' ')} · {e['current_label'][:45]}":
            e for e in de_dados
        }
        escolhido = rotulos[st.selectbox("Execução", list(rotulos.keys()))]

        c1, c2, c3 = st.columns(3)
        c1.metric("Features avaliadas", escolhido["n_features_evaluated"])
        c2.metric("Drift severo", escolhido["n_severe"] or 0)
        c3.metric("Drift moderado", escolhido["n_moderate"] or 0)
        st.caption(f"Referência: {escolhido['reference_label']} · "
                   f"Comparado com: {escolhido['current_label']}")

        metricas, _ = carregar_drift(escolhido["run_id"])
        linhas = (metricas or {}).get("metricas", [])
        if linhas:
            d = pd.DataFrame(linhas).head(15).iloc[::-1]
            d["psi"] = pd.to_numeric(d["psi"], errors="coerce").fillna(0)
            fig = go.Figure(go.Bar(
                x=d["psi"], y=d["feature"], orientation="h",
                marker_color=[CORES_STATUS.get(s, COR_OK) for s in d["status"]],
                hovertemplate="<b>%{y}</b><br>PSI: %{x:.4f}<extra></extra>",
            ))
            fig.add_vline(x=0.10, line_dash="dot", line_color=COR_MODERADO,
                          annotation_text="moderado")
            fig.add_vline(x=0.25, line_dash="dash", line_color=COR_SEVERO,
                          annotation_text="severo")
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10),
                              xaxis_title="PSI", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 3. PREDICTION DRIFT
# ============================================================
with aba_pred:
    st.subheader("Drift de predições")
    st.caption(
        "PSI sobre a distribuição das probabilidades que o modelo está devolvendo, "
        "comparada com a do treino. Pega o que o drift por feature não pega: se o "
        "pipeline quebrar e uma variável chegar sempre nula, cada PSI individual "
        "pode ficar abaixo do limiar, mas as decisões deslocam em bloco."
    )

    de_pred = [e for e in execucoes if e["run_type"] == "prediction_drift"]
    if not de_pred:
        st.info("Nenhuma execução. Rode `python -m MLOps.monitoring --prediction-drift`.")
    else:
        atual = de_pred[0]
        resumo = atual.get("summary") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("PD média no treino", f"{resumo.get('pd_media_treino', 0):.4f}")
        c2.metric("PD média em produção", f"{resumo.get('pd_media_producao', 0):.4f}",
                  delta=f"{resumo.get('pd_media_producao', 0) - resumo.get('pd_media_treino', 0):+.4f}")
        c3.metric("PSI da distribuição", f"{resumo.get('psi_pd', 0):.4f}",
                  help="PSI > 0,25 = severo")
        st.caption(f"{resumo.get('n_predicoes', 0):,} predições · {atual['current_label']}"
                   .replace(",", "."))

        mix = resumo.get("mix_de_decisoes") or {}
        if mix:
            ordem = ["APROVAR", "ANALISE_MANUAL", "RECUSAR"]
            cores = {"APROVAR": COR_OK, "ANALISE_MANUAL": COR_MODERADO, "RECUSAR": COR_SEVERO}
            fig = go.Figure(go.Bar(
                x=[k for k in ordem if k in mix],
                y=[mix[k] for k in ordem if k in mix],
                marker_color=[cores[k] for k in ordem if k in mix],
                text=[f"{mix[k]:.1%}" for k in ordem if k in mix], textposition="outside",
            ))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_title="proporção das decisões", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "O mix de decisões é o que o negócio sente primeiro: se a fatia de "
                "recusa cresce sem explicação, alguém precisa olhar antes de o "
                "prejuízo (ou a receita perdida) aparecer no resultado."
            )

# ============================================================
# 4. HISTÓRICO
# ============================================================
with aba_hist:
    st.subheader("Histórico de execuções")
    st.caption(
        "Cada execução fica registrada em `mlops.monitoring_runs`. É a série "
        "histórica que permite ver tendência — antes, o relatório era um JSON "
        "sobrescrito a cada rodada, o que dá uma foto, não um monitoramento."
    )
    h = pd.DataFrame([{
        "Execução": e["run_id"],
        "Quando": e["created_at"][:16].replace("T", " "),
        "Camada": e["run_type"],
        "Avaliadas": e["n_features_evaluated"],
        "Severo": e["n_severe"],
        "Moderado": e["n_moderate"],
        "Re-treino": "SIM" if e["retrain_recommended"] else "—",
        "Comparado com": e["current_label"],
    } for e in execucoes])
    st.dataframe(h, use_container_width=True, hide_index=True)
