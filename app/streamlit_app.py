"""
streamlit_app.py — Painel de decisão de crédito com explicabilidade (SHAP).

É a face visual do modelo: o analista escolhe um cliente, vê a probabilidade de
inadimplência, a decisão recomendada e — o ponto central — *por que* o modelo
decidiu assim, feature a feature.

O PAINEL NÃO CARREGA O MODELO. Ele é um cliente HTTP da API:

    Painel (Streamlit) ──HTTP──► API (FastAPI) ──SQL──► PostgreSQL
                                      │
                                      └──► modelo (.joblib)

Isso não é detalhe de implementação, é a arquitetura: existe UM lugar que
pontua crédito, e todo mundo — o painel do analista, o sistema de originação,
um relatório de auditoria — passa por ele. Se o painel carregasse o modelo por
conta própria, teríamos duas fontes de decisão que poderiam divergir, e o log
de auditoria não veria o que foi decidido na tela.

Rodar:
    streamlit run app/streamlit_app.py
    # ou: docker compose -f MLOps/docker-compose.yml up -d streamlit  → :8501
"""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Endereço da API. Dentro do compose os serviços se enxergam pelo nome do
# serviço ("api"); rodando no host, é localhost.
API_URL = os.environ.get("API_URL", "http://api:8000")
TIMEOUT = 30

st.set_page_config(page_title="Credit Risk — Decisão e Explicabilidade",
                   page_icon="🏦", layout="wide")

# Cores semânticas (risco/proteção), separadas do resto da identidade visual
COR_RISCO    = "#C0392B"
COR_PROTECAO = "#1E8567"


# ============================================================
# CLIENTE HTTP DA API
# ============================================================

def api_get(caminho: str, **params):
    """GET na API, devolvendo (dados, erro)."""
    try:
        resposta = requests.get(f"{API_URL}{caminho}", params=params, timeout=TIMEOUT)
    except requests.RequestException as erro:
        return None, f"não consegui falar com a API em {API_URL}: {erro}"
    if resposta.status_code >= 400:
        return None, f"HTTP {resposta.status_code}: {resposta.text[:200]}"
    return resposta.json(), None


def api_post(caminho: str, corpo: dict = None, **params):
    """POST na API, devolvendo (dados, erro)."""
    try:
        resposta = requests.post(f"{API_URL}{caminho}", json=corpo, params=params, timeout=TIMEOUT)
    except requests.RequestException as erro:
        return None, f"não consegui falar com a API em {API_URL}: {erro}"
    if resposta.status_code >= 400:
        return None, f"HTTP {resposta.status_code}: {resposta.text[:200]}"
    return resposta.json(), None


@st.cache_data(ttl=60, show_spinner="Consultando a API...")
def carregar_saude():
    return api_get("/health")


@st.cache_data(ttl=300, show_spinner="Carregando métricas do modelo...")
def carregar_metricas():
    return api_get("/model/metrics")


@st.cache_data(ttl=300, show_spinner="Mapeando a origem das variáveis...")
def carregar_origens():
    return api_get("/features/origins")


@st.cache_data(ttl=300, show_spinner="Buscando clientes...")
def carregar_ids(limite: int = 50):
    return api_get("/clients", limite=limite, apenas_rotulados=True)


@st.cache_data(ttl=60, show_spinner="Buscando o cliente no banco...")
def carregar_cliente(sk_id: int):
    return api_get(f"/clients/{sk_id}", incluir_features=False)


# ============================================================
# COMPONENTES VISUAIS
# ============================================================

def gauge(pd_value: float, aprova_abaixo: float, recusa_acima: float) -> go.Figure:
    """Velocímetro da probabilidade de inadimplência, com as faixas de decisão."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pd_value * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#2C3E50", "thickness": 0.28},
            # As bandas espelham a política de crédito vinda do modelo
            "steps": [
                {"range": [0, aprova_abaixo * 100],              "color": "#D6EAE2"},
                {"range": [aprova_abaixo * 100,
                           recusa_acima * 100],                  "color": "#F7E7C8"},
                {"range": [recusa_acima * 100, 100],             "color": "#F2D3CE"},
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
        customdata=df["origem"],
        hovertemplate="<b>%{y}</b><br>origem: %{customdata}<br>impacto: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title="← protege o cliente   |   aumenta o risco →",
        yaxis_title=None,
        showlegend=False,
    )
    return fig


def barras_origem(por_origem: dict, simulaveis: int) -> go.Figure:
    """De onde vêm as features que o modelo usou para decidir."""
    itens = list(por_origem.items())
    fig = go.Figure(go.Bar(
        x=[v for _, v in itens][::-1],
        y=[k for k, _ in itens][::-1],
        orientation="h",
        marker_color="#2C6E9B",
        text=[v for _, v in itens][::-1],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x} variáveis<extra></extra>",
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=40, t=10, b=30),
                      xaxis_title="nº de variáveis", showlegend=False)
    return fig


# ============================================================
# APP
# ============================================================

st.title("🏦 Credit Risk — Decisão e Explicabilidade")
st.caption("Home Credit Default Risk · Projeto Final — MBA Big Data e Analytics (FIA LABDATA)")

saude, erro = carregar_saude()
if erro:
    st.error(
        f"**A API não respondeu.**\n\n{erro}\n\n"
        "O painel é um cliente da API — ele não pontua nada sozinho. Suba o serviço:\n\n"
        "```bash\ndocker compose -f MLOps/docker-compose.yml up -d postgres api\n```"
    )
    st.stop()

banco = saude.get("database", {})
if banco.get("status") != "ok":
    st.warning(
        f"A API está de pé, mas o **banco não respondeu** ({banco.get('erro', 'sem detalhe')}). "
        "Sem a feature store não há de onde buscar o histórico dos clientes."
    )
    st.stop()

politica = saude["model"]["decision_policy"]
aprova_abaixo = politica["approve_below"]
recusa_acima  = politica["reject_above"]

# ---------- Faixa de métricas do modelo ----------
# Todas as métricas vêm das predições OUT-OF-FOLD: são o desempenho que o modelo
# entrega em clientes que ele nunca viu, não o desempenho inflado sobre o treino.
metricas, _ = carregar_metricas()
metricas = metricas or {}

c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC-ROC", f"{metricas.get('auc', 0):.3f}",
          help="Medido em validação cruzada de 5 folds (out-of-fold): cada cliente "
               "foi pontuado por um modelo que não o viu no treino.")
c2.metric("Gini", f"{metricas.get('gini', 0):.3f}", help="Gini = 2 × AUC − 1.")
if metricas.get("ks") is not None:
    c3.metric("KS", f"{metricas['ks']:.3f}",
              help=f"Máxima separação entre as distribuições de bons e maus pagadores, "
                   f"sobre {metricas.get('n_obs', 0):,} clientes out-of-fold.".replace(",", "."))
else:
    c3.metric("KS", "—", help="Rode `python -m Model.train` para gerar as predições out-of-fold.")
c4.metric("Clientes na feature store", f"{banco['clientes_na_feature_store']:,}".replace(",", "."))

st.divider()

# ---------- Barra lateral: seleção e ajuste do cliente ----------
with st.sidebar:
    st.header("Cliente")

    sugestoes, _ = carregar_ids(50)
    ids_sugeridos = (sugestoes or {}).get("sk_id_curr", [])

    modo = st.radio("Como escolher o cliente", ["Da lista", "Digitar o ID"],
                    horizontal=True, label_visibility="collapsed")
    if modo == "Da lista" and ids_sugeridos:
        client_id = st.selectbox("ID do cliente (SK_ID_CURR)", options=ids_sugeridos)
    else:
        client_id = st.number_input("ID do cliente (SK_ID_CURR)", min_value=1,
                                    value=ids_sugeridos[0] if ids_sugeridos else 100002, step=1)
    client_id = int(client_id)

    cliente, erro_cliente = carregar_cliente(client_id)
    if erro_cliente:
        st.error(f"Cliente {client_id} não encontrado na feature store.")
        st.stop()

    atuais = cliente["variaveis_de_negocio"]

    st.divider()
    st.subheader("Simular alterações")
    st.caption(
        "Ajuste as variáveis que um analista realmente conhece e negocia. "
        "O resto do perfil vem do histórico do cliente, no banco."
    )

    overrides = {}

    def numero(rotulo, chave, passo, ajuda=None):
        """Campo numérico que só vira override se o analista mudar o valor."""
        atual = atuais.get(chave)
        if atual is None:
            return
        novo = st.number_input(rotulo, value=float(atual), format="%.0f", step=passo, help=ajuda)
        if novo != float(atual):
            overrides[chave] = novo

    numero("Renda anual (R$)", "AMT_INCOME_TOTAL", 5000.0)
    numero("Crédito solicitado (R$)", "AMT_CREDIT", 10000.0)
    numero("Parcela anual (R$)", "AMT_ANNUITY", 1000.0)

    dias_nascimento = atuais.get("DAYS_BIRTH") or -12000
    idade_atual = int(abs(dias_nascimento) / 365.25)
    idade = st.slider("Idade (anos)", 18, 80, idade_atual)
    if idade != idade_atual:
        overrides["DAYS_BIRTH"] = -idade * 365.25

    # EXT_SOURCE: scores de bureaus externos, entre 0 e 1. São as features mais
    # fortes do modelo — por isso valem um controle dedicado.
    #
    # Quando o cliente NÃO tem o score, o slider não pode simplesmente começar
    # em 0,5: isso imputaria em silêncio um score que ele não possui, e mudaria
    # a decisão. Ausência é informação — o modelo trata NaN nativamente. Por
    # isso, para score ausente, é preciso marcar a caixa para simular um valor.
    for i in (1, 2, 3):
        chave = f"EXT_SOURCE_{i}"
        atual = atuais.get(chave)

        if atual is None:
            simular = st.checkbox(f"EXT_SOURCE_{i} — ausente neste cliente. Simular um valor?",
                                  value=False, key=f"sim_{chave}")
            if simular:
                overrides[chave] = st.slider(f"EXT_SOURCE_{i} (score externo)",
                                             0.0, 1.0, 0.5, 0.01, key=f"sld_{chave}")
        else:
            novo = st.slider(f"EXT_SOURCE_{i} (score externo)",
                             0.0, 1.0, float(atual), 0.01, key=f"sld_{chave}")
            if abs(novo - float(atual)) > 1e-9:
                overrides[chave] = novo

    if overrides:
        st.info(f"**{len(overrides)} variável(is) alterada(s).** "
                "A API recalcula as features derivadas afetadas.")
    else:
        st.caption("Nenhuma alteração — o cliente está sendo avaliado como está no banco.")

# ---------- Predição (via API) ----------
resultado, erro_pred = api_post(f"/predict/{client_id}", corpo=overrides or None)
if erro_pred:
    st.error(f"A API não conseguiu pontuar este cliente.\n\n{erro_pred}")
    st.stop()

pd_value = resultado["probability_default"]
decisao  = resultado["decision"]

col_esq, col_dir = st.columns([1, 1.35])

with col_esq:
    st.subheader("Probabilidade de inadimplência")
    st.plotly_chart(gauge(pd_value, aprova_abaixo, recusa_acima), use_container_width=True)

    cores = {"APROVAR": "🟢", "ANALISE_MANUAL": "🟡", "RECUSAR": "🔴"}
    rotulos = {"APROVAR": "APROVAR", "ANALISE_MANUAL": "ANÁLISE MANUAL", "RECUSAR": "RECUSAR"}
    st.markdown(f"### {cores[decisao]} Decisão: **{rotulos[decisao]}**")
    st.caption(
        f"Faixa de risco: **{resultado['risk_band']}** · Política: aprova abaixo de "
        f"{aprova_abaixo:.0%}, recusa acima de {recusa_acima:.0%}."
    )

    # Se o cliente tem rótulo real, mostramos — é honesto e ajuda a demonstrar
    # que o modelo acerta (ou erra) em casos concretos.
    real = resultado.get("real_outcome")
    if real is not None:
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
    explicacao, erro_exp = api_post(f"/explain/{client_id}", corpo=overrides or None, top_n=12)
    if erro_exp:
        st.warning(f"Não foi possível explicar esta decisão: {erro_exp}")
    else:
        st.plotly_chart(waterfall(explicacao["contributions"]), use_container_width=True)

st.divider()

# ============================================================
# DE ONDE VÊM AS VARIÁVEIS  (a pergunta que o painel precisa responder)
# ============================================================
st.subheader("De onde vêm as variáveis desta decisão")

origens, _ = carregar_origens()
origens = origens or {"total": resultado["n_features_expected"], "por_origem": {}}

do_banco = resultado["features_from_store"]
alteradas = resultado["features_overridden"]
derivadas = resultado["derived_recalculated"]

o1, o2, o3 = st.columns(3)
o1.metric("Simuladas por você", len(alteradas),
          help="Variáveis que o analista alterou nesta simulação.")
o2.metric("Vindas do histórico (banco)", f"{do_banco - len(alteradas):,}".replace(",", "."),
          help="Carregadas da feature_store.abt: agregações do histórico do cliente.")
o3.metric("Derivadas recalculadas", len(derivadas),
          help="Features que dependem do que você alterou e foram refeitas para "
               "o cliente continuar coerente.")

esq, dir_ = st.columns([1.1, 1])
with esq:
    st.markdown(
        f"O modelo usa **{origens['total']} variáveis**. O formulário ao lado expõe "
        f"**{len(origens.get('simulaveis', []))}** — as que um analista conhece e negocia. "
        "Todas as outras são agregações do histórico do cliente, calculadas pelo "
        "pipeline e lidas do banco no momento da decisão:"
    )
    if origens.get("por_origem"):
        st.plotly_chart(barras_origem(origens["por_origem"], len(origens.get("simulaveis", []))),
                        use_container_width=True)

with dir_:
    if alteradas:
        st.markdown("**Nesta simulação você alterou:**")
        for variavel in alteradas:
            st.markdown(f"- `{variavel}`")
        if derivadas:
            st.markdown("**E a API recalculou automaticamente:**")
            for variavel in derivadas:
                st.markdown(f"- `{variavel}`")
            st.caption(
                "Sem esse recálculo, o modelo receberia um cliente impossível — "
                "renda nova com endividamento antigo."
            )
    else:
        st.markdown("**Nenhuma variável alterada.**")
        st.caption(
            "Mexa nos controles à esquerda para simular. As features derivadas "
            "que dependerem do que você mudar são recalculadas pela API."
        )

    st.markdown(f"**Qualidade da entrada:** {resultado['n_features_missing']} de "
                f"{resultado['n_features_expected']} features vazias para este cliente.")
    st.caption(
        "Ausência é informação: quem nunca teve crédito não tem histórico de bureau. "
        "O LightGBM trata o ausente nativamente — mas o número fica visível, e não escondido."
    )

with st.expander("Sobre o modelo e a arquitetura"):
    st.markdown(
        f"**Fluxo desta tela:** painel (Streamlit) → API (FastAPI, `{API_URL}`) → "
        "PostgreSQL (feature store) → modelo LightGBM.\n\n"
        "O painel não carrega o modelo: quem pontua é a API, e toda decisão exibida "
        "aqui fica registrada em `serving.predictions` para auditoria."
    )
    st.json({
        "modelo":              saude["model"]["model_type"],
        "versao":              saude.get("model_version"),
        "AUC (out-of-fold)":   metricas.get("auc"),
        "features":            saude["model"]["n_features"],
        "árvores":             saude["model"]["n_estimators"],
        "treinado em":         saude["model"]["trained_at"],
        "política de decisão": politica,
        "clientes no banco":   banco["clientes_na_feature_store"],
    })
