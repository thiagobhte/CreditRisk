"""
monitoring.py — Monitoramento de dados e do modelo em produção.

Cobre o item (iii) da etapa individual: como detectar falhas, perda de
performance e mudança de comportamento dos dados.

Um modelo de crédito degrada **sem ninguém mexer nele**, porque o mundo muda
(inflação, sazonalidade, novo público). Monitoramos três camadas, das mais
rápidas de detectar para as mais lentas:

1. DATA DRIFT (PSI — Population Stability Index)
   Compara a distribuição de cada feature em produção contra a referência (a
   população de treino). É o sinal mais ANTECIPADO: aparece assim que os dados
   de entrada mudam, muito antes de qualquer perda medida.
       PSI < 0.10  → estável
       0.10–0.25   → drift moderado (investigar)
       PSI > 0.25  → drift severo (candidato a re-treino)

2. PREDICTION DRIFT
   PSI sobre a distribuição das PDs que o modelo está devolvendo, comparada com
   a distribuição out-of-fold do treino. Pega problemas que o data drift por
   feature não pega — por exemplo, uma quebra no pipeline de features que zera
   uma coluna e desloca todas as decisões.

3. PERFORMANCE (AUC/KS/Gini por safra)
   A medida definitiva — e a mais LENTA. No crédito, o TARGET só se materializa
   meses depois (o "label lag"). Só dá para calcular sobre safras já maduras,
   e é por isso que as duas camadas acima existem: são os proxies que seguram
   a decisão até o número real chegar.

Tudo é PERSISTIDO em `mlops.monitoring_runs` + `drift_metrics` +
`performance_metrics`. Antes, o relatório era um JSON sobrescrito a cada
execução — não havia série histórica, e sem série histórica não há
monitoramento, só uma foto.

Uso:
    python -m MLOps.monitoring --data-drift
    python -m MLOps.monitoring --data-drift --simular-choque
    python -m MLOps.monitoring --prediction-drift
    python -m MLOps.monitoring --performance
    python -m MLOps.monitoring --tudo
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import (
    NON_FEATURE_COLS, TARGET_COLUMN, OOF_PREDICTIONS_PATH, FEATURE_IMPORTANCE_PATH,
)
from MLOps.db import get_engine

# Limiares de PSI — convenção de mercado (crédito/risco)
PSI_MODERATE = 0.10
PSI_SEVERE   = 0.25

# Quantas features acompanhar. Monitorar as 836 a cada execução seria caro e
# pouco útil: o sinal está nas que o modelo realmente usa. Acompanhamos as mais
# importantes — se uma delas driftar, a decisão muda.
TOP_FEATURES_MONITORADAS = int(os.environ.get("MONITOR_TOP_FEATURES", "30"))

# Referência de performance: o AUC out-of-fold do modelo em produção.
AUC_REFERENCIA = 0.790922
# Quanto de queda se tolera antes de recomendar re-treino. 5% sobre 0,7909 dá
# um piso de 0,751 — abaixo disso o modelo perdeu discriminação de verdade,
# não é só variação de amostra.
QUEDA_TOLERADA = 0.05
# Safras menores que isso têm AUC instável demais para acionar um alerta.
TAMANHO_MINIMO_SAFRA = 300


# ============================================================
# POPULATION STABILITY INDEX
# ============================================================

def _psi_single(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Calcula o PSI de uma variável entre referência e produção.

    Usa bins por quantis da referência (decis): assim os buckets têm tamanho
    parecido no baseline, e o PSI mede quanto a massa se deslocou entre eles.
    """
    reference = reference[~np.isnan(reference)]
    current   = current[~np.isnan(current)]
    if reference.size == 0 or current.size == 0:
        return np.nan

    # Bordas por quantis da referência; únicas para evitar bins degenerados
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0  # variável constante — sem como driftar
    edges[0], edges[-1] = -np.inf, np.inf  # captura valores fora do range de treino

    ref_pct = np.histogram(reference, bins=edges)[0] / reference.size
    cur_pct = np.histogram(current,   bins=edges)[0] / current.size

    # epsilon evita log(0) / divisão por 0 em buckets vazios
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _classificar(psi: float) -> str:
    if psi is None or (isinstance(psi, float) and np.isnan(psi)):
        return "OK"
    return "SEVERO" if psi >= PSI_SEVERE else "MODERADO" if psi >= PSI_MODERATE else "OK"


def compute_data_drift(reference: pd.DataFrame, current: pd.DataFrame,
                       top_n: int = None) -> pd.DataFrame:
    """
    Calcula o PSI das variáveis numéricas comuns aos dois conjuntos.

    Retorna um DataFrame ordenado do maior drift para o menor, com o status
    (OK / MODERADO / SEVERO) de cada variável.
    """
    # Compara em minusculo: o banco devolve "sk_id_curr"/"target", enquanto
    # NON_FEATURE_COLS lista "SK_ID_CURR"/"TARGET". Sem isso, o ID do cliente
    # entrava como se fosse feature e aparecia no topo do drift — um numero
    # sequencial sempre "drifta", e o alerta seria puro ruido.
    ignorar = {c.lower() for c in NON_FEATURE_COLS + [TARGET_COLUMN]}
    feats = [c for c in reference.columns
             if c.lower() not in ignorar
             and c in current.columns
             and pd.api.types.is_numeric_dtype(reference[c])]

    linhas = []
    for c in feats:
        psi = _psi_single(reference[c].to_numpy(dtype="float64"),
                          current[c].to_numpy(dtype="float64"))
        linhas.append({"feature": c, "psi": psi, "status": _classificar(psi)})

    resultado = pd.DataFrame(linhas).sort_values("psi", ascending=False, na_position="last")
    if top_n:
        resultado = resultado.head(top_n)
    return resultado.reset_index(drop=True)


# ============================================================
# LEITURA DA POPULAÇÃO A PARTIR DO BANCO
# ============================================================

def features_monitoradas(top_n: int = None) -> list:
    """
    Quais features acompanhar: as mais importantes para o modelo.

    Lê a importância média entre folds gerada pelo train.py. Se o arquivo não
    existir, cai para um conjunto mínimo conhecido — o monitoramento não pode
    simplesmente parar de funcionar porque um CSV auxiliar sumiu.
    """
    top_n = top_n or TOP_FEATURES_MONITORADAS
    padrao = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "PAYMENT_RATE",
              "DAYS_BIRTH", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"]

    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        return padrao

    fi = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    media = fi.groupby("feature")["importance"].mean().sort_values(ascending=False)
    return media.head(top_n).index.tolist()


def _total_estimado() -> int:
    """
    Estimativa rápida de quantos clientes há na feature store.

    Usa `reltuples` do catálogo (mantido pelo autovacuum) em vez de COUNT(*):
    contar 356 mil linhas de uma tabela de 3 GB é uma varredura completa, e aqui
    só precisamos da ordem de grandeza para dimensionar a amostra.
    """
    consulta = text("""
        SELECT GREATEST(reltuples::bigint, 1)
        FROM pg_class WHERE oid = 'feature_store.abt'::regclass
    """)
    with get_engine().connect() as conn:
        return int(conn.execute(consulta).scalar_one())


def carregar_populacao(features: list, amostra: int = 20000,
                       apenas_treino: bool = None, semente: int = 1) -> pd.DataFrame:
    """
    Lê da feature store uma amostra de clientes, só com as features monitoradas.

    Duas decisões que fazem esta consulta ser viável na base cheia:

    1. Extrai os valores de dentro do JSONB direto no SQL (`jsonb_num`), em vez
       de trazer as 836 features de cada cliente para o Python e descartar 800.

    2. Usa TABLESAMPLE, e não `ORDER BY random()`. Ordenar aleatoriamente exige
       ler e ordenar as 356 mil linhas inteiras — a primeira versão deste código
       fazia isso e levava minutos. O TABLESAMPLE sorteia BLOCOS de disco: lê só
       a fração necessária. `REPEATABLE` fixa a semente, então a mesma chamada
       devolve a mesma amostra e a demonstração é reprodutível.
    """
    # jsonb_to_record desserializa o JSONB UMA vez por linha e projeta as chaves
    # pedidas. A alternativa (`features->>'X'` repetido 30 vezes) reabre o
    # documento a cada chave — 30x mais trabalho por cliente.
    #
    # Todas as colunas saem como TEXT de propósito: a ABT tem "NaN"/"Infinity"
    # gravados, e um cast para numeric no banco quebraria a consulta inteira.
    # A conversão fica no pandas, com errors="coerce" (inválido vira NaN).
    definicao = ", ".join(f'"{f}" text' for f in features)

    filtro = ""
    if apenas_treino is True:
        filtro = "WHERE a.target IS NOT NULL"
    elif apenas_treino is False:
        filtro = "WHERE a.target IS NULL"

    # Margem de 3x: o TABLESAMPLE é aproximado e o filtro de target descarta
    # parte do que veio. Melhor sortear a mais e cortar com LIMIT do que voltar
    # ao banco por ter trazido pouco.
    percentual = min(100.0, max(0.5, (amostra / _total_estimado()) * 100 * 3))

    consulta = text(f"""
        SELECT x.*
        FROM feature_store.abt AS a
             TABLESAMPLE SYSTEM ({percentual}) REPEATABLE ({semente}),
             LATERAL jsonb_to_record(a.features) AS x({definicao})
        {filtro}
        LIMIT :limite
    """)
    with get_engine().connect() as conn:
        dados = pd.DataFrame(conn.execute(consulta, {"limite": amostra}).mappings())

    # Converte para número; o que não for numérico ("NaN", "Infinity") vira NaN,
    # que é exatamente como o PSI e o modelo tratam ausência.
    for coluna in dados.columns:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    return dados


def aplicar_choque(df: pd.DataFrame, intensidade: float = 1.0) -> pd.DataFrame:
    """
    Simula uma população que MUDOU — para demonstrar o monitoramento detectando.

    Não é maquiagem: é um cenário de negócio plausível e declarado. Reproduz o
    que uma recessão faria com a carteira — renda cai, scores externos pioram,
    o crédito pedido sobe (as pessoas tomam mais para fechar as contas) e os
    atrasos aumentam.

    Sem isso não haveria como demonstrar o monitoramento: comparar a base
    com ela mesma dá PSI zero, e um painel que só sabe dizer "está tudo bem"
    não prova que saberia avisar quando não estivesse.
    """
    alterado = df.copy()
    rng = np.random.default_rng(42)   # determinístico: a demo dá o mesmo número sempre
    n = len(alterado)

    def desloca(coluna: str, fator: float = None, soma: float = None, minimo=None, maximo=None):
        if coluna not in alterado.columns:
            return
        valores = pd.to_numeric(alterado[coluna], errors="coerce")
        ruido = rng.normal(1.0, 0.05 * intensidade, n)   # heterogeneidade entre clientes
        if fator is not None:
            valores = valores * (1 + (fator - 1) * intensidade) * ruido
        if soma is not None:
            valores = valores + soma * intensidade
        if minimo is not None or maximo is not None:
            valores = valores.clip(lower=minimo, upper=maximo)
        alterado[coluna] = valores

    def degrada(coluna: str, soma: float, ruido: float, minimo=None, maximo=None):
        """
        Desloca a variável E injeta ruído nela.

        A diferença importa: um deslocamento puro move todo mundo na mesma
        direção — piora a calibração, mas preserva a ORDENAÇÃO dos clientes, e
        portanto o AUC continua alto. O que realmente destrói a capacidade
        preditiva é o sinal ficar mais ruidoso.

        É o cenário realista para os EXT_SOURCE: o bureau externo muda a
        metodologia do score, e a variável mais forte do modelo passa a
        separar bons e maus pagadores pior do que separava.
        """
        if coluna not in alterado.columns:
            return
        valores = pd.to_numeric(alterado[coluna], errors="coerce")
        valores = valores + soma * intensidade
        valores = valores + rng.normal(0, ruido * intensidade, n)
        alterado[coluna] = valores.clip(lower=minimo, upper=maximo)

    desloca("AMT_INCOME_TOTAL", fator=0.70)                      # renda cai 30%
    degrada("EXT_SOURCE_1", soma=-0.15, ruido=0.22, minimo=0, maximo=1)   # bureau muda
    degrada("EXT_SOURCE_2", soma=-0.18, ruido=0.25, minimo=0, maximo=1)   # o score e perde
    degrada("EXT_SOURCE_3", soma=-0.12, ruido=0.20, minimo=0, maximo=1)   # poder de separacao
    desloca("AMT_CREDIT", fator=1.25)                            # pedem mais crédito
    desloca("PAYMENT_RATE", fator=1.20)                          # parcela pesa mais
    desloca("ANNUITY_INCOME_PERC", fator=1.35)                   # comprometimento sobe
    desloca("INSTAL_DPD_MEAN", fator=1.60)                       # atrasos aumentam
    return alterado


# ============================================================
# PERSISTÊNCIA DAS EXECUÇÕES
# ============================================================

def _json_seguro(valor):
    """
    Prepara um valor para virar JSON.

    NaN e Infinity sao validos em Python e no numpy, mas NAO em JSON — e o
    Postgres recusa o INSERT com "Token NaN is invalid". Como o PSI devolve NaN
    quando uma feature nao tem dado suficiente, o resumo precisa ser limpo
    antes de ser gravado.
    """
    if isinstance(valor, dict):
        return {k: _json_seguro(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_json_seguro(v) for v in valor]
    if isinstance(valor, (np.floating, float)):
        valor = float(valor)
        return None if (np.isnan(valor) or np.isinf(valor)) else valor
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.bool_,)):
        return bool(valor)
    return valor


def registrar_execucao(tipo: str, referencia: str, atual: str, resumo: dict,
                       n_avaliadas: int = None, n_severo: int = None,
                       n_moderado: int = None, retreino: bool = False,
                       model_version: str = None) -> int:
    """Grava o cabeçalho de uma execução de monitoramento e devolve o run_id."""
    inserir = text("""
        INSERT INTO mlops.monitoring_runs
            (run_type, reference_label, current_label, model_version,
             n_features_evaluated, n_severe, n_moderate, retrain_recommended, summary)
        VALUES
            (:tipo, :ref, :atual, :versao, :n, :sev, :mod, :retreino, :resumo)
        RETURNING run_id
    """)
    with get_engine().begin() as conn:
        return conn.execute(inserir, {
            "tipo": tipo, "ref": referencia, "atual": atual, "versao": model_version,
            "n": n_avaliadas, "sev": n_severo, "mod": n_moderado, "retreino": retreino,
            "resumo": json.dumps(_json_seguro(resumo), ensure_ascii=False, default=str),
        }).scalar_one()


def registrar_drift(run_id: int, drift: pd.DataFrame) -> None:
    """Grava o PSI de cada feature daquela execução."""
    linhas = [
        {"run_id": run_id, "feature": r["feature"],
         "psi": None if pd.isna(r["psi"]) else float(r["psi"]), "status": r["status"]}
        for _, r in drift.iterrows()
    ]
    if not linhas:
        return
    with get_engine().begin() as conn:
        conn.execute(text("""
            INSERT INTO mlops.drift_metrics (run_id, feature, psi, status)
            VALUES (:run_id, :feature, :psi, :status)
        """), linhas)


def registrar_performance(run_id: int, metricas: list) -> None:
    """Grava AUC/KS/Gini por safra."""
    if not metricas:
        return
    with get_engine().begin() as conn:
        conn.execute(text("""
            INSERT INTO mlops.performance_metrics
                (run_id, model_version, cohort, n_obs, auc, ks, gini, default_rate)
            VALUES (:run_id, :versao, :safra, :n, :auc, :ks, :gini, :taxa)
        """), [{"run_id": run_id, **m} for m in metricas])


# ============================================================
# 1. DATA DRIFT
# ============================================================

def rodar_data_drift(amostra: int = 20000, simular_choque: bool = False,
                     intensidade: float = 1.0) -> dict:
    """
    Compara a população atual contra a de referência (treino) e persiste.

    `retrain_recommended` = True quando há drift severo em qualquer feature —
    é o gatilho que a DAG do Airflow usa para disparar o re-treino.
    """
    features = features_monitoradas()
    referencia = carregar_populacao(features, amostra=amostra, apenas_treino=True, semente=1)

    if simular_choque:
        atual = aplicar_choque(
            carregar_populacao(features, amostra=amostra, apenas_treino=True, semente=2),
            intensidade=intensidade)
        rotulo_atual = f"populacao com choque simulado (intensidade {intensidade})"
    else:
        # Sem choque, a "produção" é a população que ainda não tem desfecho —
        # os clientes do conjunto de teste. É a comparação honesta disponível.
        atual = carregar_populacao(features, amostra=amostra, apenas_treino=False, semente=3)
        rotulo_atual = "clientes sem desfecho (proxy de producao)"

    drift = compute_data_drift(referencia, atual)
    n_severo   = int((drift["status"] == "SEVERO").sum())
    n_moderado = int((drift["status"] == "MODERADO").sum())

    resumo = {
        "n_referencia": len(referencia),
        "n_atual": len(atual),
        "top_drift": drift.head(10).to_dict(orient="records"),
    }
    run_id = registrar_execucao(
        "data_drift", f"treino (amostra {len(referencia)})", rotulo_atual, resumo,
        n_avaliadas=len(drift), n_severo=n_severo, n_moderado=n_moderado,
        retreino=n_severo > 0)
    registrar_drift(run_id, drift)

    return {"run_id": run_id, "n_features_avaliadas": len(drift),
            "drift_severo": n_severo, "drift_moderado": n_moderado,
            "retrain_recommended": n_severo > 0,
            "top_drift": drift.to_dict(orient="records")}


# ============================================================
# 2. PREDICTION DRIFT
# ============================================================

def rodar_prediction_drift(janela_dias: int = 30) -> dict:
    """
    Compara a distribuição das PDs servidas em produção contra a do treino.

    A referência é o out-of-fold: as PDs que o modelo produziu sobre clientes
    que não viu. É o comportamento esperado dele em população estável.

    Por que essa camada existe além do drift por feature: se o pipeline quebrar
    e uma feature importante chegar sempre nula, cada PSI individual pode ficar
    abaixo do limiar, mas a distribuição das decisões desloca inteira — e isso
    aqui pega.
    """
    if not os.path.exists(OOF_PREDICTIONS_PATH):
        return {"erro": "oof_predictions.csv nao encontrado — rode: python -m Model.train"}

    referencia = pd.read_csv(OOF_PREDICTIONS_PATH)["PD"].to_numpy(dtype="float64")

    consulta = text("""
        SELECT probability_default::float8 AS pd, decision
        FROM serving.predictions
        WHERE created_at >= now() - make_interval(days => :dias)
    """)
    with get_engine().connect() as conn:
        producao = pd.DataFrame(conn.execute(consulta, {"dias": janela_dias}).mappings())

    if producao.empty:
        return {"erro": f"nenhuma predicao registrada nos ultimos {janela_dias} dias. "
                        f"Gere trafego com: python -m MLOps.simulate_production"}

    atual = producao["pd"].to_numpy(dtype="float64")
    psi = _psi_single(referencia, atual)
    status = _classificar(psi)

    # A taxa média de default prevista é o número que o negócio acompanha.
    mix = producao["decision"].value_counts(normalize=True).round(4).to_dict()
    resumo = {
        "psi_pd": psi,
        "status": status,
        "pd_media_treino": float(np.mean(referencia)),
        "pd_media_producao": float(np.mean(atual)),
        "n_predicoes": int(len(atual)),
        "mix_de_decisoes": mix,
    }

    run_id = registrar_execucao(
        "prediction_drift", "PD out-of-fold (treino)",
        f"predicoes dos ultimos {janela_dias} dias", resumo,
        n_avaliadas=1, n_severo=int(status == "SEVERO"), n_moderado=int(status == "MODERADO"),
        retreino=status == "SEVERO")
    registrar_drift(run_id, pd.DataFrame([{"feature": "probability_default",
                                           "psi": psi, "status": status}]))

    return {"run_id": run_id, **resumo}


# ============================================================
# 3. PERFORMANCE POR SAFRA
# ============================================================

def rodar_performance() -> dict:
    """
    Recalcula AUC/KS/Gini sobre as decisões já tomadas, por safra.

    Só é possível para clientes cujo desfecho já é conhecido. No mundo real
    esse desfecho chega meses depois da decisão (label lag); aqui usamos o
    TARGET da feature store como proxy do desfecho que "amadureceu".

    Uma safra com AUC muito abaixo do out-of-fold do treino (0,79) é o sinal
    mais forte que existe de que o modelo precisa ser re-treinado.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    consulta = text("""
        SELECT to_char(p.created_at, 'YYYY-MM')     AS safra,
               p.model_version                      AS versao,
               p.probability_default::float8        AS pd,
               a.target                             AS y
        FROM serving.predictions p
        JOIN feature_store.abt a ON a.sk_id_curr = p.sk_id_curr
        WHERE a.target IS NOT NULL
    """)
    with get_engine().connect() as conn:
        dados = pd.DataFrame(conn.execute(consulta).mappings())

    if dados.empty:
        return {"erro": "nenhuma predicao com desfecho conhecido. "
                        "Gere trafego com: python -m MLOps.simulate_production"}

    metricas = []
    for (safra, versao), grupo in dados.groupby(["safra", "versao"], dropna=False):
        y = grupo["y"].to_numpy()
        p = grupo["pd"].to_numpy()
        # AUC exige as duas classes presentes: uma safra só de bons pagadores
        # não tem o que discriminar, e o cálculo iria estourar.
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        auc = float(roc_auc_score(y, p))
        metricas.append({
            "versao": versao, "safra": safra, "n": int(len(y)),
            "auc": round(auc, 6),
            "ks": round(float(np.max(tpr - fpr)), 6),
            "gini": round(2 * auc - 1, 6),
            "taxa": round(float(np.mean(y)), 6),
        })

    if not metricas:
        return {"erro": "as safras encontradas tem uma unica classe — sem como medir AUC."}

    metricas.sort(key=lambda m: m["safra"])
    auc_medio = float(np.mean([m["auc"] for m in metricas]))

    # O veredito olha a safra MAIS RECENTE, não a média da série.
    # A média dilui: um trimestre bom carrega meses ruins e o alerta nunca
    # dispara — justamente quando o modelo já está errando em produção.
    # Ignora safras pequenas demais, onde o AUC é ruído.
    maduras = [m for m in metricas if m["n"] >= TAMANHO_MINIMO_SAFRA]
    recente = maduras[-1] if maduras else metricas[-1]
    limite = AUC_REFERENCIA * (1 - QUEDA_TOLERADA)
    degradou = recente["auc"] < limite

    resumo = {"safras": metricas, "auc_medio": round(auc_medio, 6),
              "auc_referencia_treino": AUC_REFERENCIA,
              "safra_avaliada": recente["safra"],
              "auc_safra_recente": recente["auc"],
              "limite_de_alerta": round(limite, 6)}

    run_id = registrar_execucao(
        "performance", "AUC out-of-fold do treino (0,7909)",
        f"{len(metricas)} safra(s) com desfecho conhecido", resumo,
        n_avaliadas=len(metricas), retreino=degradou,
        model_version=metricas[0]["versao"])
    registrar_performance(run_id, metricas)

    return {"run_id": run_id, "auc_medio": round(auc_medio, 6),
            "degradou": degradou, "safra_avaliada": recente["safra"],
            "auc_safra_recente": recente["auc"], "limite_de_alerta": round(limite, 6),
            "safras": metricas}


# ============================================================
# CONSULTA DO HISTÓRICO (usado pela API e pelo painel)
# ============================================================

def historico(limite: int = 20, tipo: str = None) -> list:
    """Últimas execuções de monitoramento — a série histórica que faltava."""
    filtro = "WHERE run_type = :tipo" if tipo else ""
    consulta = text(f"""
        SELECT run_id, run_type, reference_label, current_label, n_features_evaluated,
               n_severe, n_moderate, retrain_recommended, summary, created_at
        FROM mlops.monitoring_runs
        {filtro}
        ORDER BY created_at DESC
        LIMIT :limite
    """)
    parametros = {"limite": limite}
    if tipo:
        parametros["tipo"] = tipo
    with get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(consulta, parametros).mappings()]


def drift_da_execucao(run_id: int) -> list:
    """PSI por feature de uma execução específica."""
    consulta = text("""
        SELECT feature, psi, status FROM mlops.drift_metrics
        WHERE run_id = :run_id ORDER BY psi DESC NULLS LAST
    """)
    with get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(consulta, {"run_id": run_id}).mappings()]


def serie_performance(limite: int = 50) -> list:
    """Histórico de AUC/KS por safra, para o gráfico de acompanhamento."""
    consulta = text("""
        SELECT cohort AS safra, model_version AS versao, n_obs, auc, ks, gini,
               default_rate, created_at
        FROM mlops.performance_metrics
        ORDER BY cohort DESC, created_at DESC
        LIMIT :limite
    """)
    with get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(consulta, {"limite": limite}).mappings()]


# ============================================================
# CLI
# ============================================================

def _imprimir_drift(resultado: dict) -> None:
    if "erro" in resultado:
        print(f"  [!] {resultado['erro']}")
        return
    print(f"  run_id: {resultado['run_id']}")
    print(f"  features avaliadas: {resultado['n_features_avaliadas']}")
    print(f"  drift SEVERO: {resultado['drift_severo']} | MODERADO: {resultado['drift_moderado']}")
    print(f"  re-treino recomendado: {'SIM' if resultado['retrain_recommended'] else 'nao'}")
    print("\n  Top features por drift (PSI):")
    tabela = pd.DataFrame(resultado["top_drift"]).head(12)
    print("  " + tabela.to_string(index=False).replace("\n", "\n  "))


def _run_cli() -> int:
    parser = argparse.ArgumentParser(description="Monitoramento de dados e do modelo")
    parser.add_argument("--data-drift",       action="store_true", help="PSI das features")
    parser.add_argument("--prediction-drift", action="store_true", help="PSI da distribuicao de PD")
    parser.add_argument("--performance",      action="store_true", help="AUC/KS/Gini por safra")
    parser.add_argument("--tudo",             action="store_true", help="roda as tres camadas")
    parser.add_argument("--simular-choque",   action="store_true",
                        help="aplica um cenario de recessao na populacao atual (demonstracao)")
    parser.add_argument("--intensidade",      type=float, default=1.0,
                        help="intensidade do choque simulado (padrao: 1.0)")
    parser.add_argument("--amostra",          type=int, default=20000,
                        help="clientes por populacao comparada (padrao: 20000)")
    parser.add_argument("--janela-dias",      type=int, default=30,
                        help="janela do prediction drift (padrao: 30)")
    parser.add_argument("--historico",        action="store_true",
                        help="mostra as ultimas execucoes registradas")
    args = parser.parse_args()

    if args.historico:
        print("\nULTIMAS EXECUCOES DE MONITORAMENTO\n" + "=" * 78)
        for r in historico(15):
            alerta = "RE-TREINO" if r["retrain_recommended"] else "ok"
            print(f"  #{r['run_id']:<4} {r['created_at']:%Y-%m-%d %H:%M}  "
                  f"{r['run_type']:<17} severo={r['n_severe'] or 0:<3} "
                  f"moderado={r['n_moderate'] or 0:<3} -> {alerta}")
        return 0

    if not (args.data_drift or args.prediction_drift or args.performance or args.tudo):
        parser.print_help()
        return 0

    if args.data_drift or args.tudo:
        print("\n" + "=" * 78)
        print("1. DATA DRIFT (PSI por feature)")
        if args.simular_choque:
            print("   [cenario simulado: recessao — renda cai, scores pioram, atrasos sobem]")
        print("=" * 78)
        _imprimir_drift(rodar_data_drift(amostra=args.amostra,
                                         simular_choque=args.simular_choque,
                                         intensidade=args.intensidade))

    if args.prediction_drift or args.tudo:
        print("\n" + "=" * 78)
        print("2. PREDICTION DRIFT (PSI da distribuicao de PD)")
        print("=" * 78)
        resultado = rodar_prediction_drift(janela_dias=args.janela_dias)
        if "erro" in resultado:
            print(f"  [!] {resultado['erro']}")
        else:
            print(f"  run_id: {resultado['run_id']} | predicoes analisadas: {resultado['n_predicoes']:,}")
            print(f"  PD media no treino   : {resultado['pd_media_treino']:.4f}")
            print(f"  PD media em producao : {resultado['pd_media_producao']:.4f}")
            print(f"  PSI: {resultado['psi_pd']:.4f} -> {resultado['status']}")
            print(f"  mix de decisoes: {resultado['mix_de_decisoes']}")

    if args.performance or args.tudo:
        print("\n" + "=" * 78)
        print("3. PERFORMANCE POR SAFRA (AUC/KS/Gini)")
        print("=" * 78)
        resultado = rodar_performance()
        if "erro" in resultado:
            print(f"  [!] {resultado['erro']}")
        else:
            print(f"  run_id: {resultado['run_id']} | AUC medio da serie: {resultado['auc_medio']:.4f}")
            print(f"  safra avaliada: {resultado['safra_avaliada']} -> "
                  f"AUC {resultado['auc_safra_recente']:.4f} "
                  f"(limite de alerta: {resultado['limite_de_alerta']:.4f})")
            print(f"  DEGRADACAO DETECTADA: "
                  f"{'SIM - recomendar re-treino' if resultado['degradou'] else 'nao'}\n")
            print("  " + pd.DataFrame(resultado["safras"]).to_string(index=False).replace("\n", "\n  "))

    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
