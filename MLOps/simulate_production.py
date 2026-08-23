"""
simulate_production.py — Gera tráfego de produção para o monitoramento observar.

O monitoramento precisa de duas coisas que um projeto recém-implantado ainda
não tem: **decisões tomadas ao longo do tempo** e **uma população que mudou**.
Este script produz as duas, de forma declarada e reprodutível.

Não é maquiar resultado — é montar o cenário. Sem histórico de decisões, o
prediction drift não tem o que comparar e a performance por safra não tem o que
calcular; o painel só saberia dizer "sem dados". Um monitoramento que nunca foi
visto detectando nada não prova que detectaria.

O que o script faz:

    1. sorteia clientes reais da feature store;
    2. distribui as decisões ao longo de N dias (safras);
    3. nos dias mais recentes, aplica um CHOQUE crescente na população —
       o cenário de recessão descrito em monitoring.aplicar_choque();
    4. pontua cada cliente com o modelo real e grava em serving.predictions.

Assim o histórico tem um "antes" estável e um "depois" degradado, e as três
camadas de monitoramento têm o que mostrar.

Uso:
    python -m MLOps.simulate_production                     # 90 dias, 40/dia
    python -m MLOps.simulate_production --dias 120 --por-dia 60
    python -m MLOps.simulate_production --sem-choque        # cenario estavel
    python -m MLOps.simulate_production --limpar            # apaga o simulado
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import (
    OOF_PREDICTIONS_PATH, ID_COLUMN, RISK_BANDS,
    DECISION_APPROVE_BELOW, DECISION_REJECT_ABOVE,
)
from Model.predict import model_metadata
from MLOps.db import get_engine
from MLOps import store

# A partir de que fração do período o choque começa a agir. Antes disso a
# população é estável — é o "antes" contra o qual o drift vai aparecer.
INICIO_DO_CHOQUE = 0.6

ORIGEM = "batch"   # tráfego gerado por lote, não por chamada de API


_OOF_CACHE = None
_N_FEATURES = None


def _n_features_modelo() -> int:
    """Quantas features o modelo espera — lido do metadado, nao fixado no codigo."""
    global _N_FEATURES
    if _N_FEATURES is None:
        _N_FEATURES = int(model_metadata()["n_features"])
    return _N_FEATURES


def _oof_por_cliente() -> dict:
    """
    Mapa {SK_ID_CURR: PD out-of-fold}, lido uma vez.

    A PD out-of-fold de um cliente foi produzida por um modelo treinado sem
    ele — é o que o modelo entregaria de verdade sobre alguém novo.
    """
    global _OOF_CACHE
    if _OOF_CACHE is None:
        if not os.path.exists(OOF_PREDICTIONS_PATH):
            _OOF_CACHE = {}
        else:
            oof = pd.read_csv(OOF_PREDICTIONS_PATH)
            _OOF_CACHE = dict(zip(oof[ID_COLUMN].astype(int), oof["PD"].astype(float)))
    return _OOF_CACHE


def _faixa(pd_valor: float) -> str:
    """Faixa de risco — mesma tabela do config.py usada pelo predict."""
    for rotulo, teto in RISK_BANDS:
        if pd_valor < teto:
            return rotulo
    return RISK_BANDS[-1][0]


def _decisao(pd_valor: float) -> str:
    """Política de crédito do config.py (aprova <8%, recusa >30%)."""
    if pd_valor < DECISION_APPROVE_BELOW:
        return "APROVAR"
    if pd_valor > DECISION_REJECT_ABOVE:
        return "RECUSAR"
    return "ANALISE_MANUAL"


# Parâmetros do cenário de degradação, aplicados em escala logit sobre a PD.
# DESLOCAMENTO: o risco médio da carteira sobe (a recessão atinge todo mundo).
# RUÍDO: o sinal perde poder de SEPARAÇÃO — é isto, e não o deslocamento, que
# derruba o AUC. Um deslocamento uniforme piora a calibração mas mantém a
# ordenação dos clientes intacta.
CHOQUE_DESLOCAMENTO = 0.55
CHOQUE_RUIDO        = 2.20   # calibrado: leva o AUC de 0,79 para ~0,65 em intensidade plena


def _pontuar(fatia: pd.DataFrame, intensidade: float, rng) -> list:
    """
    Produz a PD "observada em produção" de cada cliente da fatia.

    HONESTIDADE DA MÉTRICA — o ponto mais delicado desta simulação, e onde a
    primeira versão deste código errou.

    Os clientes precisam ter desfecho conhecido (senão não há AUC para
    calcular), e só o conjunto de treino tem. Mas o modelo final VIU esses
    clientes no treino: repontuá-los com ele devolve uma PD otimista, e a
    performance por safra sai em ~0,84 em vez dos 0,79 reais.

    Tentar contornar isso usando OOF no período estável e repontuação no
    período com choque foi pior ainda: misturava duas réguas diferentes, e as
    safras "degradadas" apareciam com AUC MAIOR que as estáveis — o vazamento
    superava a degradação, e a comparação entre safras perdia sentido.

    A solução é usar UMA régua só: a predição out-of-fold, feita por um modelo
    que não viu aquele cliente. A degradação é então aplicada sobre essa PD, em
    escala logit, de forma explícita e declarada. Assim toda a série usa a mesma
    base honesta, e a queda de AUC que o monitoramento detecta vem da
    degradação simulada — não de vazamento.
    """
    oof = _oof_por_cliente()
    resultados = []

    for _, linha in fatia.iterrows():
        sk_id = int(linha["sk_id_curr"])
        pd_base = oof.get(sk_id)
        if pd_base is None:
            continue

        if intensidade > 0:
            # logit → aplica deslocamento e ruído → volta para probabilidade.
            # Trabalhar em logit mantém o resultado sempre em (0, 1) sem
            # precisar truncar, e é a escala em que o modelo de fato opera.
            p = float(np.clip(pd_base, 1e-6, 1 - 1e-6))
            logito = np.log(p / (1 - p))
            logito += CHOQUE_DESLOCAMENTO * intensidade
            logito += rng.normal(0, CHOQUE_RUIDO * intensidade)
            pd_obs = float(1 / (1 + np.exp(-logito)))
        else:
            pd_obs = float(pd_base)

        # n_features do cliente: o JSONB só guarda o que não é ausente
        n_presentes = len(linha["features"])
        resultados.append({
            "SK_ID_CURR": sk_id,
            "probability_default": round(pd_obs, 6),
            "risk_band": _faixa(pd_obs),
            "decision": _decisao(pd_obs),
            "n_features_expected": _n_features_modelo(),
            "n_features_missing": max(_n_features_modelo() - n_presentes, 0),
        })
    return resultados


def _clientes_sorteados(n: int, semente: int = 7) -> pd.DataFrame:
    """
    Sorteia clientes COM desfecho conhecido.

    Só esses servem: a performance por safra precisa comparar a PD prevista com
    o que de fato aconteceu. Sem TARGET não há AUC para calcular.
    """
    consulta = text("""
        SELECT sk_id_curr, target, features
        FROM feature_store.abt
        WHERE target IS NOT NULL
        ORDER BY md5(sk_id_curr::text || :semente)
        LIMIT :limite
    """)
    with get_engine().connect() as conn:
        return pd.DataFrame(
            conn.execute(consulta, {"limite": n, "semente": str(semente)}).mappings()
        )


def _gravar_lote(linhas: list) -> int:
    """Grava as decisões simuladas com a data de cada safra."""
    if not linhas:
        return 0
    inserir = text("""
        INSERT INTO serving.predictions
            (sk_id_curr, model_version, probability_default, risk_band, decision,
             source, n_features_expected, n_features_missing, request_payload,
             latency_ms, created_at)
        VALUES
            (:sk_id, :versao, :pd, :faixa, :decisao, :origem, :esperadas,
             :ausentes, :payload, :latencia, :quando)
    """)
    with get_engine().begin() as conn:
        conn.execute(inserir, linhas)
    return len(linhas)


def simular(dias: int = 90, por_dia: int = 40, com_choque: bool = True,
            intensidade_final: float = 1.0) -> dict:
    """Gera o histórico de decisões e devolve um resumo do que foi criado."""
    total = dias * por_dia
    print(f"Simulando {total:,} decisoes ao longo de {dias} dias ({por_dia}/dia)")
    print(f"  cenario: {'choque crescente na segunda metade' if com_choque else 'populacao estavel'}")

    metadados = model_metadata()
    versao = store.ensure_model_registered(metadados)

    clientes = _clientes_sorteados(total)
    if clientes.empty:
        raise RuntimeError("Nenhum cliente com desfecho na feature store. "
                           "Rode antes: python -m MLOps.load_to_db --abt --full")
    print(f"  clientes sorteados: {len(clientes):,}")

    agora = datetime.now(timezone.utc)
    rng = np.random.default_rng(11)
    gravadas, por_safra = 0, {}
    posicao = 0

    for dia in range(dias):
        # Do mais antigo para o mais recente
        data_base = agora - timedelta(days=(dias - 1 - dia))
        fatia = clientes.iloc[posicao:posicao + por_dia]
        posicao += por_dia
        if fatia.empty:
            break

        # Intensidade do choque: zero até 60% do período, crescendo depois.
        progresso = dia / max(dias - 1, 1)
        if com_choque and progresso >= INICIO_DO_CHOQUE:
            intensidade = intensidade_final * (progresso - INICIO_DO_CHOQUE) / (1 - INICIO_DO_CHOQUE)
        else:
            intensidade = 0.0

        resultados = _pontuar(fatia, intensidade, rng)

        linhas = []
        for resultado in resultados:
            # Espalha as decisões dentro do horário comercial do dia
            quando = data_base.replace(
                hour=int(rng.integers(8, 19)), minute=int(rng.integers(0, 60)),
                second=int(rng.integers(0, 60)), microsecond=0)
            linhas.append({
                "sk_id": resultado["SK_ID_CURR"], "versao": versao,
                "pd": resultado["probability_default"], "faixa": resultado["risk_band"],
                "decisao": resultado["decision"], "origem": ORIGEM,
                "esperadas": resultado["n_features_expected"],
                "ausentes": resultado["n_features_missing"],
                "payload": json.dumps({"simulado": True,
                                       "intensidade_choque": round(intensidade, 3)}),
                # rng.normal() sem `size` devolve float puro, que não tem .clip()
                "latencia": float(np.clip(rng.normal(85, 18), 20, 400)),
                "quando": quando,
            })

        gravadas += _gravar_lote(linhas)
        safra = data_base.strftime("%Y-%m")
        por_safra[safra] = por_safra.get(safra, 0) + len(linhas)
        print(f"    dia {dia + 1:>3}/{dias}  {data_base:%Y-%m-%d}  "
              f"choque={intensidade:4.2f}  {gravadas:,} decisoes", end="\r")

    print(f"\n  OK: {gravadas:,} decisoes gravadas em serving.predictions")
    print(f"  safras geradas: {', '.join(f'{k} ({v:,})' for k, v in sorted(por_safra.items()))}")
    return {"gravadas": gravadas, "por_safra": por_safra, "model_version": versao}


def limpar() -> int:
    """Apaga apenas as decisões simuladas, preservando as reais da API."""
    with get_engine().begin() as conn:
        n = conn.execute(text("""
            DELETE FROM serving.predictions
            WHERE request_payload ->> 'simulado' = 'true'
        """)).rowcount
    print(f"{n:,} decisoes simuladas removidas")
    return n


def _run_cli() -> int:
    parser = argparse.ArgumentParser(description="Gera trafego de producao para o monitoramento")
    parser.add_argument("--dias",    type=int, default=90, help="quantos dias de historico")
    parser.add_argument("--por-dia", type=int, default=40, help="decisoes por dia")
    parser.add_argument("--sem-choque", action="store_true",
                        help="gera populacao estavel (sem cenario de recessao)")
    parser.add_argument("--intensidade", type=float, default=1.0,
                        help="intensidade final do choque (padrao: 1.0)")
    parser.add_argument("--limpar", action="store_true",
                        help="apaga as decisoes simuladas e sai")
    args = parser.parse_args()

    if args.limpar:
        limpar()
        return 0

    simular(dias=args.dias, por_dia=args.por_dia, com_choque=not args.sem_choque,
            intensidade_final=args.intensidade)
    print("\nAgora rode o monitoramento:")
    print("  python -m MLOps.monitoring --tudo")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
