"""
batch_scoring.py — Pontuação da carteira em lote.

O serviço de predição atende UM cliente por vez, sob demanda (a API). Mas boa
parte do uso real de um modelo de crédito é em lote: todo dia é preciso
repontuar a carteira, ou pontuar a fila de propostas que entrou desde ontem.

É o segundo caminho de consumo do mesmo modelo:

    API        → decisão individual, síncrona, no momento da proposta
    batch      → decisão em massa, agendada, sobre a carteira inteira

Os dois leem as features da mesma `feature_store.abt` e gravam em
`serving.predictions` — o que muda é só o `source`, para o monitoramento
conseguir separar depois de onde veio cada decisão.

Uso:
    python -m MLOps.batch_scoring --limite 5000
    python -m MLOps.batch_scoring --limite 5000 --somente-novos
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text

from config import DB_CHUNK_SIZE
from Model.predict import predict, model_metadata
from MLOps.db import get_engine
from MLOps import store

ORIGEM = "batch"


def _clientes(limite: int, somente_novos: bool) -> pd.DataFrame:
    """
    Seleciona os clientes a pontuar.

    `somente_novos` pega quem ainda não tem desfecho conhecido — que é a
    situação real de uma fila de propostas: o cliente pediu crédito e ninguém
    sabe ainda se ele vai pagar.
    """
    filtro = "WHERE target IS NULL" if somente_novos else ""
    consulta = text(f"""
        SELECT sk_id_curr, features
        FROM feature_store.abt
        {filtro}
        ORDER BY sk_id_curr
        LIMIT :limite
    """)
    with get_engine().connect() as conn:
        return pd.DataFrame(conn.execute(consulta, {"limite": limite}).mappings())


def _gravar(resultados: list, versao: str, latencia_media: float) -> int:
    """
    Grava as decisões do lote numa única transação.

    Um INSERT por cliente (como faz a API) seria lento demais aqui: são
    milhares de linhas por execução, e cada ida ao banco custa mais que a
    própria predição.
    """
    if not resultados:
        return 0
    inserir = text("""
        INSERT INTO serving.predictions
            (sk_id_curr, model_version, probability_default, risk_band, decision,
             source, n_features_expected, n_features_missing, request_payload, latency_ms)
        VALUES
            (:sk_id, :versao, :pd, :faixa, :decisao, :origem, :esperadas,
             :ausentes, :payload, :latencia)
    """)
    linhas = [{
        "sk_id":     r["SK_ID_CURR"],
        "versao":    versao,
        "pd":        r["probability_default"],
        "faixa":     r["risk_band"],
        "decisao":   r["decision"],
        "origem":    ORIGEM,
        "esperadas": r["n_features_expected"],
        "ausentes":  r["n_features_missing"],
        "payload":   json.dumps({"origem": "batch_scoring"}),
        "latencia":  latencia_media,
    } for r in resultados]

    with get_engine().begin() as conn:
        conn.execute(inserir, linhas)
    return len(linhas)


def run(limite: int = 5000, somente_novos: bool = True) -> dict:
    """Pontua a carteira e devolve o resumo — usado pela CLI e pela DAG."""
    versao = store.ensure_model_registered(model_metadata())

    clientes = _clientes(limite, somente_novos)
    if clientes.empty:
        print("Nenhum cliente a pontuar. A feature store esta carregada?")
        return {"pontuados": 0}

    print(f"Pontuando {len(clientes):,} clientes (modelo {versao})")
    inicio = time.time()
    total, mix = 0, {}

    # Em blocos: o modelo aceita um lote inteiro de uma vez, mas montar um
    # DataFrame com 5 mil linhas x 836 colunas de uma só vez pesa na memória.
    for comeco in range(0, len(clientes), DB_CHUNK_SIZE):
        bloco = clientes.iloc[comeco:comeco + DB_CHUNK_SIZE]

        registros = []
        for _, linha in bloco.iterrows():
            registro = dict(linha["features"])
            registro["SK_ID_CURR"] = int(linha["sk_id_curr"])
            registros.append(registro)

        t0 = time.time()
        resultados = predict(registros)
        latencia_media = (time.time() - t0) * 1000 / max(len(resultados), 1)

        total += _gravar(resultados, versao, latencia_media)
        for r in resultados:
            mix[r["decision"]] = mix.get(r["decision"], 0) + 1
        print(f"    {total:,} pontuados...", end="\r")

    duracao = time.time() - inicio
    print(f"\n  OK: {total:,} clientes em {duracao:.0f}s ({total / max(duracao, 1):.0f}/s)")
    print("  mix de decisoes:")
    for decisao, n in sorted(mix.items(), key=lambda kv: -kv[1]):
        print(f"    {decisao:<16} {n:>7,}  ({n / total:.1%})")

    return {"pontuados": total, "mix": mix, "model_version": versao,
            "duracao_s": round(duracao, 1)}


def _run_cli() -> int:
    parser = argparse.ArgumentParser(description="Pontua a carteira em lote")
    parser.add_argument("--limite", type=int, default=5000,
                        help="quantos clientes pontuar (padrao: 5000)")
    parser.add_argument("--somente-novos", action="store_true",
                        help="so clientes sem desfecho conhecido (fila de propostas)")
    args = parser.parse_args()
    run(limite=args.limite, somente_novos=args.somente_novos)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
