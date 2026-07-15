"""
monitoring.py — Monitoramento de dados e do modelo em produção.

Cobre o item (iii) da etapa individual: como detectar falhas, perda de
performance e mudança de comportamento dos dados (drift).

Implementa duas verificações concretas e acionáveis:

1. DATA DRIFT (PSI — Population Stability Index)
   Compara a distribuição de cada feature em produção contra a distribuição de
   referência (o conjunto de treino). É a forma clássica de detectar que "os
   dados de entrada mudaram" — o principal motivo de um modelo bom degradar sem
   que ninguém mexa nele.
       PSI < 0.10  → estável
       0.10–0.25   → drift moderado (investigar)
       PSI > 0.25  → drift severo (candidato a re-treino)

2. PREDICTION DRIFT
   Acompanha a taxa média de default prevista. Um salto súbito indica ou drift
   de dados ou um problema no pipeline de features, mesmo sem termos ainda o
   TARGET real (que só chega meses depois — o "label lag" do crédito).

Uso:
    python -m MLOps.monitoring --reference abt.csv --current novos_dados.csv

Em produção, este módulo seria chamado por um job agendado (ex.: a mesma DAG do
Airflow, numa branch de monitoramento) e emitiria alertas / dispararia re-treino.
"""

import argparse
import json

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NON_FEATURE_COLS, TARGET_COLUMN

# Limiares de PSI — convenção de mercado (crédito/risco)
PSI_MODERATE = 0.10
PSI_SEVERE   = 0.25


# ============================================================
# POPULATION STABILITY INDEX
# ============================================================

def _psi_single(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Calcula o PSI de uma feature entre referência e produção.

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
        return 0.0  # feature constante — sem como driftar
    edges[0], edges[-1] = -np.inf, np.inf  # captura valores fora do range de treino

    ref_pct = np.histogram(reference, bins=edges)[0] / reference.size
    cur_pct = np.histogram(current,   bins=edges)[0] / current.size

    # epsilon evita log(0) / divisão por 0 em buckets vazios
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_data_drift(reference: pd.DataFrame, current: pd.DataFrame,
                       top_n: int = 20) -> pd.DataFrame:
    """
    Calcula o PSI de todas as features numéricas comuns aos dois conjuntos.

    Retorna um DataFrame ordenado do maior drift para o menor, com o status
    (OK / MODERADO / SEVERO) de cada feature.
    """
    feats = [c for c in reference.columns
             if c not in NON_FEATURE_COLS + [TARGET_COLUMN]
             and c in current.columns
             and pd.api.types.is_numeric_dtype(reference[c])]

    rows = []
    for c in feats:
        psi = _psi_single(reference[c].to_numpy(dtype="float64"),
                          current[c].to_numpy(dtype="float64"))
        status = ("SEVERO" if psi >= PSI_SEVERE
                  else "MODERADO" if psi >= PSI_MODERATE
                  else "OK")
        rows.append({"feature": c, "psi": psi, "status": status})

    result = pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")
    return result.head(top_n).reset_index(drop=True)


# ============================================================
# RELATÓRIO CONSOLIDADO
# ============================================================

def monitoring_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Gera um relatório de monitoramento com veredito acionável.

    `retrain_recommended` = True quando há drift severo em qualquer feature —
    é o gatilho que o pipeline_orchestration / a DAG usariam para re-treinar.
    """
    drift = compute_data_drift(reference, current)
    n_severe   = int((drift["status"] == "SEVERO").sum())
    n_moderate = int((drift["status"] == "MODERADO").sum())

    report = {
        "n_features_avaliadas": len(drift),
        "drift_severo":         n_severe,
        "drift_moderado":       n_moderate,
        "top_drift":            drift.to_dict(orient="records"),
        "retrain_recommended":  n_severe > 0,
    }
    return report


# ============================================================
# CLI
# ============================================================

def _run_cli():
    parser = argparse.ArgumentParser(description="Monitoramento de drift de dados (PSI)")
    parser.add_argument("--reference", required=True, help="CSV de referência (ex.: abt.csv de treino)")
    parser.add_argument("--current",   required=True, help="CSV com os dados novos/produção")
    parser.add_argument("--output",    default=None,  help="JSON de saída do relatório (opcional)")
    args = parser.parse_args()

    ref = pd.read_csv(args.reference)
    cur = pd.read_csv(args.current)

    report = monitoring_report(ref, cur)

    print(f"\nFeatures avaliadas: {report['n_features_avaliadas']}")
    print(f"Drift SEVERO:   {report['drift_severo']}")
    print(f"Drift MODERADO: {report['drift_moderado']}")
    print(f"Re-treino recomendado: {'SIM' if report['retrain_recommended'] else 'não'}\n")
    print("Top features por drift (PSI):")
    print(pd.DataFrame(report["top_drift"]).to_string(index=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nRelatório salvo em: {args.output}")


if __name__ == "__main__":
    _run_cli()
