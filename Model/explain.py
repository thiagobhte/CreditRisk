"""
explain.py — Explicabilidade do modelo com SHAP.

Por que SHAP e não só a importância de features do LightGBM:
    A importância nativa diz QUAIS variáveis o modelo mais usa no geral. Ela não
    responde a pergunta que o negócio (e o regulador) faz: "por que ESTE cliente
    foi recusado?". O SHAP decompõe cada predição individual, mostrando quanto
    cada feature empurrou a probabilidade para cima ou para baixo.

    Em crédito isso não é luxo: negar crédito sem saber justificar é um problema
    de conformidade, não só de modelagem.

Usa TreeExplainer, que é exato e rápido para modelos de árvore (LightGBM) —
não é a aproximação por amostragem usada em modelos genéricos.
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from Model.predict import _load_artifacts, _align


# Carregado sob demanda: construir o explainer custa tempo e a API/CLI que só
# fazem predição não precisam dele.
_EXPLAINER = None


def _get_explainer():
    """Constrói (uma vez) o TreeExplainer sobre o modelo persistido."""
    global _EXPLAINER
    if _EXPLAINER is None:
        import shap
        model, _, _ = _load_artifacts()
        _EXPLAINER = shap.TreeExplainer(model)
    return _EXPLAINER


def _shap_matrix(explanation) -> np.ndarray:
    """
    Normaliza a saída do SHAP para uma matriz (n_amostras, n_features).

    Dependendo da versão do SHAP e do tipo de modelo, o objeto devolvido pode
    trazer os valores como (n, features) — caso binário já reduzido — ou como
    (n, features, 2), com uma fatia por classe. Aqui ficamos sempre com a
    contribuição para a classe positiva (inadimplência).
    """
    values = explanation.values if hasattr(explanation, "values") else explanation
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]   # classe 1 = default
    return values


def explain_client(record: dict, top_n: int = 12) -> dict:
    """
    Explica a predição de UM cliente.

    Returns:
        {
          "base_value":  probabilidade média do modelo (ponto de partida),
          "contributions": [
              {"feature": ..., "value": ..., "shap": ..., "direction": "risco"|"protecao"},
              ...  # as top_n features de maior impacto absoluto
          ]
        }

    `shap` > 0 empurra o cliente PARA a inadimplência (aumenta o risco);
    `shap` < 0 puxa para a adimplência (protege).
    """
    _, feats, _ = _load_artifacts()
    explainer = _get_explainer()

    X = _align(pd.DataFrame([record]), feats)
    explanation = explainer(X)
    shap_values = _shap_matrix(explanation)[0]

    base = explanation.base_values
    base = np.asarray(base).ravel()
    base_value = float(base[-1]) if base.size else 0.0

    # Ordena pelo impacto ABSOLUTO: uma feature que protege fortemente é tão
    # relevante para a explicação quanto uma que agrava o risco.
    order = np.argsort(np.abs(shap_values))[::-1][:top_n]

    contributions = []
    for i in order:
        raw = X.iloc[0, i]
        contributions.append({
            "feature":   feats[i],
            "value":     None if pd.isna(raw) else float(raw),
            "shap":      float(shap_values[i]),
            "direction": "risco" if shap_values[i] > 0 else "protecao",
        })

    return {"base_value": base_value, "contributions": contributions}


def global_importance(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Importância global via SHAP: média do |valor SHAP| de cada feature sobre uma
    amostra de clientes.

    Diferente da importância nativa do LightGBM (que conta splits), esta mede o
    impacto médio real na probabilidade prevista — comparável entre features.

    Retorna DataFrame com colunas: feature, importance (ordenado desc).
    """
    _, feats, _ = _load_artifacts()
    explainer = _get_explainer()

    X = _align(df, feats)
    shap_values = _shap_matrix(explainer(X))

    mean_abs = np.abs(shap_values).mean(axis=0)
    out = pd.DataFrame({"feature": feats, "importance": mean_abs})
    return out.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
