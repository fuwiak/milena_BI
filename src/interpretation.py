"""Интерпретация модели: важности, SHAP-значения, объяснение прогноза по клиенту (3.2).

SHAP нужен и для аналитиков (визуализация глобальных/локальных эффектов),
и для BI-дашборда (колонка «причины высокого риска» для каждого клиента).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .model import TrainedModel
from .utils import get_logger

logger = get_logger(__name__)


def permutation_importance_report(
    model: TrainedModel,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """Permutation importance на готовом пайплайне (учитывает препроцессинг)."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model.pipeline, X, y, n_repeats=n_repeats,
        random_state=random_state, scoring=scoring, n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def compute_shap_values(model: TrainedModel, X: pd.DataFrame, sample_size: int = 5_000):
    """Считает SHAP для подвыборки (LightGBM/XGBoost — TreeExplainer; иначе LinearExplainer)."""
    import shap

    pre = model.pipeline.named_steps["preprocessor"]
    clf = model.pipeline.named_steps["classifier"]
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)

    X_t = pre.transform(X)
    feature_names = pre.get_feature_names_out()

    if hasattr(clf, "booster_") or hasattr(clf, "get_booster"):
        explainer = shap.TreeExplainer(clf)
    elif hasattr(clf, "coef_"):
        explainer = shap.LinearExplainer(clf, X_t)
    else:
        explainer = shap.Explainer(clf, X_t)

    shap_values = explainer.shap_values(X_t)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    return shap_values, feature_names


def explain_one(
    model: TrainedModel,
    row: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """Возвращает top-N признаков, повлиявших на итоговый риск-скор для клиента."""
    shap_values, names = compute_shap_values(model, row, sample_size=1)
    arr = shap_values[0]
    idx = np.argsort(-np.abs(arr))[:top_n]
    return [{"feature": names[i], "shap": float(arr[i]), "value": float(np.ravel(row.values)[i] if i < row.shape[1] else np.nan)}
            for i in idx]


def shap_top_features(
    model: TrainedModel,
    X: pd.DataFrame,
    sample_size: int = 5_000,
    top_n: int = 20,
) -> pd.DataFrame:
    """Глобальный топ признаков по среднему |SHAP|."""
    shap_values, names = compute_shap_values(model, X, sample_size=sample_size)
    imp = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": names, "shap_mean_abs": imp})
        .sort_values("shap_mean_abs", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
