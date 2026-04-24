"""Мониторинг дрейфа данных и модели (Глава 3.4 ВКР).

Реализовано:
    * PSI (Population Stability Index) — основной банковский метрик drift'а;
    * двухвыборочный KS-тест для числовых признаков;
    * chi-square для категориальных;
    * сравнение распределения скоров модели на train vs current;
    * расчёт просадки ROC-AUC между обучающей и свежей выборкой;
    * сохранение JSON-отчёта и (опционально) HTML-отчёта Evidently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .metrics import binary_report
from .model import TrainedModel, predict_proba
from .utils import get_logger, save_json

logger = get_logger(__name__)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. <0.1 — стабильно; 0.1..0.25 — умеренный дрейф; >0.25 — сильный."""
    reference = pd.Series(reference).dropna().values
    current = pd.Series(current).dropna().values
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=quantiles)
    cur_counts, _ = np.histogram(current, bins=quantiles)
    ref_pct = ref_counts / max(ref_counts.sum(), 1) + 1e-6
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + 1e-6
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    r = pd.Series(reference).dropna()
    c = pd.Series(current).dropna()
    if len(r) < 2 or len(c) < 2:
        return float("nan"), float("nan")
    s, p = stats.ks_2samp(r, c)
    return float(s), float(p)


def chi2_test(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    ref_counts = reference.value_counts(dropna=False)
    cur_counts = current.value_counts(dropna=False)
    idx = ref_counts.index.union(cur_counts.index)
    table = pd.DataFrame({"ref": ref_counts.reindex(idx, fill_value=0),
                          "cur": cur_counts.reindex(idx, fill_value=0)})
    if table.values.sum() == 0:
        return float("nan"), float("nan")
    chi2, p, _, _ = stats.chi2_contingency(table.T.values + 1)
    return float(chi2), float(p)


@dataclass
class DriftReport:
    features: pd.DataFrame
    score_psi: float
    model_metric_current: dict
    model_metric_reference: dict
    needs_retrain: bool
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "features": self.features.to_dict(orient="records"),
            "score_psi": self.score_psi,
            "model_metric_current": self.model_metric_current,
            "model_metric_reference": self.model_metric_reference,
            "needs_retrain": bool(self.needs_retrain),
            "summary": self.summary,
        }


def data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    num_features: Iterable[str],
    cat_features: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for f in num_features:
        if f in reference.columns and f in current.columns:
            p = psi(reference[f].values, current[f].values)
            s, pv = ks_test(reference[f].values, current[f].values)
            rows.append({"feature": f, "kind": "num", "psi": p, "ks_stat": s, "p_value": pv,
                         "drift": bool(p > config.DRIFT_PSI_THRESHOLD
                                       or (not np.isnan(pv) and pv < config.DRIFT_KS_PVALUE))})
    for f in cat_features:
        if f in reference.columns and f in current.columns:
            chi2, pv = chi2_test(reference[f].astype(str), current[f].astype(str))
            rows.append({"feature": f, "kind": "cat", "psi": np.nan, "ks_stat": chi2,
                         "p_value": pv,
                         "drift": bool((not np.isnan(pv)) and pv < config.DRIFT_KS_PVALUE)})
    return pd.DataFrame(rows).sort_values(["drift", "psi"], ascending=[False, False]).reset_index(drop=True)


def full_drift_report(
    model: TrainedModel,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: Optional[str] = config.TARGET_COL,
    output_path: Path | str | None = None,
) -> DriftReport:
    """Основная функция: сравнивает reference и current, вычисляет drift признаков и метрик."""
    num, cat = model.numeric_features, model.categorical_features
    feat_df = data_drift(reference, current, num, cat)

    ref_scores = predict_proba(model, reference)
    cur_scores = predict_proba(model, current)
    score_psi = psi(ref_scores, cur_scores)

    model_metric_current: dict = {}
    model_metric_reference: dict = {}
    if target_col and target_col in reference.columns and target_col in current.columns:
        try:
            model_metric_reference = binary_report(reference[target_col].values, ref_scores)
            model_metric_current = binary_report(current[target_col].values, cur_scores)
        except Exception as exc:
            logger.warning("Не удалось посчитать метрики на current: %s", exc)

    needs_retrain = False
    if model_metric_current and model_metric_reference:
        drop = (model_metric_reference["roc_auc"] - model_metric_current["roc_auc"]) / \
               max(model_metric_reference["roc_auc"], 1e-6)
        needs_retrain = drop > config.MODEL_METRIC_DROP_PCT
    needs_retrain = needs_retrain or score_psi > config.DRIFT_PSI_THRESHOLD or \
                    feat_df["drift"].mean() > 0.3

    report = DriftReport(
        features=feat_df,
        score_psi=float(score_psi),
        model_metric_current=model_metric_current,
        model_metric_reference=model_metric_reference,
        needs_retrain=bool(needs_retrain),
        summary={
            "features_drifted": int(feat_df["drift"].sum()),
            "features_total": len(feat_df),
            "pct_drifted": float(feat_df["drift"].mean()),
        },
    )
    if output_path:
        save_json(report.to_dict(), output_path)
        logger.info("Drift-отчёт сохранён: %s", output_path)
    return report


def try_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_html: Path | str,
) -> bool:
    """Пробует построить HTML-отчёт Evidently. Возвращает True при успехе."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        r = Report([DataDriftPreset()])
        r.run(reference_data=reference, current_data=current)
        r.save_html(str(output_html))
        return True
    except Exception as exc:
        logger.warning("Evidently отчёт не собран: %s", exc)
        return False
