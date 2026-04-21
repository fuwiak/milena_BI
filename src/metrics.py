"""Метрики качества моделей прогнозирования дефолта (Глава 3.2 ВКР)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov для бинарного скоринга — стандартная метрика в банках."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_neg = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_pos - cum_neg)))


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


def binary_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Сводный отчёт по всем основным метрикам качества."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "gini": float(gini(y_true, y_score)),
        "ks": float(ks_statistic(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "precision@thr": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall@thr": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1@thr": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": threshold,
    }


def pr_table(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    """precision/recall для набора порогов — для выбора cut-off в EWS."""
    p, r, t = precision_recall_curve(y_true, y_score)
    thresholds = np.concatenate([t, [1.0]])
    return pd.DataFrame({"threshold": thresholds, "precision": p, "recall": r})


def decile_report(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Lift / gain по децилям скоринга — показывает концентрацию риска в верхних бакетах."""
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["bucket"] = pd.qcut(df.index, q=n_bins, labels=False, duplicates="drop") + 1
    g = df.groupby("bucket").agg(n=("y", "size"), n_bad=("y", "sum"),
                                 avg_score=("score", "mean")).reset_index()
    g["bad_rate"] = g["n_bad"] / g["n"]
    g["cum_bad"] = g["n_bad"].cumsum()
    g["cum_bad_pct"] = g["cum_bad"] / g["n_bad"].sum()
    base = df["y"].mean()
    g["lift"] = g["bad_rate"] / max(base, 1e-9)
    return g
