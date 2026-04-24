"""Корреляционный и факторный анализ, отбор ключевых индикаторов (Глава 2.4 ВКР)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

logger = get_logger(__name__)


def correlation_with_target(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COL,
    method: str = "spearman",
    exclude_leaky: bool = True,
) -> pd.DataFrame:
    """Корреляция числовых признаков с целевой переменной.

    exclude_leaky=True — отсекает колонки из LEAKY_COL_PREFIXES (dpd, pdn_current и т.д.),
    чтобы отчёт по признакам для ML-модели был согласован с get_feature_columns.
    """
    from .feature_engineering import _is_leaky

    nums = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    if exclude_leaky:
        nums = [c for c in nums if not _is_leaky(c)]
    corr = df[nums + [target_col]].corr(method=method)[target_col].drop(target_col)
    out = (
        corr.rename("corr")
        .reset_index()
        .rename(columns={"index": "feature"})
        .assign(abs_corr=lambda x: x["corr"].abs())
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )
    return out


def pearson_matrix(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Корреляционная матрица для поиска мультиколлинеарности."""
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[features].corr()


def detect_multicollinear(
    df: pd.DataFrame, features: list[str], threshold: float = 0.9
) -> list[tuple[str, str, float]]:
    """Возвращает пары сильно коррелированных признаков (|r| > threshold)."""
    corr = df[features].corr().abs()
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and r > threshold:
                pairs.append((a, b, float(r)))
    return sorted(pairs, key=lambda x: -x[2])


def woe_iv_table(
    df: pd.DataFrame,
    feature: str,
    target_col: str = config.TARGET_COL,
    bins: int = 10,
) -> pd.DataFrame:
    """Вычисляет WoE и IV для одного признака (стандартный банковский подход)."""
    s = df[[feature, target_col]].copy().dropna()
    if s[feature].dtype.kind in "biufc":
        try:
            s["bin"] = pd.qcut(s[feature], q=bins, duplicates="drop")
        except Exception:
            s["bin"] = pd.cut(s[feature], bins=bins)
    else:
        s["bin"] = s[feature].astype(str)

    g = s.groupby("bin", observed=False)[target_col].agg(["sum", "count"])
    g.columns = ["bad", "total"]
    g["good"] = g["total"] - g["bad"]
    good_total, bad_total = g["good"].sum(), g["bad"].sum()
    g["dist_good"] = g["good"] / (good_total + 1e-9)
    g["dist_bad"] = g["bad"] / (bad_total + 1e-9)
    g["woe"] = np.log((g["dist_good"] + 1e-9) / (g["dist_bad"] + 1e-9))
    g["iv_part"] = (g["dist_good"] - g["dist_bad"]) * g["woe"]
    g["iv"] = g["iv_part"].sum()
    return g.reset_index()


def information_value_ranking(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = config.TARGET_COL,
    bins: int = 10,
) -> pd.DataFrame:
    """Ранжирование признаков по Information Value (IV)."""
    rows = []
    for f in features:
        try:
            iv = float(woe_iv_table(df, f, target_col, bins=bins)["iv"].iloc[0])
        except Exception as exc:  # pragma: no cover
            logger.warning("IV не посчитан для %s: %s", f, exc)
            iv = np.nan
        rows.append({"feature": f, "iv": iv})
    out = pd.DataFrame(rows).dropna().sort_values("iv", ascending=False).reset_index(drop=True)
    out["strength"] = pd.cut(
        out["iv"],
        bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
        labels=["unpredictive", "weak", "medium", "strong", "suspicious"],
    )
    return out


def select_key_indicators(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = config.TARGET_COL,
    top_k: int = 15,
) -> list[str]:
    """Итоговый список ключевых индикаторов для EWS / ML-модели.

    Комбинирует ранжирование по IV и Spearman, удаляет мультиколлинеарность.
    """
    iv = information_value_ranking(df, features, target_col).set_index("feature")["iv"]
    sp = correlation_with_target(df[features + [target_col]], target_col).set_index("feature")["abs_corr"]
    score = (iv.rank(ascending=True) + sp.rank(ascending=True)).sort_values(ascending=False)

    chosen: list[str] = []
    used: set[str] = set()
    corr = df[features].corr().abs()
    for f in score.index:
        if f in used:
            continue
        chosen.append(f)
        if len(chosen) >= top_k:
            break
        related = corr.columns[(corr[f] > 0.85) & (corr[f].index != f)].tolist()
        used.update(related)
    return chosen
