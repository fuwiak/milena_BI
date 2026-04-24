"""Построение целевой переменной 'дефолт' для моделирования и EWS.

Соответствует Главам 2.3, 3.1 ВКР.

Определение дефолта (бинарная метка):
    default_target = 1, если на отчётную дату выполнено хотя бы одно из условий:
        * dpd > DEFAULT_DPD_THRESHOLD (по умолчанию 90 дней);
        * pdn_current > PDN_DEFAULT_THRESHOLD;
        * quality_category in {IV, V};
        * заполнен bankruptcy_stage (стадия банкротства).

Для панельной задачи (прогноз дефолта через N месяцев) дополнительно вычисляется
target_future_Nm — признак дефолта по тому же договору на горизонте N месяцев.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

logger = get_logger(__name__)


def build_default_flag(df: pd.DataFrame, target_col: str = config.TARGET_COL) -> pd.DataFrame:
    """Добавляет колонку default_target по состоянию на отчётную дату."""
    df = df.copy()
    dpd = pd.to_numeric(df.get("dpd"), errors="coerce").fillna(0)
    pdn = pd.to_numeric(df.get("pdn_current"), errors="coerce").fillna(0)
    quality = df.get("quality_category", pd.Series(index=df.index, dtype="string")).astype(str)
    is_bankrupt = df.get("is_bankrupt", pd.Series(0, index=df.index)).astype(int)

    df[target_col] = (
        (dpd > config.DEFAULT_DPD_THRESHOLD)
        | (pdn > config.PDN_DEFAULT_THRESHOLD)
        | (quality.isin(config.QUALITY_BAD_CATEGORIES))
        | (is_bankrupt == 1)
    ).astype(np.int8)

    rate = df[target_col].mean()
    logger.info("Доля дефолтов на срезе: %.3f%% (N=%d)", rate * 100, len(df))
    return df


def build_forward_target(
    df: pd.DataFrame,
    horizon_months: int = config.FORWARD_HORIZON_MONTHS,
    target_col: str = config.FORWARD_TARGET_COL,
    valid_mask_col: str = "has_future_obs",
) -> pd.DataFrame:
    """Для каждого (credit_id, report_date) ставит 1, если хотя бы на одном из
    ближайших `horizon_months` месяцев договор окажется в дефолте.

    Дополнительно строит bool-колонку `valid_mask_col`:
        True  — у строки есть будущее наблюдение в пределах horizon_months (метка надёжна);
        False — наблюдения кончаются раньше → метка 0 может быть «ложным нулём»,
                такие строки нужно выкидывать при обучении.

    Реализация — O(N) numpy-проход по группам (обрабатывает ~450к строк за несколько секунд).
    """
    if not {"credit_id", "report_date_as_of"}.issubset(df.columns):
        raise ValueError("Нужны колонки credit_id и report_date_as_of")

    df = build_default_flag(df)
    df = df.sort_values(["credit_id", "report_date_as_of"]).reset_index(drop=True)

    ym_arr = df["report_date_as_of"].dt.to_period("M").astype("int64").to_numpy()
    y_arr = df[config.TARGET_COL].astype(np.int8).to_numpy()

    result = np.zeros(len(df), dtype=np.int8)
    has_future = np.zeros(len(df), dtype=bool)

    for _, idx in df.groupby("credit_id", sort=False, observed=True).indices.items():
        idx = np.asarray(idx)
        g_ym = ym_arr[idx]
        g_y = y_arr[idx]
        order = np.argsort(g_ym)
        g_ym_sorted = g_ym[order]
        g_y_sorted = g_y[order]
        n = len(idx)
        for pos in range(n):
            t = g_ym_sorted[pos]
            j_lo = np.searchsorted(g_ym_sorted, t + 1, side="left")
            j_hi = np.searchsorted(g_ym_sorted, t + horizon_months, side="right")
            if j_hi > j_lo:
                has_future[idx[order[pos]]] = True
                if g_y_sorted[j_lo:j_hi].max() == 1:
                    result[idx[order[pos]]] = 1

    df[target_col] = result
    df[valid_mask_col] = has_future

    rate_all = df[target_col].mean()
    rate_valid = df.loc[df[valid_mask_col], target_col].mean() if df[valid_mask_col].any() else 0.0
    logger.info(
        "Forward-дефолт, горизонт %dм: %.3f%% (на строках с валидной разметкой: %.3f%%)",
        horizon_months, rate_all * 100, rate_valid * 100,
    )
    return df
