"""Feature engineering для модели прогнозирования дефолта (Глава 3.1 ВКР).

Формирует:
    * поведенческие агрегаты (rolling-фичи по платежам и задолженности);
    * производные коэффициенты (utilization, payment_ratio, pdn_delta и т.п.);
    * признаки по клиенту (возраст, стаж, регион, программа);
    * one-hot / target encoding по категориальным полям.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

logger = get_logger(__name__)


def add_rolling_features(df: pd.DataFrame, windows: tuple[int, ...] = (1, 3, 6)) -> pd.DataFrame:
    """Добавляет rolling-средние/std и diff/pct_change ключевых признаков по credit_id.

    Оптимизации:
        * один раз сортируем;
        * для rolling используем groupby().rolling(...) batch-API — быстрее lambda-transform;
        * для diff/pct_change — встроенные groupby-методы (Cython).
    """
    required = {"credit_id", "report_date_as_of"}
    if not required.issubset(df.columns):
        return df

    df = df.sort_values(["credit_id", "report_date_as_of"]).reset_index(drop=True)

    base_cols = [c for c in ["total_debt", "available_limit", "pdn_current", "dpd", "payment_sum_1m"]
                 if c in df.columns]
    if not base_cols:
        return df

    logger.info("Rolling features: %d признаков × %d окон на %d строках",
                len(base_cols), len(windows), len(df))

    grp = df.groupby("credit_id", sort=False, observed=True)

    for col in base_cols:
        df[f"{col}_diff_1m"] = grp[col].diff(1)
        try:
            df[f"{col}_pct_change_1m"] = (
                grp[col].pct_change(1, fill_method=None)
                .replace([np.inf, -np.inf], np.nan)
            )
        except TypeError:
            df[f"{col}_pct_change_1m"] = grp[col].pct_change(1).replace([np.inf, -np.inf], np.nan)

    for w in windows:
        roll = grp[base_cols].rolling(w, min_periods=1)
        mean_df = roll.mean().reset_index(level=0, drop=True)
        for col in base_cols:
            df[f"{col}_mean_{w}m"] = mean_df[col].values
        if w >= 2:
            std_df = roll.std().reset_index(level=0, drop=True)
            for col in base_cols:
                df[f"{col}_std_{w}m"] = std_df[col].values

    return df


def add_pdn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Производные признаки по ПДН и ПСК."""
    df = df.copy()
    if {"pdn_initial", "pdn_current"}.issubset(df.columns):
        df["pdn_delta"] = df["pdn_current"] - df["pdn_initial"]
        df["pdn_ratio"] = df["pdn_current"] / (df["pdn_initial"].abs() + 1e-6)
    if {"psk_current"}.issubset(df.columns):
        df["psk_bucket"] = pd.cut(df["psk_current"], bins=[-np.inf, 10, 20, 30, np.inf],
                                  labels=["low", "mid", "high", "extreme"])
    return df


def add_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки платёжной дисциплины."""
    df = df.copy()
    p1 = df.get("payment_sum_1m")
    p2 = df.get("payment_sum_2m")
    p3 = df.get("payment_sum_3m")
    if all(x is not None for x in (p1, p2, p3)):
        df["payment_mean_3m"] = df[["payment_sum_1m", "payment_sum_2m", "payment_sum_3m"]].mean(axis=1)
        df["payment_std_3m"] = df[["payment_sum_1m", "payment_sum_2m", "payment_sum_3m"]].std(axis=1)
        df["payment_trend"] = df["payment_sum_1m"] - df["payment_sum_3m"]
    if {"cnt_all_payments", "months_on_book"}.issubset(df.columns):
        df["payments_per_month"] = df["cnt_all_payments"] / (df["months_on_book"] + 1e-6)
    return df


def add_dpd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки по просрочке."""
    df = df.copy()
    if "dpd" in df.columns:
        df["dpd_bucket"] = pd.cut(
            df["dpd"].fillna(0),
            bins=[-1, 0, 5, 30, 60, 90, 180, 10_000],
            labels=["current", "1-5", "6-30", "31-60", "61-90", "91-180", "180+"],
        )
    if {"expired_actual_days_count", "months_on_book"}.issubset(df.columns):
        df["overdue_density"] = df["expired_actual_days_count"] / (30 * df["months_on_book"] + 1e-6)
    return df


def build_feature_set(
    df: pd.DataFrame,
    include_rolling: bool = True,
) -> pd.DataFrame:
    """Склеивает все feature-блоки. Работает и на «последнем срезе», и на панели."""
    df = add_pdn_features(df)
    df = add_payment_features(df)
    df = add_dpd_features(df)
    if include_rolling:
        df = add_rolling_features(df)
    return df


TECHNICAL_COLS: tuple[str, ...] = (
    "credit_id",
    "borrower_id",
    "report_date_as_of",
    "date_borrower_birthday",
    "date_return",
    "date_issued",
    "date_signing",
    "last_payment_date",
    "employer",
    config.TARGET_COL,
    config.FORWARD_TARGET_COL,
    "has_future_obs",
    "segment",
    "cluster",
    "sold_flg",
    "forgiven_flg",
)

LEAKY_COL_PREFIXES: tuple[str, ...] = (
    "dpd",
    "pdn_current",
    "quality_category",
    "bankruptcy_stage",
    "is_bankrupt",
    "expired_actual_days_count",
    "percent_expired_actual_days_count",
    "reserve_rate",
    "ifrs_provision_rate",
    "total_debt_reserve",
    "total_debt_provision",
    "available_limit_reserve",
    "available_limit_provision",
    "overdue_density",
)


def _is_leaky(col: str) -> bool:
    return any(col == p or col.startswith(p + "_") for p in LEAKY_COL_PREFIXES)


def get_feature_columns(
    df: pd.DataFrame, exclude_leaky: bool = True
) -> tuple[list[str], list[str]]:
    """Возвращает (numeric_features, categorical_features) для модели.

    exclude_leaky=True (default) — исключает поля, из которых построен таргет, чтобы
    избежать target leakage. Отключайте только для диагностики.
    """
    tech = set(TECHNICAL_COLS)
    num, cat = [], []
    for c in df.columns:
        if c in tech:
            continue
        if exclude_leaky and _is_leaky(c):
            continue
        dt = df[c].dtype
        if dt.kind in "biufc":
            num.append(c)
        else:
            cat.append(c)
    return num, cat
