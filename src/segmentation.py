"""Сегментация клиентов по поведенческим и риск-характеристикам (глава 2 ВКР).

Реализованы два подхода:

1. rule_based_segment — сегментация на основе заранее заданных бизнес-правил
   (активность, просрочка, использование лимита) — четыре сегмента:
       * 'stable'     — регулярные платежи, отсутствие просрочки;
       * 'growing'    — активное использование лимита, но платёжная дисциплина в норме;
       * 'stress'     — ранние признаки ухудшения (рост долга, падение платежей);
       * 'delinquent' — уже просрочка / банкротство.

2. cluster_segment — KMeans-кластеризация по стандартизированным поведенческим признакам.
   Количество кластеров подбирается по правилу локтя (см. find_optimal_k).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

logger = get_logger(__name__)

BEHAVIOR_FEATURES: list[str] = [
    "utilization",
    "payment_ratio_1m",
    "payment_ratio_mom",
    "dpd",
    "pdn_current",
    "days_since_last_payment",
    "cnt_all_payments",
]


def rule_based_segment(df: pd.DataFrame) -> pd.Series:
    """Назначает сегмент по простым бизнес-правилам."""
    seg = pd.Series("stable", index=df.index, dtype="object")

    util = df.get("utilization", pd.Series(0, index=df.index))
    dpd = df.get("dpd", pd.Series(0, index=df.index))
    pay_mom = df.get("payment_ratio_mom", pd.Series(1, index=df.index))
    dslp = df.get("days_since_last_payment", pd.Series(0, index=df.index))
    is_bankrupt = df.get("is_bankrupt", pd.Series(0, index=df.index))

    seg.loc[(util > 0.6) & (dpd <= 5)] = "growing"
    seg.loc[(pay_mom < 0.5) | (dslp > 45)] = "stress"
    seg.loc[(dpd > 30) | (is_bankrupt == 1)] = "delinquent"
    return seg


def segment_profile(
    df: pd.DataFrame,
    seg_col: str = "segment",
    target_col: str = config.TARGET_COL,
    *,
    dedupe_credit_last_obs: bool = True,
) -> pd.DataFrame:
    """Профиль по каждому сегменту: число договоров, долг, уровень дефолта.

    Если передана вся панель (несколько строк на ``credit_id``), без дедупликации
    один договор может учитываться в *разных* сегментах в разные месяцы: тогда
    сумма ``n_contracts`` по строкам таблицы превышает реальное число договоров.

    По умолчанию оставляем по каждому ``credit_id`` одну строку — с максимальной
    ``report_date_as_of`` (как актуальный срез портфеля).
    """
    if seg_col not in df.columns:
        raise KeyError(seg_col)

    work = df
    if (
        dedupe_credit_last_obs
        and "credit_id" in df.columns
        and "report_date_as_of" in df.columns
        and df["credit_id"].duplicated().any()
    ):
        before = len(df)
        work = (
            df.sort_values(["report_date_as_of", "credit_id"], na_position="last")
            .drop_duplicates(subset=["credit_id"], keep="last")
        )
        logger.info(
            "segment_profile: срез последней даты по credit_id: %d -> %d строк",
            before,
            len(work),
        )

    agg = {"credit_id": "nunique"}
    for c in ["total_debt", "available_limit", "payment_sum_1m"]:
        if c in df.columns:
            agg[c] = "sum"
    for c in ["pdn_current", "dpd", "utilization"]:
        if c in df.columns:
            agg[c] = "mean"
    if target_col in df.columns:
        agg[target_col] = "mean"

    out = work.groupby(seg_col, dropna=False).agg(agg).reset_index()
    out = out.rename(columns={"credit_id": "n_contracts", target_col: "default_rate"})
    return out.sort_values("n_contracts", ascending=False).reset_index(drop=True)


def _prepare_matrix(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    features = [c for c in features if c in df.columns]
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X


def find_optimal_k(df: pd.DataFrame, features: Iterable[str] = BEHAVIOR_FEATURES,
                   k_range: Iterable[int] = (2, 3, 4, 5, 6, 7)) -> pd.DataFrame:
    """Считает inertia KMeans и силуэт для выбора K."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X = _prepare_matrix(df, features)
    Xs = StandardScaler().fit_transform(X)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels, sample_size=min(10_000, len(Xs)),
                               random_state=config.RANDOM_STATE)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def cluster_segment(df: pd.DataFrame, n_clusters: int = 4,
                    features: Iterable[str] = BEHAVIOR_FEATURES) -> pd.Series:
    """Обучает KMeans и возвращает метки кластеров."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = _prepare_matrix(df, features)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(Xs)
    logger.info("KMeans (k=%d) обучен на %d строках, %d признаках", n_clusters, len(X), X.shape[1])
    return pd.Series(labels, index=df.index, name="cluster")
