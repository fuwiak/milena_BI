"""Загрузка и первичная очистка витрины портфеля кредитных карт.

Соответствует Главе 2.1 ВКР: структура таблицы, типизация полей,
первичная обработка пропусков, фильтрация списанных и прощённых долгов.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger, reduce_mem_usage

logger = get_logger(__name__)


def load_raw_data(path: Path | str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Читает xlsx/parquet/pickle-файл витрины.

    При первом чтении xlsx кладёт рядом быстрый кэш:
        * .parquet (если установлен pyarrow/fastparquet),
        * либо .pkl (fallback) — всегда работает.
    Повторные запуски грузят кэш за ~1 секунду вместо ~2 минут.
    """
    path = Path(path)
    parquet_cache = path.with_suffix(".parquet")
    pickle_cache = path.with_suffix(".pkl")

    if parquet_cache.exists():
        logger.info("Загружаю данные из parquet-кэша: %s", parquet_cache)
        try:
            return pd.read_parquet(parquet_cache)
        except Exception as exc:
            logger.warning("Parquet-кэш битый (%s), пересобираю", exc)

    if pickle_cache.exists():
        logger.info("Загружаю данные из pickle-кэша: %s", pickle_cache)
        try:
            return pd.read_pickle(pickle_cache)
        except Exception as exc:
            logger.warning("Pickle-кэш битый (%s), пересобираю", exc)

    logger.info("Загружаю данные из %s (xlsx, долго, ~1-2 мин)", path)
    df = pd.read_excel(path, engine="openpyxl")
    logger.info("Прочитано строк: %d, колонок: %d", len(df), df.shape[1])

    saved = False
    try:
        df.to_parquet(parquet_cache, index=False)
        logger.info("Сохранён parquet-кэш: %s", parquet_cache)
        saved = True
    except Exception as exc:
        logger.warning("Parquet недоступен (%s), падаю в pickle", exc)
    if not saved:
        try:
            df.to_pickle(pickle_cache)
            logger.info("Сохранён pickle-кэш: %s", pickle_cache)
        except Exception as exc:
            logger.warning("Не удалось сохранить pickle-кэш: %s", exc)
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит типы колонок: даты, числа, категории."""
    df = df.copy()
    for col in config.DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in config.NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in config.FLAG_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)

    for col in config.CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("UNKNOWN")
    return df


def drop_noisy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет колонки с большим числом пропусков / низкой инф.ценностью."""
    return df.drop(columns=[c for c in config.DROP_COLS if c in df.columns], errors="ignore")


def filter_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Из анализа исключаем проданные и прощённые долги."""
    mask = ((df.get("sold_flg", 0) != 1) & (df.get("forgiven_flg", 0) != 1))
    before = len(df)
    df = df.loc[mask].copy()
    logger.info("Фильтрация sold_flg/forgiven_flg: %d -> %d", before, len(df))
    return df


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет базовые производные поля (возраст, стаж, utilization и т.п.)."""
    df = df.copy()

    ref = df.get("report_date_as_of", pd.Series(pd.Timestamp.today(), index=df.index))

    if "date_borrower_birthday" in df.columns:
        df["age"] = ((ref - df["date_borrower_birthday"]).dt.days / 365.25).round(1)
    if "date_issued" in df.columns:
        df["months_on_book"] = ((ref - df["date_issued"]).dt.days / 30.44).round(1)
    if "last_payment_date" in df.columns:
        df["days_since_last_payment"] = (ref - df["last_payment_date"]).dt.days

    if {"initial_amount", "available_limit"}.issubset(df.columns):
        df["utilization"] = (
            (df["initial_amount"] - df["available_limit"]) / (df["initial_amount"].abs() + 1e-6)
        ).clip(lower=0, upper=1.5)

    if {"payment_sum_1m", "total_debt"}.issubset(df.columns):
        df["payment_ratio_1m"] = df["payment_sum_1m"] / (df["total_debt"].abs() + 1e-6)

    if {"payment_sum_1m", "payment_sum_2m"}.issubset(df.columns):
        df["payment_ratio_mom"] = df["payment_sum_1m"] / (df["payment_sum_2m"].abs() + 1e-6)

    if "bankruptcy_stage" in df.columns:
        df["is_bankrupt"] = (
            df["bankruptcy_stage"].notna() & (df["bankruptcy_stage"].astype(str) != "UNKNOWN")
        ).astype(np.int8)

    return df


def take_last_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет по каждому credit_id только последнюю отчётную дату (для табличных моделей)."""
    if "report_date_as_of" not in df.columns or "credit_id" not in df.columns:
        return df
    df = df.sort_values(["credit_id", "report_date_as_of"])
    df = df.drop_duplicates(subset=["credit_id"], keep="last")
    logger.info("После take_last_slice: %d уникальных договоров", len(df))
    return df


def load_and_prepare(
    path: Path | str = config.RAW_DATA_PATH,
    *,
    last_slice_only: bool = False,
    filter_sold_forgiven: bool = True,
) -> pd.DataFrame:
    """Единая точка входа: загрузка → типизация → очистка → производные поля."""
    df = load_raw_data(path)
    df = cast_types(df)
    df = drop_noisy_columns(df)
    if filter_sold_forgiven:
        df = filter_portfolio(df)
    df = add_derived_fields(df)
    if last_slice_only:
        df = take_last_slice(df)
    df = reduce_mem_usage(df)
    return df
