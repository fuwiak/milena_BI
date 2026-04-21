"""EDA и описание портфеля кредитных карт (Главы 2.1, 2.3 ВКР).

Содержит функции для:
  * краткого описания структуры датасета (типы, пропуски, уникальность);
  * расчёта распределений ключевых поведенческих показателей;
  * агрегатов портфеля по регионам, программам, категориям качества;
  * динамики дефолтности и поведенческих метрик во времени;
  * сохранения HTML-отчёта со всеми графиками.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Описательные таблицы
# ---------------------------------------------------------------------------
def describe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает сводку по типам/пропускам/уникальности всех колонок."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_missing": int(s.isna().sum()),
                "pct_missing": round(s.isna().mean() * 100, 2),
                "n_unique": int(s.nunique(dropna=True)),
                "sample": s.dropna().iloc[0] if s.dropna().size else None,
            }
        )
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Расширенная описательная статистика по числовым полям."""
    nums = [c for c in config.NUMERIC_COLS if c in df.columns]
    desc = df[nums].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    desc["n_missing"] = df[nums].isna().sum()
    desc["pct_missing"] = (df[nums].isna().mean() * 100).round(2)
    return desc


# ---------------------------------------------------------------------------
# Агрегаты по портфелю
# ---------------------------------------------------------------------------
def portfolio_kpi(df: pd.DataFrame, target_col: str = config.TARGET_COL) -> dict:
    """Ключевые KPI портфеля для BI-дашборда."""
    out = {
        "n_contracts": int(df["credit_id"].nunique()) if "credit_id" in df else len(df),
        "n_borrowers": int(df["borrower_id"].nunique()) if "borrower_id" in df else np.nan,
        "total_debt": float(df["total_debt"].sum()) if "total_debt" in df else np.nan,
        "available_limit": float(df["available_limit"].sum()) if "available_limit" in df else np.nan,
        "avg_pdn_current": float(df["pdn_current"].mean()) if "pdn_current" in df else np.nan,
        "avg_dpd": float(df["dpd"].mean()) if "dpd" in df else np.nan,
        "default_rate": float(df[target_col].mean()) if target_col in df else np.nan,
        "reserve_amount": float(df["total_debt_reserve"].sum())
        if "total_debt_reserve" in df
        else np.nan,
    }
    return out


def aggregate_by(
    df: pd.DataFrame,
    by: str,
    target_col: str = config.TARGET_COL,
) -> pd.DataFrame:
    """Агрегирует показатели портфеля в разрезе by (регион/программа/работодатель)."""
    if by not in df.columns:
        raise KeyError(by)
    agg_map: dict = {"credit_id": "nunique"}
    for c in ["total_debt", "available_limit", "total_debt_reserve", "payment_sum_1m"]:
        if c in df.columns:
            agg_map[c] = "sum"
    for c in ["pdn_current", "dpd", "utilization"]:
        if c in df.columns:
            agg_map[c] = "mean"
    if target_col in df.columns:
        agg_map[target_col] = "mean"
    out = df.groupby(by, dropna=False).agg(agg_map).reset_index()
    out = out.rename(columns={"credit_id": "n_contracts", target_col: "default_rate"})
    return out.sort_values("n_contracts", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Временная динамика
# ---------------------------------------------------------------------------
def default_rate_over_time(df: pd.DataFrame, target_col: str = config.TARGET_COL) -> pd.DataFrame:
    """Динамика доли дефолтов и объёма задолженности по отчётным датам."""
    if "report_date_as_of" not in df.columns:
        return pd.DataFrame()
    g = df.groupby(df["report_date_as_of"].dt.to_period("M"))
    agg_kwargs: dict = {
        "n_contracts": ("credit_id", "nunique"),
        "total_debt": ("total_debt", "sum") if "total_debt" in df.columns else ("credit_id", "count"),
    }
    if target_col in df.columns:
        agg_kwargs["default_rate"] = (target_col, "mean")
    if "dpd" in df.columns:
        agg_kwargs["avg_dpd"] = ("dpd", "mean")
    out = g.agg(**agg_kwargs).reset_index()
    out["report_date_as_of"] = out["report_date_as_of"].dt.to_timestamp()
    return out


def pre_default_trajectory(
    df: pd.DataFrame,
    feature: str = "total_debt",
    months_before: int = 6,
) -> pd.DataFrame:
    """Средняя траектория признака за N месяцев ДО дефолта (Глава 2.3).

    Идея: для каждого credit_id находим месяц первого дефолта, и выравниваем ось времени
    так, чтобы t=0 совпадало с первым месяцем дефолта, t=-1, -2, ... — месяцы до него.
    """
    if {"credit_id", "report_date_as_of", feature, config.TARGET_COL}.issubset(df.columns) is False:
        return pd.DataFrame()
    tmp = df[["credit_id", "report_date_as_of", feature, config.TARGET_COL]].copy()
    tmp["ym"] = tmp["report_date_as_of"].dt.to_period("M").astype("int64")
    first_def = (
        tmp.loc[tmp[config.TARGET_COL] == 1]
        .groupby("credit_id")["ym"]
        .min()
        .rename("first_default_ym")
    )
    tmp = tmp.join(first_def, on="credit_id")
    tmp = tmp.dropna(subset=["first_default_ym"])
    tmp["t"] = tmp["ym"] - tmp["first_default_ym"]
    tmp = tmp[(tmp["t"] <= 0) & (tmp["t"] >= -months_before)]
    return tmp.groupby("t")[feature].agg(["mean", "median", "count"]).reset_index()


# ---------------------------------------------------------------------------
# Полный EDA-отчёт
# ---------------------------------------------------------------------------
def generate_eda_report(
    df: pd.DataFrame,
    output_dir: Path = config.REPORTS_DIR,
    sample_size: int = 50_000,
) -> Path:
    """Генерирует HTML-отчёт с описательной статистикой и графиками.

    Для ускорения рендеринга гистограмм используется случайный сэмпл (sample_size).
    Все агрегаты и описательные статистики считаются на полном df.
    """
    import plotly.express as px
    import plotly.io as pio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eda.html"

    schema = describe_schema(df)
    kpi = portfolio_kpi(df)
    df_plot = df.sample(n=min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df

    html_parts: list[str] = [
        "<html><head><meta charset='utf-8'><title>EDA портфеля КК</title>",
        "<style>body{font-family:Arial;padding:24px;} h1,h2{color:#1f3b6d;}"
        " table{border-collapse:collapse;} td,th{border:1px solid #ddd;padding:4px 8px;}"
        "</style></head><body>",
        "<h1>EDA — Портфель кредитных карт</h1>",
        "<h2>Ключевые KPI</h2>",
        pd.DataFrame([kpi]).T.rename(columns={0: "value"}).to_html(),
        "<h2>Схема таблицы и пропуски</h2>",
        schema.to_html(index=False),
    ]

    if "region_from_address" in df.columns:
        top_reg = aggregate_by(df, "region_from_address").head(20)
        fig = px.bar(top_reg, x="region_from_address", y="n_contracts",
                     title="ТОП-20 регионов по числу договоров")
        html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))

    dyn = default_rate_over_time(df)
    if not dyn.empty:
        fig = px.line(dyn, x="report_date_as_of", y="default_rate",
                      title="Динамика доли дефолтов по месяцам")
        html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    for num_col in ["dpd", "pdn_current", "utilization", "payment_sum_1m"]:
        if num_col in df_plot.columns:
            fig = px.histogram(df_plot, x=num_col, nbins=60, title=f"Распределение {num_col}")
            html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    html_parts.append("</body></html>")
    report_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info("EDA-отчёт сохранён: %s", report_path)
    return report_path
