"""Сборка компактного кэша для Streamlit-дашборда (для деплоя на Railway / любой хостинг).

Скрипт читает исходный XLSX + модель, выполняет feature engineering (включая forward-target
как в run_pipeline для таблицы сегментов ВКР), скоринг и EWS,
затем сохраняет три компактных parquet-файла, которые дашборд умеет читать мгновенно:

    reports/dashboard_panel.parquet      — панельные данные (для drilldown-графиков), compact
    reports/dashboard_scored.parquet     — последний срез с risk_score / zone / rules / recs
    reports/dashboard_timeseries.parquet — агрегаты по датам (default rate, total debt)

После этого xlsx больше не нужен в рантайме дашборда — на Railway нужно закоммитить только
~5-20 МБ parquet вместо 107 МБ xlsx.

Запуск (нужен ``lightgbm`` для распаковки ``model.pkl`` — используйте venv проекта):
    .venv/bin/python scripts/build_dashboard_cache.py
или:
    ./scripts/with_venv.sh scripts/build_dashboard_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data_loader import load_and_prepare, take_last_slice  # noqa: E402
from src.eda import default_rate_over_time  # noqa: E402
from src.ews import apply_rules, load_rules, zones_after_rules  # noqa: E402
from src.feature_engineering import build_feature_set  # noqa: E402
from src.model import predict_proba  # noqa: E402
from src.recommendations import recommend_for_client  # noqa: E402
from src.segmentation import rule_based_segment  # noqa: E402
from src.target import build_forward_target  # noqa: E402
from src.utils import configure_logging, get_logger, load_pickle  # noqa: E402

configure_logging()
logger = get_logger(__name__)


PANEL_COLS: list[str] = [
    "credit_id",
    "borrower_id",
    "report_date_as_of",
    "total_debt",
    "available_limit",
    "dpd",
    "pdn_current",
    "payment_sum_1m",
    "quality_category",
    "is_bankrupt",
    "sex",
    "loan_program",
    "region_from_address",
    "age",
    "months_on_book",
    "utilization",
    "payment_ratio_mom",
    "days_since_last_payment",
    "default_target",
    "default_future",
    "total_debt_reserve",
]

SCORED_EXTRA: list[str] = [
    "risk_score",
    "zone",
    "rules_triggered",
    "rules_weight_sum",
    "recommendations",
    "segment",
    "priority",
]


def _select_existing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].copy()


def _shrink_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Даункастим числа для уменьшения размера parquet."""
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include=["object"]).columns:
        if c in ("rules_triggered", "recommendations"):
            continue
        nunique = df[c].nunique(dropna=True)
        if nunique and nunique < max(1000, len(df) // 20):
            df[c] = df[c].astype("category")
    return df


def main() -> None:
    out_dir = config.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== 1. Загрузка и feature engineering (с rolling) ===")
    df = load_and_prepare(config.RAW_DATA_PATH)
    df = build_feature_set(df, include_rolling=True)
    df = build_forward_target(df, horizon_months=config.FORWARD_HORIZON_MONTHS)
    df["segment"] = rule_based_segment(df)
    logger.info("Панель: %d строк, %d колонок", *df.shape)

    logger.info("=== 2. Последний срез ===")
    if "report_date_as_of" in df.columns:
        last = take_last_slice(df).reset_index(drop=True)
    else:
        last = df.copy()
    logger.info("Срез на последнюю дату: %d договоров", len(last))

    logger.info("=== 3. Скоринг и EWS ===")
    model = load_pickle(config.MODEL_PATH) if config.MODEL_PATH.exists() else None
    model_cols: list[str] = []
    if model is not None:
        model_cols = list(getattr(model, "numeric_features", []) or []) + \
                     list(getattr(model, "categorical_features", []) or [])
    if model is None:
        logger.warning("model.pkl не найден — risk_score будет 0")
        last["risk_score"] = 0.0
        last["zone"] = "green"
        last["rules_triggered"] = [[] for _ in range(len(last))]
        last["rules_weight_sum"] = 0
    else:
        last["risk_score"] = predict_proba(model, last)
        try:
            rules, zones = load_rules()
        except Exception:
            rules, zones = config.EWS_RULES, config.EWS_ZONES
        rules_df = apply_rules(last, rules)
        last["rules_triggered"] = rules_df["rules_triggered"].values
        last["rules_weight_sum"] = rules_df["rules_weight_sum"].values
        last["zone"] = zones_after_rules(last["risk_score"], last["rules_weight_sum"], zones)

    last["recommendations"] = last.apply(recommend_for_client, axis=1)
    last["priority"] = last["risk_score"] * last.get("total_debt", pd.Series(0, index=last.index)).clip(lower=0)

    logger.info("=== 4. Timeseries для графиков ===")
    ts = default_rate_over_time(df) if "report_date_as_of" in df.columns else pd.DataFrame()

    logger.info("=== 5. Сохранение parquet ===")
    panel_out = _shrink_dtypes(_select_existing(df, PANEL_COLS))
    scored_cols = list(dict.fromkeys(PANEL_COLS + SCORED_EXTRA + model_cols))
    scored_out = _shrink_dtypes(_select_existing(last, scored_cols))
    ts_out = ts.copy() if isinstance(ts, pd.DataFrame) else pd.DataFrame()

    panel_path = out_dir / "dashboard_panel.parquet"
    scored_path = out_dir / "dashboard_scored.parquet"
    ts_path = out_dir / "dashboard_timeseries.parquet"

    panel_out.to_parquet(panel_path, index=False, compression="zstd")
    scored_out.to_parquet(scored_path, index=False, compression="zstd")
    if not ts_out.empty:
        ts_out.to_parquet(ts_path, index=False, compression="zstd")

    for p in (panel_path, scored_path, ts_path):
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            logger.info("  %-40s %6.2f MB", p.name, size_mb)

    logger.info("OK. Дашборд теперь может работать без raw xlsx.")


if __name__ == "__main__":
    main()
