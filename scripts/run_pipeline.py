"""Полный пайплайн обучения: данные → EDA → features → модель → EWS → артефакты.

Запуск:
    .venv/bin/python scripts/run_pipeline.py
    .venv/bin/python scripts/run_pipeline.py --data data/3_5.xlsx --last-slice --model-type lightgbm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.correlation import information_value_ranking, select_key_indicators  # noqa: E402
from src.data_loader import load_and_prepare  # noqa: E402
from src.eda import generate_eda_report, portfolio_kpi  # noqa: E402
from src.ews import build_ews, save_rules  # noqa: E402
from src.feature_engineering import build_feature_set, get_feature_columns  # noqa: E402
from src.model import save_model, train_model  # noqa: E402
from src.recommendations import build_recommendations_table  # noqa: E402
from src.segmentation import rule_based_segment, segment_profile  # noqa: E402
from src.target import build_default_flag, build_forward_target  # noqa: E402
from src.utils import configure_logging, get_logger, save_json  # noqa: E402

_logfile = configure_logging(app="pipeline")
logger = get_logger("pipeline")
logger.info("Лог пишется в файл: %s", _logfile)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(config.RAW_DATA_PATH))
    p.add_argument("--model-type", default="lightgbm", choices=["lightgbm", "logreg"])
    p.add_argument("--last-slice", action="store_true",
                   help="Использовать только последний срез по каждому договору (быстрее)")
    p.add_argument("--fast", action="store_true",
                   help="Быстрый прогон: --last-slice + без rolling-фичей + без EDA-отчёта")
    p.add_argument("--no-eda", action="store_true", help="Пропустить генерацию EDA-отчёта")
    p.add_argument("--no-rolling", action="store_true",
                   help="Пропустить rolling-фичи (ускоряет панельный режим в ~3-5 раз)")
    p.add_argument("--target", choices=["forward", "current"], default="forward",
                   help="forward — предсказывать дефолт через N месяцев (EWS, default); "
                        "current — предсказывать текущее состояние дефолта (классификация)")
    p.add_argument("--horizon", type=int, default=config.FORWARD_HORIZON_MONTHS,
                   help="Горизонт forward-прогноза в месяцах")
    p.add_argument("--sample", type=int, default=None, help="Обучаться на сэмпле (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.fast:
        args.last_slice = True
        args.no_rolling = True
        args.no_eda = True
        args.target = "current"
        logger.info("FAST-режим: last-slice, без rolling, без EDA, target=current")

    if args.target == "forward" and args.last_slice:
        logger.warning(
            "forward-target несовместим с --last-slice (нужна история). "
            "Переключаюсь на last_slice_only=False."
        )
        args.last_slice = False

    logger.info("=== Шаг 1. Загрузка и подготовка витрины ===")
    df = load_and_prepare(args.data, last_slice_only=args.last_slice)

    logger.info("=== Шаг 2. Feature engineering ===")
    include_rolling = not (args.last_slice or args.no_rolling)
    df = build_feature_set(df, include_rolling=include_rolling)

    logger.info("=== Шаг 3. Построение целевой переменной ===")
    df = build_default_flag(df)
    if args.target == "forward":
        logger.info("Строю forward-target на горизонте %d мес.", args.horizon)
        df = build_forward_target(df, horizon_months=args.horizon)
        target_col = config.FORWARD_TARGET_COL
    else:
        target_col = config.TARGET_COL

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=config.RANDOM_STATE)
        logger.info("Обучаемся на сэмпле: %d строк", len(df))

    if not args.no_eda:
        logger.info("=== Шаг 4. EDA-отчёт ===")
        try:
            generate_eda_report(df)
        except Exception as exc:
            logger.warning("EDA-отчёт пропущен: %s", exc)

    logger.info("=== Шаг 5. Корреляционный анализ и отбор индикаторов ===")
    num_cols, _ = get_feature_columns(df)
    try:
        from src.correlation import correlation_with_target

        sample = df.sample(n=min(100_000, len(df)), random_state=config.RANDOM_STATE)
        top_by_corr = correlation_with_target(sample, target_col=target_col).head(30)[
            "feature"
        ].tolist()
        iv = information_value_ranking(sample, top_by_corr, target_col=target_col).head(25)
        logger.info("TOP IV (target=%s):\n%s", target_col, iv.head(10).to_string(index=False))
        save_json(iv.to_dict(orient="records"), config.REPORTS_DIR / "iv_ranking.json")
    except Exception as exc:
        logger.warning("IV-анализ пропущен: %s", exc)

    logger.info("=== Шаг 6. Сегментация (rule-based) ===")
    df["segment"] = rule_based_segment(df)
    if "report_date_as_of" in df.columns:
        _last = (df.sort_values("report_date_as_of")
                   .drop_duplicates(subset=["credit_id"], keep="last"))
    else:
        _last = df
    # Профиль сегментов для ВКР/дашборда — последний срез по договору (сумма = числу договоров)
    sp_last = segment_profile(
        _last, "segment", target_col=target_col, dedupe_credit_last_obs=True
    )
    logger.info("Сегменты (последний срез по credit_id):\n%s", sp_last.to_string(index=False))
    save_json(
        sp_last.to_dict(orient="records"),
        config.REPORTS_DIR / "segment_profile.json",
    )
    # Дополнительно: профиль по всей панели (траектории) — для внутреннего анализа
    sp_panel = segment_profile(
        df, "segment", target_col=target_col, dedupe_credit_last_obs=False
    )
    save_json(
        sp_panel.to_dict(orient="records"),
        config.REPORTS_DIR / "segment_profile_panel.json",
    )

    logger.info("=== Шаг 7. Обучение модели (%s, target=%s) ===", args.model_type, target_col)
    train_df = df
    if args.target == "forward" and "has_future_obs" in df.columns:
        before = len(train_df)
        train_df = df.loc[df["has_future_obs"]].copy()
        logger.info(
            "Для обучения оставляю только строки с валидной forward-разметкой: %d -> %d",
            before, len(train_df),
        )
    trained = train_model(train_df, target_col=target_col, model_type=args.model_type)
    save_model(trained)
    save_json(
        {
            "target_col": target_col,
            "forward_horizon_months": args.horizon if args.target == "forward" else None,
            "train_metrics": trained.metrics_train,
            "test_metrics": trained.metrics_test,
            "cv_metrics": trained.cv_metrics,
            "model_type": trained.model_type,
            "trained_at": trained.trained_at,
            "n_numeric": len(trained.numeric_features),
            "n_categorical": len(trained.categorical_features),
        },
        config.REPORTS_DIR / "model_metrics.json",
    )
    if trained.feature_importance is not None:
        trained.feature_importance.head(40).to_csv(
            config.REPORTS_DIR / "feature_importance.csv", index=False
        )

    logger.info("=== Шаг 8. EWS на текущем срезе ===")
    save_rules(config.EWS_RULES)
    last_slice = (
        df.sort_values("report_date_as_of").drop_duplicates(subset=["credit_id"], keep="last")
        if "report_date_as_of" in df.columns
        else df
    )
    ews = build_ews(last_slice, trained)
    recs = build_recommendations_table(ews)
    recs.head(500).to_csv(config.REPORTS_DIR / "ews_top500.csv", index=False)
    ews.summary.to_csv(config.REPORTS_DIR / "ews_summary.csv", index=False)
    logger.info("EWS summary:\n%s", ews.summary.to_string(index=False))

    logger.info("=== Шаг 9. KPI портфеля ===")
    kpi = portfolio_kpi(last_slice)
    save_json(kpi, config.REPORTS_DIR / "portfolio_kpi.json")
    logger.info("KPI: %s", kpi)

    logger.info("PIPELINE DONE. Артефакты:")
    logger.info("  - модель:       %s", config.MODEL_PATH)
    logger.info("  - правила EWS:  %s", config.EWS_RULES_PATH)
    logger.info("  - отчёты:       %s", config.REPORTS_DIR)


if __name__ == "__main__":
    main()
