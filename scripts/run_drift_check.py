"""Мониторинг data/model drift на новой выгрузке (Глава 3.4 ВКР).

Запуск:
    python scripts/run_drift_check.py --new data/new_slice.xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data_loader import load_and_prepare  # noqa: E402
from src.drift_monitor import full_drift_report, try_evidently_report  # noqa: E402
from src.feature_engineering import build_feature_set  # noqa: E402
from src.target import build_default_flag  # noqa: E402
from src.utils import configure_logging, get_logger, load_pickle  # noqa: E402

_logfile = configure_logging(app="drift")
logger = get_logger("drift")
logger.info("Лог пишется в файл: %s", _logfile)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default=str(config.RAW_DATA_PATH),
                   help="Опорная (обучающая) витрина")
    p.add_argument("--new", required=True, help="Новый срез данных")
    p.add_argument("--out-json", default=str(config.REPORTS_DIR / "drift_report.json"))
    p.add_argument("--out-html", default=str(config.REPORTS_DIR / "drift_report.html"))
    return p.parse_args()


def _prepare(path: str):
    df = load_and_prepare(path, last_slice_only=True)
    df = build_feature_set(df, include_rolling=False)
    df = build_default_flag(df)
    return df


def main() -> None:
    args = parse_args()
    model = load_pickle(config.MODEL_PATH)

    logger.info("Готовлю reference...")
    reference = _prepare(args.reference)
    logger.info("Готовлю current...")
    current = _prepare(args.new)

    report = full_drift_report(model, reference, current, output_path=args.out_json)
    logger.info("PSI скоринга: %.4f", report.score_psi)
    logger.info("Дрейф в %d/%d признаках", report.summary["features_drifted"],
                report.summary["features_total"])
    logger.info("Нужно переобучение: %s", report.needs_retrain)

    ok = try_evidently_report(reference, current, args.out_html)
    logger.info("Evidently HTML: %s", args.out_html if ok else "skipped")


if __name__ == "__main__":
    main()
