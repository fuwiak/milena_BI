"""Батч-скоринг новых данных обученной моделью + экспорт EWS.

Запуск:
    python scripts/run_inference.py --input data/new_slice.xlsx --output reports/scored.csv
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
from src.ews import build_ews  # noqa: E402
from src.feature_engineering import build_feature_set  # noqa: E402
from src.recommendations import build_recommendations_table  # noqa: E402
from src.utils import configure_logging, get_logger, load_pickle  # noqa: E402

_logfile = configure_logging(app="inference")
logger = get_logger("inference")
logger.info("Лог пишется в файл: %s", _logfile)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=str(config.REPORTS_DIR / "scored.csv"))
    p.add_argument("--model", default=str(config.MODEL_PATH))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Загружаю модель из %s", args.model)
    model = load_pickle(args.model)

    df = load_and_prepare(args.input, last_slice_only=True, filter_sold_forgiven=False)
    df = build_feature_set(df, include_rolling=False)

    ews = build_ews(df, model)
    recs = build_recommendations_table(ews)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out_path, index=False)
    logger.info("Сохранено %d строк в %s", len(recs), out_path)
    logger.info("Сводка по зонам:\n%s", ews.summary.to_string(index=False))


if __name__ == "__main__":
    main()
