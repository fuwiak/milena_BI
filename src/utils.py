"""Вспомогательные функции: логирование в stdout + файл, сохранение/загрузка моделей."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import joblib

from . import config

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_FILE_HANDLERS_ATTACHED: set[str] = set()  # чтобы не дублировать file-handler'ы


def _build_file_handler(logfile: Path) -> RotatingFileHandler:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        logfile, maxBytes=10 * 1024 * 1024, backupCount=7, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    return handler


def configure_logging(
    app: str = "pipeline",
    level: int = logging.INFO,
    log_dir: Path | str = config.LOGS_DIR,
) -> Path:
    """Настраивает корневое логирование на stdout + файл.

    Файл: logs/<app>_YYYYMMDD.log, ротация 10 MB × 7 файлов.
    Применяется к root-логгеру — все дочерние (src.*, api, pipeline) пишут туда же.
    Возвращает путь к активному файлу логов.
    """
    root = logging.getLogger()
    root.setLevel(level)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{app}_{dt.datetime.now():%Y%m%d}.log"

    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
                     for h in root.handlers)
    if not has_stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(sh)

    key = str(logfile)
    if key not in _FILE_HANDLERS_ATTACHED:
        root.addHandler(_build_file_handler(logfile))
        _FILE_HANDLERS_ATTACHED.add(key)

    os.environ.setdefault("MILENA_LOGFILE", key)
    return logfile


def get_logger(name: str = "milena_bi", level: int = logging.INFO) -> logging.Logger:
    """Дочерний логгер. Если корень ещё не сконфигурен — включаем stdout+файл по умолчанию."""
    if not logging.getLogger().handlers:
        configure_logging(app="milena_bi", level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def save_pickle(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path: Path) -> Any:
    return joblib.load(Path(path))


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def reduce_mem_usage(df, verbose: bool = False):
    """Снижает расход памяти DataFrame через сужение типов (для больших панелей)."""
    import numpy as np

    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type.kind in {"i", "u"}:
            c_min, c_max = df[col].min(), df[col].max()
            for dt in (np.int8, np.int16, np.int32, np.int64):
                if c_min >= np.iinfo(dt).min and c_max <= np.iinfo(dt).max:
                    df[col] = df[col].astype(dt)
                    break
        elif col_type.kind == "f":
            df[col] = df[col].astype(np.float32)
    if verbose:
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Mem: {start_mem:.1f} -> {end_mem:.1f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% saved)")
    return df
