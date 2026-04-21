"""Обучение и сериализация моделей прогнозирования дефолта (Глава 3.1-3.2 ВКР).

Базовое решение: LogisticRegression как интерпретируемый baseline.
Продвинутое: LightGBM поверх единого ColumnTransformer. Сохраняются оба,
LightGBM используется как production-модель.

Пайплайн:
    ColumnTransformer(num=SimpleImputer+StandardScaler, cat=OneHotEncoder(min_freq=0.01))
        -> Classifier

Используется TimeSeriesSplit, если известны отчётные даты, иначе StratifiedKFold.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# sklearn шумит про имена фич (LightGBM обёртка), пропуски в пустых std_1m и т.п.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .feature_engineering import get_feature_columns
from .metrics import binary_report
from .utils import get_logger, save_pickle

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Структуры
# ---------------------------------------------------------------------------
@dataclass
class TrainedModel:
    """Контейнер обученной модели со всеми артефактами, нужными сервису и дашборду."""

    pipeline: Pipeline
    numeric_features: list[str]
    categorical_features: list[str]
    metrics_train: dict = field(default_factory=dict)
    metrics_test: dict = field(default_factory=dict)
    cv_metrics: dict = field(default_factory=dict)
    feature_importance: pd.DataFrame | None = None
    model_type: str = "lightgbm"
    trained_at: str = ""


# ---------------------------------------------------------------------------
# Предобработка
# ---------------------------------------------------------------------------
def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    try:
        cat_enc = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=True)
    except TypeError:
        cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", cat_enc),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_classifier(model_type: str):
    if model_type == "logreg":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=config.RANDOM_STATE,
        )
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier  # lazy import
        return LGBMClassifier(**config.LGBM_PARAMS)
    raise ValueError(f"Unknown model_type={model_type}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_model(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COL,
    model_type: str = "lightgbm",
    test_size: float = config.TEST_SIZE,
    time_col: Optional[str] = "report_date_as_of",
    n_folds: int = config.N_FOLDS,
) -> TrainedModel:
    """Обучает модель и считает метрики train/test + cross-validation."""
    if target_col not in df.columns:
        raise KeyError(f"target {target_col} отсутствует в датасете")

    num_cols, cat_cols = get_feature_columns(df)
    logger.info("Признаки: numeric=%d, categorical=%d", len(num_cols), len(cat_cols))

    X = df[num_cols + cat_cols]
    y = df[target_col].astype(int)

    if time_col in df.columns and df[time_col].notna().any():
        df_sorted = df.sort_values(time_col)
        X_sorted = df_sorted[num_cols + cat_cols]
        y_sorted = df_sorted[target_col].astype(int)
        split = int(len(df_sorted) * (1 - test_size))
        X_train, X_test = X_sorted.iloc[:split], X_sorted.iloc[split:]
        y_train, y_test = y_sorted.iloc[:split], y_sorted.iloc[split:]
        cv = TimeSeriesSplit(n_splits=n_folds)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE
        )
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("classifier", _make_classifier(model_type)),
    ])

    logger.info("Запускаем CV (%d folds, ROC-AUC)...", n_folds)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info("CV ROC-AUC: mean=%.4f, std=%.4f", cv_scores.mean(), cv_scores.std())

    pipeline.fit(X_train, y_train)

    train_scores = pipeline.predict_proba(X_train)[:, 1]
    test_scores = pipeline.predict_proba(X_test)[:, 1]

    metrics_train = binary_report(y_train.values, train_scores)
    metrics_test = binary_report(y_test.values, test_scores)
    logger.info("Train ROC-AUC=%.4f | Test ROC-AUC=%.4f | KS=%.3f | Gini=%.3f",
                metrics_train["roc_auc"], metrics_test["roc_auc"],
                metrics_test["ks"], metrics_test["gini"])

    feature_importance = _extract_feature_importance(pipeline, num_cols, cat_cols)

    return TrainedModel(
        pipeline=pipeline,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        metrics_train=metrics_train,
        metrics_test=metrics_test,
        cv_metrics={"cv_roc_auc_mean": float(cv_scores.mean()),
                    "cv_roc_auc_std": float(cv_scores.std())},
        feature_importance=feature_importance,
        model_type=model_type,
        trained_at=pd.Timestamp.utcnow().isoformat(),
    )


def _extract_feature_importance(
    pipeline: Pipeline, num_cols: list[str], cat_cols: list[str]
) -> pd.DataFrame | None:
    """Достаёт feature importance / коэффициенты модели в едином формате."""
    try:
        pre = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]
        names = pre.get_feature_names_out()
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_).ravel()
        else:
            return None
        return (
            pd.DataFrame({"feature": names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as exc:
        logger.warning("Не удалось извлечь feature_importance: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(model: TrainedModel, path: Path | str = config.MODEL_PATH) -> Path:
    save_pickle(model, path)
    logger.info("Модель сохранена: %s", path)
    return Path(path)


def predict_proba(model: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    """Возвращает вероятность дефолта (класс 1) для нового среза."""
    cols = model.numeric_features + model.categorical_features
    missing = [c for c in cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    return model.pipeline.predict_proba(df[cols])[:, 1]
