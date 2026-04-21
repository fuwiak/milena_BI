"""Early Warning System — система раннего предупреждения (Глава 3.3 ВКР).

Комбинация:
    1. ML-скор риска дефолта (вероятность из модели);
    2. набор бизнес-правил с весами (см. config.EWS_RULES);
    3. дискретизация по зонам: green / yellow / red;
    4. приоритизация клиентов по произведению скора и объёма задолженности.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from . import config
from .model import TrainedModel, predict_proba
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class EWSResult:
    scored: pd.DataFrame  # credit_id, score, rule_hits, zone, priority
    summary: pd.DataFrame  # агрегат по зонам


# ---------------------------------------------------------------------------
# Rule engine — векторизованный (pandas Series в namespace eval)
# ---------------------------------------------------------------------------
def _vectorized_eval(expr: str, df: pd.DataFrame) -> np.ndarray:
    """Эвалюирует выражение в векторизованном режиме: все операции идут над pandas.Series
    целиком, а не построчно. Правила вида `col.isin([...])` тоже поддерживаются.

    Возвращает булев numpy-массив длины len(df). При ошибке — все False.
    """
    env: dict = {c: df[c] for c in df.columns}
    env["__builtins__"] = {}
    env["pd"] = pd
    env["np"] = np
    try:
        result = eval(expr, env)
    except Exception as exc:  # pragma: no cover
        logger.warning("Правило '%s' не применилось: %s", expr, exc)
        return np.zeros(len(df), dtype=bool)

    if isinstance(result, pd.Series):
        return result.fillna(False).astype(bool).to_numpy()
    if isinstance(result, np.ndarray):
        return result.astype(bool)
    # скалярный результат → раскатываем на всю длину
    return np.full(len(df), bool(result), dtype=bool)


def apply_rules(
    df: pd.DataFrame,
    rules: list[dict] | None = None,
) -> pd.DataFrame:
    """Применяет набор бизнес-правил к DataFrame в векторизованном режиме.

    Возвращает df с колонками rule_<name> (0/1), rules_weight_sum и rules_triggered (list[str]).
    """
    rules = rules or config.EWS_RULES
    out = df.copy()

    n = len(out)
    weight_sum = np.zeros(n, dtype=np.float32)
    # Матрица сработок: rows x n_rules, 0/1
    hit_matrix = np.zeros((n, len(rules)), dtype=np.int8)

    for j, rule in enumerate(rules):
        mask = _vectorized_eval(rule["expr"], out)
        col = f"rule_{rule['name']}"
        out[col] = mask.astype(np.int8)
        hit_matrix[:, j] = mask.astype(np.int8)
        weight_sum += mask.astype(np.float32) * float(rule.get("weight", 1))

    out["rules_weight_sum"] = weight_sum

    # rules_triggered — строим через numpy-маску без apply(axis=1),
    # имена кастим к обычному str (np.str_ в CSV выглядит как "np.str_('...')")
    rule_names = [r["name"] for r in rules]
    rule_arr = np.array(rule_names, dtype=object)
    out["rules_triggered"] = [[str(x) for x in rule_arr[row == 1]] for row in hit_matrix]
    return out


# ---------------------------------------------------------------------------
# ML-scoring + зоны
# ---------------------------------------------------------------------------
def assign_zone(score: float, zones: dict = None) -> str:
    z = zones or config.EWS_ZONES
    if score < z["green"]["score_max"]:
        return "green"
    if score < z["yellow"]["score_max"]:
        return "yellow"
    return "red"


def build_ews(
    df: pd.DataFrame,
    model: TrainedModel,
    rules: list[dict] | None = None,
    zones: dict | None = None,
) -> EWSResult:
    """Для каждого договора считает ML-скор, применяет правила и назначает зону."""
    scores = predict_proba(model, df)

    scored = pd.DataFrame({
        "credit_id": df["credit_id"].values if "credit_id" in df.columns else np.arange(len(df)),
        "borrower_id": df.get("borrower_id", pd.Series(index=df.index)).values,
        "report_date_as_of": df.get("report_date_as_of", pd.NaT).values if "report_date_as_of" in df.columns else pd.NaT,
        "risk_score": scores,
        "total_debt": df.get("total_debt", pd.Series(0, index=df.index)).values,
    })
    scored["zone"] = scored["risk_score"].apply(lambda s: assign_zone(s, zones))

    rules_df = apply_rules(df, rules)
    scored["rules_triggered"] = rules_df["rules_triggered"].values
    scored["rules_weight_sum"] = rules_df["rules_weight_sum"].values

    # правила могут эскалировать зону
    escalate = scored["rules_weight_sum"] >= 3
    scored.loc[escalate & (scored["zone"] == "green"), "zone"] = "yellow"
    scored.loc[scored["rules_weight_sum"] >= 5, "zone"] = "red"

    scored["priority"] = scored["risk_score"] * scored["total_debt"].clip(lower=0)
    scored = scored.sort_values("priority", ascending=False).reset_index(drop=True)

    summary = (
        scored.groupby("zone")
        .agg(n_contracts=("credit_id", "nunique"),
             sum_debt=("total_debt", "sum"),
             avg_score=("risk_score", "mean"))
        .reindex(["green", "yellow", "red"])
        .reset_index()
    )
    logger.info("EWS: green=%d, yellow=%d, red=%d",
                int(summary.loc[summary["zone"] == "green", "n_contracts"].fillna(0).iloc[0]),
                int(summary.loc[summary["zone"] == "yellow", "n_contracts"].fillna(0).iloc[0]),
                int(summary.loc[summary["zone"] == "red", "n_contracts"].fillna(0).iloc[0]))
    return EWSResult(scored=scored, summary=summary)


# ---------------------------------------------------------------------------
# Хранение бизнес-правил в YAML
# ---------------------------------------------------------------------------
def save_rules(rules: list[dict], path: Path | str = config.EWS_RULES_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"rules": rules, "zones": config.EWS_ZONES}, f, allow_unicode=True,
                       sort_keys=False)
    logger.info("EWS rules сохранены: %s", path)
    return path


def load_rules(path: Path | str = config.EWS_RULES_PATH) -> tuple[list[dict], dict]:
    with open(Path(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("rules", []), data.get("zones", config.EWS_ZONES)
