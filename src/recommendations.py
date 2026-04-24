"""Автоматическая генерация рекомендаций по снижению дефолтности (Глава 3.5 ВКР).

Правила следуют из результатов модели и EWS:
    * клиенты 'red' + большой unused limit → снизить лимит;
    * 'yellow' + высокий ПДН → предложить реструктуризацию / снижение ставки;
    * 'red' + активные платежи → персональная коммуникация / льготный период;
    * 'red' + нет платежей > 60 дней → приоритет коллекшн-подразделения;
    * сегменты с ростом дефолтности → ужесточить скоринговый cut-off.
"""
from __future__ import annotations

import pandas as pd

from .ews import EWSResult
from .utils import get_logger

logger = get_logger(__name__)


def recommend_for_client(row: pd.Series) -> list[str]:
    """Возвращает список рекомендаций по одному клиенту (строка с EWS-результатом)."""
    recs: list[str] = []
    zone = row.get("zone", "green")
    rules = set(str(r) for r in (row.get("rules_triggered", []) or []))

    if zone == "red":
        if "no_recent_payment" in rules or "high_dpd" in rules or "bankruptcy_trigger" in rules:
            recs.append("Передать в коллекшн с высоким приоритетом")
        else:
            recs.append("Персональная коммуникация: предложить реструктуризацию")
        recs.append("Заблокировать/снизить кредитный лимит")
    elif zone == "yellow":
        if "utilization_high" in rules:
            recs.append("Снизить доступный лимит по карте")
        if "pdn_spike" in rules:
            recs.append("Предложить рефинансирование с меньшей ставкой")
        if "payments_drop" in rules:
            recs.append("Отправить напоминание о платеже + льготные условия")
        if not recs:
            recs.append("Мониторинг: включить в еженедельный watch-list")
    else:
        recs.append("Оставить текущие условия; возможен upsell / повышение лимита")
    return recs


def build_recommendations_table(ews: EWSResult) -> pd.DataFrame:
    """Добавляет поле recommendations к скорингу EWS."""
    df = ews.scored.copy()
    df["recommendations"] = df.apply(recommend_for_client, axis=1)
    return df


def segment_policy(
    segment_summary: pd.DataFrame,
    default_rate_col: str = "default_rate",
    threshold_high: float = 0.10,
    threshold_low: float = 0.02,
) -> pd.DataFrame:
    """На уровне сегментов предлагает корректировку cut-off'ов и политик."""
    df = segment_summary.copy()
    def rule(r):
        if r[default_rate_col] > threshold_high:
            return "ужесточить cut-off, усилить верификацию, сократить средний лимит"
        if r[default_rate_col] < threshold_low:
            return "смягчить cut-off; возможен кросс-продукт и upsell"
        return "оставить текущую политику; расширенный мониторинг"
    df["policy_recommendation"] = df.apply(rule, axis=1)
    return df
