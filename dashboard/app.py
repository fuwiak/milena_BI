"""Streamlit BI-дашборд для мониторинга кредитных рисков (Главы 4.2-4.3 ВКР).

Пять экранов:
    1. Overview          — обзор портфеля и ключевые KPI
    2. Segments          — сегментация и срезы
    3. Early Warning     — топ-клиенты «красной зоны» с объяснениями
    4. Client Drilldown  — детализация по credit_id с временным профилем
    5. Drift / качество  — мониторинг drift при наличии новой выгрузки

Источник данных:
    * На проде (Railway, Streamlit Cloud, Docker) дашборд читает три parquet-файла
      из reports/ — они генерируются заранее через `python scripts/build_dashboard_cache.py`.
    * Если parquet отсутствуют (локальная разработка), автоматически делается fallback
      на исходный xlsx и полный пайплайн в памяти.

Локальный запуск:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.eda import aggregate_by, default_rate_over_time, portfolio_kpi  # noqa: E402
from src.segmentation import rule_based_segment, segment_profile  # noqa: E402
from src.utils import load_pickle  # noqa: E402

st.set_page_config(
    page_title="Кредитные риски — BI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PANEL_CACHE = config.REPORTS_DIR / "dashboard_panel.parquet"
SCORED_CACHE = config.REPORTS_DIR / "dashboard_scored.parquet"
TS_CACHE = config.REPORTS_DIR / "dashboard_timeseries.parquet"
CACHE_AVAILABLE = PANEL_CACHE.exists() and SCORED_CACHE.exists()


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Загружаю данные...", ttl=3600)
def load_panel() -> pd.DataFrame:
    if CACHE_AVAILABLE:
        return pd.read_parquet(PANEL_CACHE)
    # fallback для локальной разработки — полный пайплайн
    from src.data_loader import load_and_prepare
    from src.feature_engineering import build_feature_set
    from src.target import build_default_flag

    df = load_and_prepare(config.RAW_DATA_PATH)
    df = build_feature_set(df, include_rolling=False)
    df = build_default_flag(df)
    df["segment"] = rule_based_segment(df)
    return df


@st.cache_data(show_spinner="Загружаю срез со скорингом...", ttl=3600)
def load_scored() -> pd.DataFrame:
    if CACHE_AVAILABLE:
        return pd.read_parquet(SCORED_CACHE)
    # fallback — считаем скоринг в рантайме
    from src.ews import apply_rules, assign_zone, load_rules
    from src.model import predict_proba
    from src.recommendations import recommend_for_client

    df_full = load_panel()
    if "report_date_as_of" in df_full.columns:
        df_last = (df_full.sort_values("report_date_as_of")
                          .drop_duplicates(subset=["credit_id"], keep="last")
                          .reset_index(drop=True))
    else:
        df_last = df_full.copy()

    if config.MODEL_PATH.exists():
        model = load_pickle(config.MODEL_PATH)
        df_last["risk_score"] = predict_proba(model, df_last)
        try:
            rules, zones = load_rules()
        except Exception:
            rules, zones = config.EWS_RULES, config.EWS_ZONES
        rules_df = apply_rules(df_last, rules)
        df_last["rules_triggered"] = rules_df["rules_triggered"].values
        df_last["rules_weight_sum"] = rules_df["rules_weight_sum"].values
        df_last["zone"] = [assign_zone(s, zones) for s in df_last["risk_score"]]
        df_last.loc[df_last["rules_weight_sum"] >= 5, "zone"] = "red"
    else:
        df_last["risk_score"] = 0.0
        df_last["zone"] = "green"
        df_last["rules_triggered"] = [[] for _ in range(len(df_last))]
        df_last["rules_weight_sum"] = 0

    df_last["recommendations"] = df_last.apply(recommend_for_client, axis=1)
    df_last["priority"] = df_last["risk_score"] * df_last.get(
        "total_debt", pd.Series(0, index=df_last.index)
    ).clip(lower=0)
    return df_last


@st.cache_data(show_spinner=False, ttl=3600)
def load_timeseries() -> pd.DataFrame:
    if TS_CACHE.exists():
        return pd.read_parquet(TS_CACHE)
    df = load_panel()
    if "report_date_as_of" in df.columns:
        return default_rate_over_time(df)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model_artifact():
    if config.MODEL_PATH.exists():
        try:
            return load_pickle(config.MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Не удалось загрузить модель: {exc}")
            return None
    return None


df_full = load_panel()
df_scored = load_scored()
df_last = df_scored  # последний срез уже содержит всё нужное
df_ts = load_timeseries()


# ---------------------------------------------------------------------------
# Сайдбар
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Кредитные риски КК")
st.sidebar.caption(
    f"BI-решение для мониторинга кредитных рисков\n\n"
    f"Горизонт прогноза EWS: **{config.FORWARD_HORIZON_MONTHS} мес.**"
)
if not CACHE_AVAILABLE:
    st.sidebar.warning(
        "⚠️ Кэш parquet не найден — используется live-режим из xlsx.\n\n"
        "Для продакшена запустите:\n`python scripts/build_dashboard_cache.py`"
    )
else:
    st.sidebar.success("✅ Работаем из parquet-кэша")

view = st.sidebar.radio(
    "Экран",
    ["Обзор портфеля", "Сегменты", "Early Warning", "Карточка клиента", "Drift / качество модели"],
)

st.sidebar.divider()
st.sidebar.caption(
    f"Данные: {len(df_full):,} строк, {df_full['credit_id'].nunique():,} договоров".replace(",", " ")
)


# ---------------------------------------------------------------------------
# Экран 1 — Обзор
# ---------------------------------------------------------------------------
if view == "Обзор портфеля":
    st.title("Обзор портфеля кредитных карт")
    kpi = portfolio_kpi(df_last)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Договоров", f"{kpi.get('n_contracts', 0):,}".replace(",", " "))
    c2.metric("Клиентов", f"{kpi.get('n_borrowers', 0):,}".replace(",", " "))
    c3.metric("Общий долг, млн ₽", f"{kpi.get('total_debt', 0)/1e6:,.1f}".replace(",", " "))
    c4.metric("Доля дефолтов", f"{kpi.get('default_rate', 0)*100:.2f}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("Средний ПДН", f"{kpi.get('avg_pdn_current', 0):.1f}")
    c6.metric("Средний DPD", f"{kpi.get('avg_dpd', 0):.1f} дн.")
    c7.metric("Резервы, млн ₽", f"{kpi.get('reserve_amount', 0)/1e6:,.1f}".replace(",", " "))

    st.subheader("Динамика дефолтности и объёма задолженности")
    if not df_ts.empty:
        fig1 = px.line(df_ts, x="report_date_as_of", y="default_rate",
                       title="Доля дефолтов, %", markers=True)
        fig1.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig1, use_container_width=True)

        if "total_debt" in df_ts.columns:
            fig2 = px.area(df_ts, x="report_date_as_of", y="total_debt",
                           title="Суммарная задолженность, ₽")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Нет временнóго ряда для отображения.")

    st.subheader("Распределение по категориям качества")
    if "quality_category" in df_last.columns:
        qa = aggregate_by(df_last, "quality_category")
        st.dataframe(qa, use_container_width=True)


# ---------------------------------------------------------------------------
# Экран 2 — Сегменты
# ---------------------------------------------------------------------------
elif view == "Сегменты":
    st.title("Сегментация клиентов")

    seg_type = st.radio("Тип сегментации", ["Rule-based", "KMeans"], horizontal=True)
    df_seg = df_last.copy()

    if seg_type == "Rule-based":
        if "segment" not in df_seg.columns:
            df_seg["segment"] = rule_based_segment(df_seg)
    else:
        try:
            from src.segmentation import cluster_segment
            k = st.slider("K кластеров", 2, 8, 4)
            df_seg["segment"] = cluster_segment(df_seg, n_clusters=k).astype(str)
        except Exception as exc:
            st.error(f"KMeans недоступен: {exc}")
            df_seg["segment"] = rule_based_segment(df_seg)

    prof = segment_profile(df_seg, "segment")
    st.subheader("Профиль сегментов")
    st.dataframe(prof, use_container_width=True)

    if not prof.empty:
        fig = px.bar(prof, x="segment", y="n_contracts", color="default_rate",
                     title="Число договоров и дефолтность по сегментам",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    dim_choices = [c for c in ["region_from_address", "loan_program", "quality_category", "sex"]
                   if c in df_last.columns]
    if dim_choices:
        dim = st.selectbox("Срез по измерению", dim_choices)
        agg = aggregate_by(df_last, dim).head(20)
        st.dataframe(agg, use_container_width=True)
        fig2 = px.bar(agg, x=dim, y="default_rate", title=f"Доля дефолтов по {dim}")
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Экран 3 — EWS
# ---------------------------------------------------------------------------
elif view == "Early Warning":
    st.title("Система раннего предупреждения (EWS)")
    st.caption(
        f"Модель предсказывает вероятность выхода в дефолт в ближайшие "
        f"{config.FORWARD_HORIZON_MONTHS} месяцев. "
        f"Сочетается с бизнес-правилами для назначения зоны риска."
    )

    z_counts = (df_scored["zone"].value_counts()
                                 .reindex(["green", "yellow", "red"])
                                 .fillna(0).astype(int))
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Зелёная", int(z_counts.get("green", 0)))
    c2.metric("🟡 Жёлтая", int(z_counts.get("yellow", 0)))
    c3.metric("🔴 Красная", int(z_counts.get("red", 0)))

    zone_filter = st.multiselect("Зоны", ["green", "yellow", "red"], default=["yellow", "red"])
    top_n = st.slider("Сколько клиентов показать", 10, 500, 50, step=10)

    view_df = df_scored[df_scored["zone"].isin(zone_filter)].copy()
    if "priority" not in view_df.columns:
        view_df["priority"] = (view_df["risk_score"]
                               * view_df.get("total_debt", 0).clip(lower=0))
    view_df = view_df.sort_values("priority", ascending=False).head(top_n)

    cols_to_show = [c for c in [
        "credit_id", "borrower_id", "zone", "risk_score", "total_debt",
        "dpd", "pdn_current", "rules_triggered", "recommendations",
    ] if c in view_df.columns]

    def color_zone(v: str) -> str:
        return {"red": "background-color:#ffcccc",
                "yellow": "background-color:#fff4cc",
                "green": "background-color:#d9f5d9"}.get(v, "")

    st.dataframe(
        view_df[cols_to_show].style.map(color_zone, subset=["zone"]),
        use_container_width=True, height=500,
    )

    st.subheader("Распределение риск-скора")
    fig = px.histogram(df_scored, x="risk_score", nbins=60, color="zone",
                       color_discrete_map={"green": "#7fbf7f",
                                            "yellow": "#f0c040",
                                            "red": "#e06060"})
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Экран 4 — карточка клиента
# ---------------------------------------------------------------------------
elif view == "Карточка клиента":
    st.title("Карточка клиента (drilldown)")

    default_id = str(df_last["credit_id"].iloc[0])
    credit_id = st.text_input("credit_id", value=default_id)
    sub = df_full[df_full["credit_id"].astype(str) == str(credit_id)]
    if "report_date_as_of" in sub.columns:
        sub = sub.sort_values("report_date_as_of")

    if sub.empty:
        st.error("Договор не найден")
    else:
        last = sub.iloc[-1]
        scored_row = df_scored[df_scored["credit_id"].astype(str) == str(credit_id)].head(1)

        st.subheader("Сводка")
        c1, c2, c3, c4 = st.columns(4)
        if not scored_row.empty:
            c1.metric(
                f"P(дефолт в {config.FORWARD_HORIZON_MONTHS}м)",
                f"{float(scored_row['risk_score'].iloc[0]):.1%}",
            )
            c2.metric("Зона", str(scored_row["zone"].iloc[0]))
        c3.metric("DPD", int(last.get("dpd", 0) or 0))
        c4.metric("Задолженность", f"{float(last.get('total_debt', 0) or 0):,.0f} ₽")

        st.subheader("Временной профиль")
        metric_choices = [c for c in ["total_debt", "dpd", "pdn_current",
                                      "payment_sum_1m", "available_limit"]
                          if c in sub.columns]
        if metric_choices and "report_date_as_of" in sub.columns:
            metric = st.selectbox("Метрика", metric_choices)
            fig = px.line(sub, x="report_date_as_of", y=metric, markers=True,
                          title=f"{metric} — {credit_id}")
            st.plotly_chart(fig, use_container_width=True)

        def _as_list(value) -> list:
            """Безопасно приводит значение к list — работает для None / numpy-array /
            списков / строк (parquet round-trip превращает list → numpy.ndarray,
            а `array or []` падает на ValueError)."""
            if value is None:
                return []
            try:
                if hasattr(value, "tolist"):
                    value = value.tolist()
            except Exception:  # noqa: BLE001
                pass
            if isinstance(value, (list, tuple)):
                return [x for x in value if x is not None and str(x) != "nan"]
            return [value]

        if not scored_row.empty:
            st.subheader("Сработавшие правила EWS")
            rules_fired = _as_list(scored_row["rules_triggered"].iloc[0])
            if rules_fired:
                for r in rules_fired:
                    st.markdown(f"- `{r}`")
            else:
                st.info("Правила не сработали — клиент в зелёной зоне.")

            st.subheader("Рекомендации")
            recs = _as_list(scored_row["recommendations"].iloc[0])
            for r in recs:
                st.markdown(f"- {r}")

        st.subheader("Объяснение скоринга (SHAP)")
        model = load_model_artifact()
        if model is not None and not scored_row.empty:
            try:
                from src.interpretation import explain_one
                cols = list(model.numeric_features) + list(model.categorical_features)
                # scored_row хранит все признаки модели (см. build_dashboard_cache.py)
                missing = [c for c in cols if c not in scored_row.columns]
                if missing:
                    # fallback — добавляем NaN для отсутствующих колонок, imputer модели их заполнит
                    row_for_shap = scored_row.copy()
                    for c in missing:
                        row_for_shap[c] = pd.NA
                else:
                    row_for_shap = scored_row
                expl = explain_one(model, row_for_shap[cols].head(1), top_n=7)
                st.dataframe(pd.DataFrame(expl), use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.info(f"SHAP недоступен: {exc}")
        elif model is None:
            st.info("Модель ещё не обучена — запустите scripts/run_pipeline.py")


# ---------------------------------------------------------------------------
# Экран 5 — drift
# ---------------------------------------------------------------------------
else:
    st.title("Мониторинг качества модели и drift")
    st.caption(
        "Загрузите новый срез портфеля — дашборд посчитает PSI по скорам и признакам, "
        "KS-тест, а также ROC-AUC на новой выгрузке (если есть колонка target)."
    )

    uploaded = st.file_uploader("Новый срез (xlsx / parquet)", type=["xlsx", "parquet"])
    if uploaded is not None:
        import io
        try:
            if uploaded.name.endswith(".parquet"):
                new_df = pd.read_parquet(io.BytesIO(uploaded.getvalue()))
            else:
                new_df = pd.read_excel(io.BytesIO(uploaded.getvalue()))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Не удалось прочитать файл: {exc}")
            new_df = None

        model = load_model_artifact()
        if new_df is not None and model is None:
            st.error("Модель не обучена. Запустите scripts/run_pipeline.py.")
        elif new_df is not None:
            from src.drift_monitor import full_drift_report
            with st.spinner("Считаем drift..."):
                report = full_drift_report(
                    model, reference=df_last, current=new_df,
                    target_col=config.TARGET_COL,
                )
            c1, c2 = st.columns(2)
            c1.metric("PSI скоринга", f"{report.score_psi:.3f}")
            c2.metric("Нужно переобучение?", "ДА" if report.needs_retrain else "нет")
            st.dataframe(report.features, use_container_width=True)
            if report.model_metric_current:
                st.json(report.model_metric_current)
    else:
        st.info("Выгрузите новую витрину для сравнения с обучающим срезом.")
