"""FastAPI-сервис для расчёта риска дефолта и отдачи данных BI-дашборду (Глава 4.1 ВКР).

Эндпоинты:
    GET  /health                       — healthcheck;
    POST /score                        — скоринг по одному договору (JSON-пейлоад признаков);
    POST /score/batch                  — скоринг по массиву договоров;
    POST /ews                          — ML-скор + бизнес-правила → зона + рекомендации;
    GET  /portfolio/kpi                — ключевые KPI портфеля (для overview-плитки);
    GET  /portfolio/by/{dim}           — агрегат по регионам/программам/qualities;
    GET  /portfolio/timeseries         — динамика дефолтности и задолженности;
    GET  /client/{credit_id}           — карточка клиента с историей показателей;
    GET  /client/{credit_id}/explain   — top SHAP-причины высокого риска.

Аутентификация (опционально) — простейший API-Key через заголовок `X-API-Key`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.data_loader import load_and_prepare
from src.eda import aggregate_by, default_rate_over_time, portfolio_kpi
from src.ews import apply_rules, assign_zone, load_rules
from src.feature_engineering import build_feature_set
from src.model import TrainedModel, predict_proba
from src.recommendations import recommend_for_client
from src.target import build_default_flag
from src.utils import configure_logging, get_logger, load_pickle

configure_logging(app="api")
logger = get_logger("api")

API_KEY = os.getenv("MILENA_API_KEY")

app = FastAPI(
    title="Milena BI — Credit Card Risk Service",
    version="0.1.0",
    description="Сервис прогнозирования дефолта и EWS для портфеля кредитных карт.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def get_model() -> TrainedModel:
    if not config.MODEL_PATH.exists():
        raise RuntimeError(
            f"Модель не найдена: {config.MODEL_PATH}. Запустите scripts/run_pipeline.py."
        )
    return load_pickle(config.MODEL_PATH)


@lru_cache(maxsize=1)
def get_portfolio() -> pd.DataFrame:
    df = load_and_prepare(config.RAW_DATA_PATH)
    df = build_feature_set(df, include_rolling=False)
    df = build_default_flag(df)
    return df


@lru_cache(maxsize=1)
def get_last_slice() -> pd.DataFrame:
    df = get_portfolio()
    if "report_date_as_of" in df.columns:
        df = df.sort_values("report_date_as_of")
        df = df.drop_duplicates(subset=["credit_id"], keep="last")
    return df


def _check_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


class ClientFeatures(BaseModel):
    """Минимальный набор признаков для online-скоринга."""
    credit_id: Optional[str] = None
    total_debt: float = 0
    available_limit: float = 0
    initial_amount: float = 0
    dpd: float = 0
    pdn_current: float = 0
    pdn_initial: float = 0
    psk_current: float = 0
    payment_sum_1m: float = 0
    payment_sum_2m: float = 0
    payment_sum_3m: float = 0
    cnt_all_payments: int = 0
    limit_conversion_ratio: float = 0
    reserve_rate: float = 0
    quality_category: str = "UNKNOWN"
    loan_program: str = "UNKNOWN"
    region_from_address: str = "UNKNOWN"
    sex: str = "UNKNOWN"
    bankruptcy_stage: Optional[str] = None
    age: Optional[float] = None
    months_on_book: Optional[float] = None
    days_since_last_payment: Optional[float] = None
    utilization: Optional[float] = None
    payment_ratio_1m: Optional[float] = None
    payment_ratio_mom: Optional[float] = None
    is_bankrupt: Optional[int] = 0

    class Config:
        extra = "allow"


class ScoreResponse(BaseModel):
    credit_id: Optional[str]
    risk_score: float
    zone: str
    rules_triggered: list[str]
    recommendations: list[str]


class BatchRequest(BaseModel):
    records: list[ClientFeatures] = Field(..., min_length=1)


@app.get("/health")
def health() -> dict:
    try:
        _ = get_model()
        model_loaded = True
    except Exception:
        model_loaded = False
    return {"status": "ok", "model_loaded": model_loaded}


def _score_df(df: pd.DataFrame) -> pd.DataFrame:
    model = get_model()
    rules, zones = (config.EWS_RULES, config.EWS_ZONES)
    try:
        rules, zones = load_rules()
    except Exception:
        pass

    df = build_feature_set(df, include_rolling=False)
    scores = predict_proba(model, df)
    rules_df = apply_rules(df, rules)
    out = df.copy()
    out["risk_score"] = scores
    out["zone"] = [assign_zone(s, zones) for s in scores]
    out["rules_triggered"] = rules_df["rules_triggered"].values
    out["rules_weight_sum"] = rules_df["rules_weight_sum"].values
    out.loc[out["rules_weight_sum"] >= 5, "zone"] = "red"
    out["recommendations"] = out.apply(recommend_for_client, axis=1)
    return out


@app.post("/score", response_model=ScoreResponse)
def score_one(payload: ClientFeatures, x_api_key: Optional[str] = Header(None)) -> Any:
    _check_api_key(x_api_key)
    df = pd.DataFrame([payload.model_dump()])
    out = _score_df(df).iloc[0]
    return ScoreResponse(
        credit_id=str(out.get("credit_id")) if pd.notna(out.get("credit_id")) else None,
        risk_score=float(out["risk_score"]),
        zone=out["zone"],
        rules_triggered=list(out["rules_triggered"]),
        recommendations=list(out["recommendations"]),
    )


@app.post("/score/batch")
def score_batch(payload: BatchRequest, x_api_key: Optional[str] = Header(None)) -> list[dict]:
    _check_api_key(x_api_key)
    df = pd.DataFrame([r.model_dump() for r in payload.records])
    out = _score_df(df)
    return out[["credit_id", "risk_score", "zone", "rules_triggered", "recommendations"]].to_dict(
        orient="records"
    )


@app.post("/ews")
def run_ews(x_api_key: Optional[str] = Header(None),
            zone: Optional[str] = Query(None, pattern="^(green|yellow|red)$"),
            limit: int = Query(100, ge=1, le=10_000)) -> list[dict]:
    _check_api_key(x_api_key)
    df = get_last_slice()
    scored = _score_df(df)
    scored["priority"] = scored["risk_score"] * scored.get("total_debt", 0).clip(lower=0)
    scored = scored.sort_values("priority", ascending=False)
    if zone:
        scored = scored[scored["zone"] == zone]
    cols = ["credit_id", "borrower_id", "risk_score", "zone",
            "rules_triggered", "recommendations", "total_debt", "dpd", "pdn_current"]
    return scored[[c for c in cols if c in scored.columns]].head(limit).to_dict(orient="records")


@app.get("/portfolio/kpi")
def get_kpi() -> dict:
    df = get_last_slice()
    return portfolio_kpi(df)


@app.get("/portfolio/by/{dim}")
def get_agg(dim: str) -> list[dict]:
    df = get_last_slice()
    if dim not in df.columns:
        raise HTTPException(status_code=400, detail=f"unknown dimension {dim}")
    return aggregate_by(df, dim).head(100).to_dict(orient="records")


@app.get("/portfolio/timeseries")
def get_timeseries() -> list[dict]:
    df = get_portfolio()
    return default_rate_over_time(df).to_dict(orient="records")


@app.get("/client/{credit_id}")
def get_client(credit_id: str) -> dict:
    df = get_portfolio()
    sub = df[df["credit_id"].astype(str) == str(credit_id)]
    if sub.empty:
        raise HTTPException(status_code=404, detail="credit_id не найден")
    history = sub.sort_values("report_date_as_of")
    last = history.iloc[-1].to_dict()
    scored = _score_df(history.tail(1)).iloc[0]
    return {
        "credit_id": credit_id,
        "latest_snapshot": {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
                            for k, v in last.items() if not isinstance(v, (list, dict))},
        "risk_score": float(scored["risk_score"]),
        "zone": scored["zone"],
        "rules_triggered": list(scored["rules_triggered"]),
        "recommendations": list(scored["recommendations"]),
        "history": history[["report_date_as_of", "total_debt", "dpd", "pdn_current",
                            "payment_sum_1m"]].astype(str).to_dict(orient="records"),
    }


@app.get("/client/{credit_id}/explain")
def explain_client(credit_id: str, top_n: int = Query(5, ge=1, le=20)) -> list[dict]:
    from src.interpretation import explain_one
    df = get_last_slice()
    sub = df[df["credit_id"].astype(str) == str(credit_id)]
    if sub.empty:
        raise HTTPException(status_code=404, detail="credit_id не найден")
    model = get_model()
    cols = model.numeric_features + model.categorical_features
    return explain_one(model, sub[cols].head(1), top_n=top_n)
