"""Конфигурация проекта: пути, списки признаков, пороги EWS, параметры моделей."""
from __future__ import annotations

from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
LOGS_DIR: Path = ROOT_DIR / "logs"

RAW_DATA_PATH: Path = DATA_DIR / "3_5.xlsx"
MODEL_PATH: Path = MODELS_DIR / "model.pkl"
EWS_RULES_PATH: Path = MODELS_DIR / "ews_rules.yaml"
REFERENCE_STATS_PATH: Path = MODELS_DIR / "reference_stats.json"

for p in (MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)

ID_COLS: list[str] = ["credit_id", "borrower_id"]

DATE_COLS: list[str] = [
    "report_date_as_of",
    "date_borrower_birthday",
    "date_return",
    "date_issued",
    "date_signing",
    "last_payment_date",
]

CATEGORICAL_COLS: list[str] = [
    "sex",
    "loan_program",
    "bankruptcy_stage",
    "employer",
    "quality_category",
    "region_from_address",
]

DROP_COLS: list[str] = ["position", "attested_income", "education"]

NUMERIC_COLS: list[str] = [
    "initial_amount",
    "total_debt",
    "available_limit",
    "reserve_rate",
    "expired_actual_days_count",
    "percent_expired_actual_days_count",
    "dpd",
    "card_commission_expired_days",
    "limit_conversion_ratio",
    "total_debt_reserve",
    "available_limit_reserve",
    "pdn_initial",
    "pdn_current",
    "psk_current",
    "ifrs_provision_rate",
    "total_debt_provision",
    "available_limit_provision",
    "payment_sum_1m",
    "payment_sum_2m",
    "payment_sum_3m",
    "cnt_all_payments",
]

FLAG_COLS: list[str] = ["sold_flg", "forgiven_flg"]

TARGET_COL: str = "default_target"
FORWARD_TARGET_COL: str = "default_future"
FORWARD_HORIZON_MONTHS: int = 3
DEFAULT_DPD_THRESHOLD: int = 90
PDN_DEFAULT_THRESHOLD: float = 80.0
QUALITY_BAD_CATEGORIES: tuple[str, ...] = ("IV", "V", "4", "5")

EWS_ZONES = {
    "green": {"score_max": 0.15},
    "yellow": {"score_min": 0.15, "score_max": 0.40},
    "red": {"score_min": 0.40},
}

EWS_RULES: list[dict] = [
    {"name": "high_dpd", "expr": "dpd > 30", "weight": 2},
    {"name": "bad_quality", "expr": "quality_category.isin(['III','IV','V','3','4','5'])", "weight": 1},
    {"name": "pdn_spike", "expr": "pdn_current > 60", "weight": 1},
    {"name": "no_recent_payment", "expr": "days_since_last_payment > 60", "weight": 2},
    {"name": "payments_drop", "expr": "payment_ratio_mom < 0.3", "weight": 1},
    {"name": "utilization_high", "expr": "utilization > 0.9", "weight": 1},
    {"name": "bankruptcy_trigger", "expr": "is_bankrupt == 1", "weight": 3},
]

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
N_FOLDS: int = 5

LGBM_PARAMS: dict = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 600,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

DRIFT_PSI_THRESHOLD: float = 0.25
DRIFT_KS_PVALUE: float = 0.01
MODEL_METRIC_DROP_PCT: float = 0.10
