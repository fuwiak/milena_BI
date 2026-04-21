# BI-решение для мониторинга кредитных рисков и выявления ранних сигналов дефолта

ВКР: разработка BI-решения для портфеля кредитных карт банка на основе методов машинного
обучения и системы раннего предупреждения (Early Warning System).

## Структура проекта

```
milena_BI/
├── data/                      # Исходные данные (3_5.xlsx) и справочники
├── src/                       # Основной Python-пакет
│   ├── config.py              # Пути, списки признаков, константы
│   ├── data_loader.py         # Загрузка и первичная очистка данных
│   ├── eda.py                 # EDA, распределения, временные ряды (гл. 2.1, 2.3)
│   ├── segmentation.py        # Rule-based и кластерная сегментация (гл. 2.2)
│   ├── correlation.py         # Корреляционный и факторный анализ (гл. 2.4)
│   ├── feature_engineering.py # Признаки для ML (гл. 3.1)
│   ├── target.py              # Построение целевой переменной (дефолт)
│   ├── model.py               # Обучение baseline и advanced моделей (гл. 3.1-3.2)
│   ├── metrics.py             # ROC-AUC, PR-AUC, Gini, KS
│   ├── interpretation.py      # SHAP, permutation importance, PDP
│   ├── ews.py                 # Early Warning System (гл. 3.3)
│   ├── drift_monitor.py       # Data / model drift (гл. 3.4)
│   ├── recommendations.py     # Автогенерация рекомендаций (гл. 3.5)
│   └── utils.py
├── api/
│   └── main.py                # FastAPI-сервис (гл. 4.1)
├── dashboard/
│   └── app.py                 # Streamlit BI-дашборд (гл. 4.2, 4.3)
├── scripts/
│   ├── run_pipeline.py        # Полный ETL + обучение + EWS
│   ├── run_inference.py       # Батч-скоринг на новых данных
│   └── run_drift_check.py     # Мониторинг дрейфа
├── models/                    # Сериализованные модели (.pkl)
├── reports/                   # Графики, отчёты по EDA и drift
└── notebooks/                 # Экспериментальные ноутбуки
```

## Колонки датасета (40 признаков)

| # | Поле | Описание |
|---|------|----------|
| 1 | report_date_as_of | дата среза витрины |
| 2 | credit_id | идентификатор договора кредитной карты |
| 3 | borrower_id | идентификатор клиента |
| 4 | sex | пол |
| 5 | position | должность (drop per EDA) |
| 6 | attested_income | подтверждённый доход (drop per EDA) |
| 7 | education | уровень образования (drop per EDA) |
| 8 | date_borrower_birthday | дата рождения клиента |
| 9 | loan_program | программа кредитования |
| 10 | bankruptcy_stage | стадия банкротства |
| 11 | employer | работодатель |
| 12 | initial_amount | начальный лимит |
| 13 | date_return | плановая дата возврата |
| 14 | date_issued | дата выдачи |
| 15 | date_signing | дата подписания договора |
| 16 | total_debt | общая задолженность |
| 17 | available_limit | доступный лимит |
| 18 | reserve_rate | ставка резервирования |
| 19 | quality_category | категория качества ссуды |
| 20 | expired_actual_days_count | кол-во фактических дней просрочки |
| 21 | percent_expired_actual_days_count | доля дней просрочки |
| 22 | dpd | days past due |
| 23 | card_commission_expired_days | дни просрочки комиссии |
| 24 | limit_conversion_ratio | коэффициент использования лимита |
| 25 | total_debt_reserve | резерв под общий долг |
| 26 | available_limit_reserve | резерв под доступный лимит |
| 27 | pdn_initial | ПДН на выдаче |
| 28 | pdn_current | текущий ПДН |
| 29 | psk_current | текущая ПСК |
| 30 | region_from_address | регион |
| 31 | ifrs_provision_rate | ставка резерва МСФО |
| 32 | total_debt_provision | резерв МСФО общего долга |
| 33 | available_limit_provision | резерв МСФО доступного лимита |
| 34 | payment_sum_1m | сумма платежей за 1м |
| 35 | payment_sum_2m | сумма платежей за 2м |
| 36 | payment_sum_3m | сумма платежей за 3м |
| 37 | cnt_all_payments | кол-во платежей |
| 38 | last_payment_date | дата последнего платежа |
| 39 | sold_flg | флаг продажи долга |
| 40 | forgiven_flg | флаг прощения долга |

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Постановка задачи ML

По умолчанию модель обучается на **forward-target** — предсказание факта выхода договора
в дефолт в ближайшие `FORWARD_HORIZON_MONTHS = 3` месяца
(конфигурируется в `src/config.py`).

На этапе training-а для каждой (credit_id, report_date_as_of) строится метка:

    default_future = 1, если хотя бы в одном из следующих 3 месяцев
                     default_target (dpd>90 | pdn>80 | quality∈{IV,V} | bankruptcy) = 1

Строки без будущих наблюдений отбрасываются (`has_future_obs=False`), чтобы не
обучаться на «ложных нулях». Inference и EWS выполняются на последнем срезе — модель
выдаёт `P(default через 3м)`, которая и попадает в карточку клиента и дашборд.

Альтернативный режим `--target current` оставляет исходную задачу «классифицировать
текущее состояние дефолта» (полезно для сравнения).

## Запуск

```bash
# 1. Полный пайплайн: EDA → features → обучение → EWS → сохранение моделей
python scripts/run_pipeline.py --data data/3_5.xlsx

# 2. FastAPI сервис (для онлайн-скоринга)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Streamlit BI-дашборд
streamlit run dashboard/app.py

# 4. Проверка drift на новой выгрузке
python scripts/run_drift_check.py --new data/new_slice.xlsx
```

## Ключевые артефакты

- `models/model.pkl` — обученная модель (LightGBM) + пайплайн препроцессинга
- `models/ews_rules.yaml` — бизнес-правила EWS
- `reports/eda.html` — отчёт EDA
- `reports/drift_report.html` — evidently drift-отчёт

## Соответствие плану ВКР

| Глава | Файлы |
|-------|-------|
| 2.1 — описание данных | `src/eda.py`, `dashboard/app.py` |
| 2.2 — сегментация | `src/segmentation.py` |
| 2.3 — динамика дефолтности | `src/eda.py`, `src/target.py` |
| 2.4 — корреляции и отбор | `src/correlation.py` |
| 3.1 — baseline | `src/model.py`, `src/feature_engineering.py` |
| 3.2 — метрики, SHAP | `src/metrics.py`, `src/interpretation.py` |
| 3.3 — EWS | `src/ews.py` |
| 3.4 — drift | `src/drift_monitor.py` |
| 3.5 — рекомендации | `src/recommendations.py` |
| 4.1 — архитектура, API | `api/main.py` |
| 4.2-4.3 — BI-дашборд | `dashboard/app.py` |
