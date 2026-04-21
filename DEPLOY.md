# Деплой Streamlit-дашборда на Railway через Dockerfile

Это руководство описывает, как опубликовать BI-дашборд `dashboard/app.py`
на [Railway](https://railway.app/) с использованием **Dockerfile**
(Procfile/Nixpacks не используются).

---

## Архитектура для деплоя

Дашборд на проде **не читает** исходный XLSX (107 МБ — слишком много для git).
Вместо этого он использует три компактных parquet-кэша:

| Файл | Размер | Что внутри |
|------|--------|-----------|
| `reports/dashboard_panel.parquet`       | ~10 МБ | панельные данные для drilldown (credit_id × дата × 19 колонок) |
| `reports/dashboard_scored.parquet`      | ~1 МБ  | последний срез c `risk_score`, `zone`, `rules_triggered`, `recommendations` |
| `reports/dashboard_timeseries.parquet`  | ~10 КБ | агрегаты по датам для графиков динамики |

Эти файлы генерируются **локально** один раз перед пушем в git:

```bash
python scripts/build_dashboard_cache.py
```

Все три parquet-файла коммитятся в репозиторий — Docker-образ собирается из них.

---

## Файлы инфраструктуры

| Файл | Назначение |
|------|-----------|
| `Dockerfile`       | Рецепт сборки образа: python:3.11-slim + libgomp1 (для LightGBM) + зависимости + код + healthcheck |
| `.dockerignore`    | Исключает `.venv/`, `data/`, `logs/`, `*.xlsx`, `.git/` — в образ не попадёт ничего лишнего |
| `railway.json`     | Говорит Railway использовать `DOCKERFILE` builder и `/_stcore/health` для healthcheck |
| `.streamlit/config.toml` | Тема, headless-режим, лимит upload |
| `requirements.txt` | Runtime-зависимости (pandas, lightgbm, streamlit, pyarrow, sklearn, plotly) |

---

## Шаг 1. Локальная проверка Docker-образа

```bash
# 1. Собираем parquet-кэш дашборда (один раз)
python scripts/build_dashboard_cache.py

# 2. Собираем образ
docker build -t milena-bi .

# 3. Запускаем локально (имитируем Railway — порт приходит через $PORT)
docker run --rm -p 8080:8080 -e PORT=8080 milena-bi

# 4. В браузере http://localhost:8080
#    В сайдбаре должно быть "✅ Работаем из parquet-кэша"
#    Healthcheck: curl http://localhost:8080/_stcore/health → 200 OK
```

### Проверка размера образа

```bash
docker images milena-bi
# Ожидаемо ~600-800 МБ (python:3.11-slim + ML-стек).
```

---

## Шаг 2. Подготовка git-репозитория

```bash
# 1. Убеждаемся что parquet-кэш собран
ls -lh reports/dashboard_*.parquet   # ~11 МБ суммарно

# 2. Инициализируем git
git init
git add .
git status       # убедитесь: data/3_5.xlsx и data/3_5.pkl НЕ в коммите
git commit -m "Initial commit: BI dashboard ready for Railway (Docker)"

# 3. Пушим на GitHub
git remote add origin https://github.com/<user>/milena-bi.git
git branch -M main
git push -u origin main
```

### Что должно попасть в репозиторий
- [x] `Dockerfile`, `.dockerignore`, `railway.json`
- [x] `dashboard/`, `src/`, `scripts/`
- [x] `models/model.pkl` (~4 МБ), `models/ews_rules.yaml`
- [x] `reports/dashboard_*.parquet` (~11 МБ)
- [x] `reports/*.json`, `reports/feature_importance.csv`, `reports/interpretation.txt`
- [x] `.streamlit/config.toml`
- [x] `requirements.txt`, `requirements-dev.txt`

### Что **не** должно попасть (проверьте `.gitignore` и `.dockerignore`)
- [ ] `data/*.xlsx`, `data/*.pkl` — исходные данные (≫100 МБ)
- [ ] `.venv/` — виртуалка
- [ ] `logs/*.log`
- [ ] `reports/eda.html`

---

## Шаг 3. Деплой на Railway

1. [railway.app](https://railway.app/) → **New Project** → **Deploy from GitHub repo**.
2. Выбираем репозиторий `milena-bi`. Railway автоматически:
   - находит `Dockerfile` в корне,
   - видит `railway.json` с `"builder": "DOCKERFILE"`,
   - собирает образ (первая сборка ~3-5 минут, потом кэшируется),
   - запускает контейнер, пробрасывая `$PORT` из env.
3. **Settings → Networking → Generate Domain** — получите публичный URL
   вида `https://<project>.up.railway.app`.

### Healthcheck
Railway пингует `/_stcore/health` (стандартный endpoint Streamlit).
Также в самом `Dockerfile` есть `HEALTHCHECK`, который работает и в локальном
docker (см. `docker inspect --format='{{json .State.Health}}' <container>`).

---

## Шаг 4. Переменные окружения (опционально)

Railway сам задаёт `PORT`. Дополнительно полезны:

| Переменная | Значение | Зачем |
|------------|----------|-------|
| `PYTHONUNBUFFERED` | `1` | Уже задано в Dockerfile — логи Streamlit пишутся сразу |
| `STREAMLIT_SERVER_FILE_WATCHER_TYPE` | `none` | Уже задано в Dockerfile — меньше I/O на проде |

---

## Обновление прод-данных

Когда приходит новая выгрузка (`data/3_5.xlsx` обновился):

```bash
# 1. Пересобираем модель и EWS
python scripts/run_pipeline.py

# 2. Обновляем parquet-кэш дашборда
python scripts/build_dashboard_cache.py

# 3. Коммитим и пушим — Railway автоматически соберёт новый образ
git add models/model.pkl reports/
git commit -m "Refresh: данные на $(date +%Y-%m-%d)"
git push
```

---

## Деплой на другие контейнерные хостинги

Тот же `Dockerfile` без изменений работает на:

### Render
Dashboard → **New Web Service** → подключить GitHub → **Runtime: Docker**.
Healthcheck path: `/_stcore/health`.

### Fly.io
```bash
flyctl launch --dockerfile Dockerfile --name milena-bi --region fra
flyctl deploy
```

### Google Cloud Run
```bash
gcloud run deploy milena-bi --source . --region europe-west1 \
    --allow-unauthenticated --port 8080
```

### Локальный production-стиль запуск
```bash
docker build -t milena-bi .
docker run -d --name milena-bi -p 8501:8501 --restart unless-stopped milena-bi
```

---

## Частые проблемы

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `libgomp.so.1: cannot open shared object file` | Нет OpenMP для LightGBM | Dockerfile уже ставит `libgomp1` через apt-get |
| `FileNotFoundError: reports/dashboard_scored.parquet` | Забыли собрать parquet-кэш | `python scripts/build_dashboard_cache.py` + коммит |
| `Error: Cannot find package 'pyarrow'` | pyarrow не в requirements | Уже есть: `pyarrow==17.0.0` |
| Контейнер убивается с OOM | Free tier Railway — 512 МБ RAM | Используйте parquet-кэш (fallback на xlsx требует >1 ГБ) |
| `docker build` очень долгий | Первая сборка без кэша | Последующие билды инкрементальны благодаря слоистому `COPY requirements.txt` |
| `Port already in use` локально | Streamlit уже висит | `lsof -i :8501` → `kill <pid>` |

---

**Готово!** После пуша в `main` дашборд будет доступен по публичному URL Railway через 3-5 минут (первая сборка Docker-образа).
