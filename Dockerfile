# syntax=docker/dockerfile:1.6
# =============================================================================
# Dockerfile для Streamlit-дашборда мониторинга кредитных рисков.
# Ориентирован на Railway, но совместим с любым контейнерным хостингом
# (Render, Fly.io, GCP Cloud Run, локальный Docker).
# =============================================================================

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Системные зависимости:
#   libgomp1  — OpenMP-рантайм для LightGBM (без него import lightgbm падает)
#   curl      — healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ставим зависимости отдельным слоем — Docker закеширует если requirements.txt не менялся
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Копируем только то, что нужно для рантайма (см. .dockerignore)
COPY src/        ./src/
COPY dashboard/  ./dashboard/
COPY scripts/    ./scripts/
COPY models/     ./models/
COPY reports/    ./reports/
COPY .streamlit/ ./.streamlit/

# Railway/Render пробрасывают PORT через env. По умолчанию 8501 для локального docker run.
ENV PORT=8501
EXPOSE 8501

# Healthcheck для оркестратора
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/_stcore/health || exit 1

# sh -c чтобы $PORT развернулся в момент старта контейнера
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.port=${PORT} --server.address=0.0.0.0"]
