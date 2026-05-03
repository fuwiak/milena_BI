#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Нет интерпретатора: $PY" >&2
  echo "Создайте окружение в корне репозитория:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
cd "$ROOT"
exec "$PY" "$@"
