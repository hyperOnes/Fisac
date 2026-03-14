#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "chat_api/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "chat_api/.env.local"
  set +a
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv at $ROOT/.venv"
  exit 1
fi

PY=".venv/bin/python"
PIP=".venv/bin/pip"

need_py_pkg() {
  local mod="$1"
  "$PY" - <<PY > /dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("$mod") else 1)
PY
}

if ! need_py_pkg fastapi || ! need_py_pkg pydantic || ! need_py_pkg httpx || ! need_py_pkg uvicorn; then
  echo "Installing backend test dependencies (fastapi pydantic httpx uvicorn)..."
  if ! "$PIP" install fastapi pydantic httpx uvicorn; then
    echo "Dependency install failed; integration tests may be skipped."
  fi
fi

echo "Running backend unit + integration tests (non-slow)..."
"$PY" -m pytest -q -m "not slow" tests

if [[ -d "chat_ui/node_modules" ]]; then
  echo "Running frontend tests..."
  npm --prefix chat_ui test -- --run
else
  echo "Skipping frontend tests (chat_ui/node_modules missing). Run: npm --prefix chat_ui install"
fi

echo "Running slow chat validations (soak + latency)..."
"$PY" -m pytest -q -m slow tests/test_chat_bridge.py
