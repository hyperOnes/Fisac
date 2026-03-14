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
  echo "Create it with: python3 -m venv .venv"
  exit 1
fi

PY=".venv/bin/python"
PIP=".venv/bin/pip"
API_PORT="8000"
UI_PORT="5173"
REPLACE_RUNNING="${DEV_CHAT_REPLACE_RUNNING:-1}"
API_STARTUP_TIMEOUT="${DEV_CHAT_API_TIMEOUT_SECONDS:-45}"
UI_STARTUP_TIMEOUT="${DEV_CHAT_UI_TIMEOUT_SECONDS:-30}"

need_py_pkg() {
  local mod="$1"
  "$PY" - <<PY > /dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("$mod") else 1)
PY
}

if ! need_py_pkg fastapi || ! need_py_pkg uvicorn || ! need_py_pkg pydantic || ! need_py_pkg httpx; then
  echo "Installing backend deps (fastapi uvicorn pydantic httpx)..."
  "$PIP" install fastapi uvicorn pydantic httpx
fi

if [[ ! -d "chat_ui/node_modules" ]]; then
  echo "Installing frontend deps..."
  npm --prefix chat_ui install
fi

ensure_port_available() {
  local port="$1"
  local pids=""
  local remaining=""
  local pid=""

  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi

  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN -n -P 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  if [[ "$REPLACE_RUNNING" != "1" ]]; then
    echo "Port $port is already in use. Stop the existing process and retry."
    exit 1
  fi

  echo "Port $port is already in use. Stopping existing process(es): ${pids//$'\n'/ }"
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill "$pid" >/dev/null 2>&1 || true
  done <<< "$pids"
  sleep 1

  remaining="$(lsof -tiTCP:"$port" -sTCP:LISTEN -n -P 2>/dev/null || true)"
  if [[ -n "$remaining" ]]; then
    echo "Process(es) still using port $port, sending SIGKILL: ${remaining//$'\n'/ }"
    while IFS= read -r pid; do
      [[ -n "$pid" ]] || continue
      kill -9 "$pid" >/dev/null 2>&1 || true
    done <<< "$remaining"
    sleep 1
  fi

  if lsof -iTCP:"$port" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
    echo "Port $port is still in use after stopping existing process(es)."
    exit 1
  fi
}

ensure_port_available "$API_PORT"
ensure_port_available "$UI_PORT"

echo "Initializing chat DB..."
"$PY" - <<'PY'
from chat_api.config import get_settings
from chat_api.db import init_db

init_db(get_settings())
print("DB ready")
PY

echo "Starting Fisac Chat API on http://127.0.0.1:${API_PORT}"
"$PY" -m uvicorn chat_api.main:app --host 127.0.0.1 --port "$API_PORT" > /tmp/fisac-chat-api.log 2>&1 &
API_PID=$!
UI_PID=""

cleanup() {
  kill "$API_PID" >/dev/null 2>&1 || true
  if [[ -n "$UI_PID" ]]; then
    kill "$UI_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_http_ok() {
  local url="$1"
  local timeout="$2"
  local started
  started="$(date +%s)"
  while true; do
    if ! kill -0 "$API_PID" >/dev/null 2>&1; then
      echo "API process exited before readiness check passed."
      echo "--- /tmp/fisac-chat-api.log ---"
      tail -n 120 /tmp/fisac-chat-api.log || true
      return 1
    fi
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - started >= timeout )); then
      echo "Timed out waiting for $url"
      echo "--- /tmp/fisac-chat-api.log ---"
      tail -n 120 /tmp/fisac-chat-api.log || true
      return 1
    fi
    sleep 1
  done
}

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for readiness checks."
  exit 1
fi

echo "Waiting for API health..."
wait_for_http_ok "http://127.0.0.1:${API_PORT}/api/health" "$API_STARTUP_TIMEOUT"
echo "Waiting for API model status..."
wait_for_http_ok "http://127.0.0.1:${API_PORT}/api/model/status" "$API_STARTUP_TIMEOUT"
echo "Model status:"
curl -fsS "http://127.0.0.1:${API_PORT}/api/model/status" || true
echo

echo "Starting Fisac Chat UI on http://127.0.0.1:${UI_PORT}"
npm --prefix chat_ui run dev -- --host 127.0.0.1 --port "$UI_PORT" > /tmp/fisac-chat-ui.log 2>&1 &
UI_PID=$!

echo "Waiting for UI..."
started_ui="$(date +%s)"
while true; do
  if ! kill -0 "$UI_PID" >/dev/null 2>&1; then
    echo "UI process exited before readiness check passed."
    echo "--- /tmp/fisac-chat-ui.log ---"
    tail -n 120 /tmp/fisac-chat-ui.log || true
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${UI_PORT}" >/dev/null 2>&1; then
    break
  fi
  if (( "$(date +%s)" - started_ui >= UI_STARTUP_TIMEOUT )); then
    echo "Timed out waiting for UI readiness."
    echo "--- /tmp/fisac-chat-ui.log ---"
    tail -n 120 /tmp/fisac-chat-ui.log || true
    exit 1
  fi
  sleep 1
done

echo "Logs: /tmp/fisac-chat-api.log and /tmp/fisac-chat-ui.log"
echo "Open: http://127.0.0.1:${UI_PORT}"
echo "Press Ctrl+C to stop."

wait "$UI_PID"
