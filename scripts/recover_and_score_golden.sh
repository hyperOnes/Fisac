#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

echo "[1/3] Recovering candidate bundles..."
"$PY" tools/recover_golden_window.py "$@"

echo "[2/3] Replaying canonical prompts + scoring candidates..."
"$PY" tools/replay_golden_window.py

echo "[3/3] Golden recovery complete."
if [[ -f artifacts/golden_recovery/golden_baseline_manifest.json ]]; then
  echo "Best candidate summary:"
  cat artifacts/golden_recovery/golden_baseline_manifest.json
else
  echo "No best candidate manifest generated."
fi
