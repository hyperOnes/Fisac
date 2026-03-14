#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv. Create with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

RUN_TRAINING=1
if [[ "${1:-}" == "--skip-training" ]]; then
  RUN_TRAINING=0
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="benchmark_runs/genius_pipeline_${TS}"
mkdir -p "$OUT_DIR"

BASELINE_RAW="${FISAC_BASELINE_ROBUST_RAW:-/Users/sebastian/Fisac/benchmark_runs/robust_5seed.json}"
BASELINE_CHAT="${FISAC_BASELINE_CHAT_BENCH:-/Users/sebastian/Fisac/benchmark_runs/chat_reasoning_benchmark_baseline.json}"
JURY_HEAD="$OUT_DIR/jury_head.pt"
SYNTH_HEAD="$OUT_DIR/synthesis_head.json"

echo "[1/7] Running tests..."
PYTHONPATH="$ROOT" "$PY" -m pytest -q -m "not slow" tests | tee "$OUT_DIR/pytest.log"

if [[ "$RUN_TRAINING" -eq 1 ]]; then
  echo "[2/7] Building reasoning dataset..."
  "$PY" scripts/build_reasoning_dataset.py --samples 4000 --out "$OUT_DIR/reasoning_dataset.jsonl" | tee "$OUT_DIR/build_reasoning_dataset.log"

  echo "[3/7] Training jury + synthesis heads..."
  "$PY" scripts/train_fisac_jury_head.py --data "$OUT_DIR/reasoning_dataset.jsonl" --out "$JURY_HEAD" --version "jury_head_${TS}" | tee "$OUT_DIR/train_jury_head.log"
  "$PY" scripts/train_fisac_synthesis_head.py --data "$OUT_DIR/reasoning_dataset.jsonl" --out "$SYNTH_HEAD" --version "synthesis_head_${TS}" | tee "$OUT_DIR/train_synthesis_head.log"
else
  echo "[2/7] Skipping training as requested."
  echo "[3/7] Skipping training as requested."
fi

echo "[4/7] Running chat reasoning benchmark (5 seeds)..."
CHAT_BENCH="$OUT_DIR/chat_reasoning_benchmark.json"
PYTHONPATH="$ROOT" "$PY" chat_reasoning_benchmark.py \
  --steps 200 \
  --seeds 1,2,3,4,5 \
  --jury-mode 1 \
  --external-context 1 \
  --learned-heads 0 \
  --out "$CHAT_BENCH" | tee "$OUT_DIR/chat_reasoning_benchmark.log"

if [[ "$RUN_TRAINING" -eq 1 ]]; then
  PYTHONPATH="$ROOT" "$PY" chat_reasoning_benchmark.py \
    --steps 120 \
    --seeds 1,2,3,4,5 \
    --jury-mode 1 \
    --external-context 1 \
    --learned-heads 1 \
    --jury-head-path "$JURY_HEAD" \
    --synthesis-head-path "$SYNTH_HEAD" \
    --out "$OUT_DIR/chat_reasoning_benchmark_learned.json" | tee "$OUT_DIR/chat_reasoning_benchmark_learned.log"
fi

echo "[5/7] Running reasoning ablation..."
PYTHONPATH="$ROOT" "$PY" scripts/reasoning_ablation.py \
  --steps 120 \
  --seeds 1,2,3,4,5 \
  --jury-head-path "$JURY_HEAD" \
  --synthesis-head-path "$SYNTH_HEAD" \
  --out-dir "$OUT_DIR/ablation" | tee "$OUT_DIR/reasoning_ablation.log"

echo "[6/7] Running robust non-regression benchmark..."
ROBUST_RAW="$OUT_DIR/robust_raw.json"
"$PY" benchmark_robust.py \
  --seeds 7,11,19,23,29 \
  --synthetic-steps 800 \
  --real-steps 1000 \
  --structural-steps 8000 \
  --experts 64 \
  --dim 64 \
  --top-k 4 \
  --batch-size 16 \
  --device cpu > "$ROBUST_RAW"
"$PY" benchmark_stats.py --input "$ROBUST_RAW" --output-dir "$OUT_DIR" > "$OUT_DIR/benchmark_stats.log"

echo "[7/7] Evaluating gates..."
GATE_REPORT="$OUT_DIR/gate_report.json"
PYTHONPATH="$ROOT" "$PY" - "$CHAT_BENCH" "$ROBUST_RAW" "$BASELINE_RAW" "$GATE_REPORT" "$BASELINE_CHAT" <<'PYCODE'
import json
import math
import sys
from pathlib import Path

chat_path = Path(sys.argv[1])
robust_path = Path(sys.argv[2])
baseline_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])
baseline_chat_path = Path(sys.argv[5]) if len(sys.argv) > 5 else Path("")

chat = json.loads(chat_path.read_text())
rows = chat.get("rows", [])
summary = chat.get("summary", {})
chat_gates = {
    "no_signal_rate_le_0.01": float(summary.get("no_signal_rate", {}).get("mean", 1.0)) <= 0.01,
    "echo_rate_le_0.01": float(summary.get("echo_rate", {}).get("mean", 1.0)) <= 0.01,
    "repeat_streak_max_le_2": float(summary.get("repeat_streak_max", {}).get("mean", 99.0)) <= 2.0,
    "context_available_rate_eq_1": float(summary.get("context_available_rate", {}).get("mean", 0.0)) >= 0.999,
    "winner_consistency_ge_0.8": float(summary.get("winner_consistency", {}).get("mean", 0.0)) >= 0.8,
}

expert_vals = [float(r["metrics"]["expert_judgment_score"]) for r in rows if "metrics" in r and "expert_judgment_score" in r["metrics"]]
expert_mean = float(sum(expert_vals) / max(1, len(expert_vals)))
baseline_expert_mean = None
if baseline_chat_path.exists():
    bchat = json.loads(baseline_chat_path.read_text())
    brows = bchat.get("rows", [])
    bvals = [float(r["metrics"]["expert_judgment_score"]) for r in brows if "metrics" in r and "expert_judgment_score" in r["metrics"]]
    if bvals:
        baseline_expert_mean = float(sum(bvals) / len(bvals))

if baseline_expert_mean is not None and baseline_expert_mean > 1e-8:
    chat_gates["expert_judgment_plus_30pct_vs_baseline"] = expert_mean >= (1.30 * baseline_expert_mean)
else:
    chat_gates["expert_judgment_score_ge_0.12"] = expert_mean >= 0.12

robust = json.loads(robust_path.read_text())
rrows = robust.get("rows", [])
wins = sum(1 for r in rrows if float(r.get("relative_post_shift_gain_vs_gru", 0.0)) > 0.0)
comparisons = len(rrows)
peak_mem = max((float(r.get("peak_mem_gb", float("nan"))) for r in rrows), default=float("nan"))
cur_gain = [float(r.get("relative_post_shift_gain_vs_gru", 0.0)) for r in rrows]
cur_gain_mean = float(sum(cur_gain) / max(1, len(cur_gain)))

baseline_ok = baseline_path.exists()
baseline_drop_ok = True
baseline_gain_mean = None
if baseline_ok:
    brow = json.loads(baseline_path.read_text()).get("rows", [])
    bg = [float(r.get("relative_post_shift_gain_vs_gru", 0.0)) for r in brow]
    if bg:
        baseline_gain_mean = float(sum(bg) / len(bg))
        if baseline_gain_mean > 1e-8:
            baseline_drop_ok = cur_gain_mean >= (0.95 * baseline_gain_mean)

robust_gates = {
    "win_rate_vs_gru_ge_18_20": wins >= 18,
    "peak_mem_gb_le_12": (math.isfinite(peak_mem) and peak_mem <= 12.0),
    "gain_vs_gru_drop_le_5pct": baseline_drop_ok,
}

all_ok = all(chat_gates.values()) and all(robust_gates.values())
report = {
    "ok": all_ok,
    "chat_gates": chat_gates,
    "robust_gates": robust_gates,
    "chat_summary": summary,
    "expert_judgment_mean": expert_mean,
    "baseline_expert_judgment_mean": baseline_expert_mean,
    "robust": {
        "wins_vs_gru": wins,
        "comparisons": comparisons,
        "peak_mem_gb": peak_mem,
        "current_gain_vs_gru_mean": cur_gain_mean,
        "baseline_gain_vs_gru_mean": baseline_gain_mean,
        "baseline_path": str(baseline_path),
        "baseline_used": baseline_ok,
    },
}
out_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if not all_ok:
    sys.exit(2)
PYCODE

cat > "$OUT_DIR/scorecard.json" <<EOF
{
  "ok": true,
  "output_dir": "$OUT_DIR",
  "chat_benchmark": "$CHAT_BENCH",
  "chat_benchmark_learned": "$OUT_DIR/chat_reasoning_benchmark_learned.json",
  "ablation": "$OUT_DIR/ablation/ablation_summary.json",
  "robust_raw": "$ROBUST_RAW",
  "robust_summary": "$OUT_DIR/summary_stats.json",
  "gate_report": "$GATE_REPORT",
  "jury_head": "$JURY_HEAD",
  "synthesis_head": "$SYNTH_HEAD"
}
EOF

echo "Pipeline complete. Artifacts: $OUT_DIR"
