#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run A/B/C ablation for Fisac genius chat stack")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_runs/reasoning_ablation"))
    p.add_argument("--jury-head-path", type=str, default="")
    p.add_argument("--synthesis-head-path", type=str, default="")
    return p.parse_args()


def _run_case(
    name: str,
    *,
    jury_mode: int,
    external_context: int,
    learned_heads: int,
    jury_head_path: str,
    synthesis_head_path: str,
    steps: int,
    seeds: str,
    out: Path,
) -> dict:
    cmd = [
        sys.executable,
        "chat_reasoning_benchmark.py",
        "--steps",
        str(steps),
        "--seeds",
        seeds,
        "--jury-mode",
        str(jury_mode),
        "--external-context",
        str(external_context),
        "--learned-heads",
        str(learned_heads),
        "--out",
        str(out),
    ]
    if jury_head_path.strip():
        cmd.extend(["--jury-head-path", jury_head_path.strip()])
    if synthesis_head_path.strip():
        cmd.extend(["--synthesis-head-path", synthesis_head_path.strip()])
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed: {proc.stderr[:400]}")
    payload = json.loads(out.read_text(encoding="utf-8"))
    return {
        "name": name,
        "config": payload["config"],
        "summary": payload["summary"],
        "path": str(out),
    }


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("A_runtime_only", 1, 0, 0),
        ("B_context_only", 0, 1, 0),
        ("A_plus_B", 1, 1, 0),
        ("A_plus_B_plus_C", 1, 1, 1),
    ]

    results = []
    for name, jury_mode, ext, learned in cases:
        out = args.out_dir / f"{name}.json"
        results.append(
            _run_case(
                name,
                jury_mode=jury_mode,
                external_context=ext,
                learned_heads=learned,
                jury_head_path=args.jury_head_path,
                synthesis_head_path=args.synthesis_head_path,
                steps=args.steps,
                seeds=args.seeds,
                out=out,
            )
        )

    report = {"cases": results}
    report_path = args.out_dir / "ablation_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
