from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _run_step(
    name: str,
    cmd: List[str],
    cwd: Path,
    out_dir: Path,
    stdout_path: Path | None = None,
) -> Dict[str, object]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    elapsed = float(time.perf_counter() - start)

    log_path = out_dir / f"{name}.log"
    log_lines = []
    log_lines.append("$ " + " ".join(cmd))
    log_lines.append("")
    log_lines.append("=== STDOUT ===")
    log_lines.append(proc.stdout)
    log_lines.append("")
    log_lines.append("=== STDERR ===")
    log_lines.append(proc.stderr)
    log_path.write_text("\n".join(log_lines))

    if stdout_path is not None:
        stdout_path.write_text(proc.stdout)

    result = {
        "name": name,
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "elapsed_seconds": elapsed,
        "log_path": str(log_path),
        "stdout_path": str(stdout_path) if stdout_path is not None else None,
    }
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pytest + robust benchmarks + stats and write timestamped artifacts.")
    parser.add_argument("--seeds", type=str, default="7,11,19,23,29")
    parser.add_argument("--synthetic-steps", type=int, default=800)
    parser.add_argument("--real-steps", type=int, default=1000)
    parser.add_argument("--structural-steps", type=int, default=8000)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sleep-interval", type=int, default=1000)
    parser.add_argument("--fuse-window", type=int, default=4000)
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--csv-path", type=str, default="StarAge/reports/smoke_run_quick/raw_records.csv")
    parser.add_argument("--output-root", type=str, default="benchmark_runs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / args.output_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "created_at": datetime.now().isoformat(),
        "repo_root": str(repo_root),
        "output_dir": str(out_dir),
        "config": {
            "seeds": args.seeds,
            "synthetic_steps": args.synthetic_steps,
            "real_steps": args.real_steps,
            "structural_steps": args.structural_steps,
            "experts": args.experts,
            "dim": args.dim,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "sleep_interval": args.sleep_interval,
            "fuse_window": args.fuse_window,
            "device": args.device,
            "csv_path": args.csv_path,
        },
        "steps": [],
    }

    robust_json = out_dir / "robust_raw.json"

    step = _run_step(
        name="01_pytest",
        cmd=[sys.executable, "-m", "pytest", "-q"],
        cwd=repo_root,
        out_dir=out_dir,
    )
    manifest["steps"].append(step)

    for task in ("task_a", "task_b", "task_c", "task_real"):
        task_steps = args.real_steps if task == "task_real" else args.synthetic_steps
        liquid_json = out_dir / f"benchmark_liquid_{task}.json"
        cmd = [
            sys.executable,
            "benchmark_liquid.py",
            "--task",
            task,
            "--steps",
            str(task_steps),
            "--experts",
            str(args.experts),
            "--dim",
            str(args.dim),
            "--device",
            args.device,
            "--seed",
            "7",
            "--csv-path",
            args.csv_path,
        ]
        step = _run_step(
            name=f"02_benchmark_liquid_{task}",
            cmd=cmd,
            cwd=repo_root,
            out_dir=out_dir,
            stdout_path=liquid_json,
        )
        manifest["steps"].append(step)

    robust_cmd = [
        sys.executable,
        "benchmark_robust.py",
        "--seeds",
        args.seeds,
        "--synthetic-steps",
        str(args.synthetic_steps),
        "--real-steps",
        str(args.real_steps),
        "--structural-steps",
        str(args.structural_steps),
        "--experts",
        str(args.experts),
        "--dim",
        str(args.dim),
        "--top-k",
        str(args.top_k),
        "--batch-size",
        str(args.batch_size),
        "--sleep-interval",
        str(args.sleep_interval),
        "--fuse-window",
        str(args.fuse_window),
        "--device",
        args.device,
        "--csv-path",
        args.csv_path,
    ]
    step = _run_step(
        name="03_benchmark_robust",
        cmd=robust_cmd,
        cwd=repo_root,
        out_dir=out_dir,
        stdout_path=robust_json,
    )
    manifest["steps"].append(step)

    step = _run_step(
        name="04_benchmark_stats",
        cmd=[
            sys.executable,
            "benchmark_stats.py",
            "--input",
            str(robust_json),
            "--output-dir",
            str(out_dir),
        ],
        cwd=repo_root,
        out_dir=out_dir,
        stdout_path=out_dir / "benchmark_stats_outputs.json",
    )
    manifest["steps"].append(step)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "manifest": str(manifest_path),
                "robust_raw": str(robust_json),
                "paired_rows_json": str(out_dir / "paired_rows.json"),
                "paired_rows_csv": str(out_dir / "paired_rows.csv"),
                "summary_stats_json": str(out_dir / "summary_stats.json"),
                "summary_table_md": str(out_dir / "summary_table.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
