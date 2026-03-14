from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List

import torch

from benchmark_liquid import (
    _adaptation_steps,
    _post_shift_error,
    _peak_memory_gb,
    _resolve_device,
    run_ewma_baseline,
    run_gru_baseline,
    run_liquid_task,
)
from environment_stream import StreamConfig
from silicon_synapse import SiliconSynapse


@dataclass
class TaskResult:
    seed: int
    task: str
    mse: float
    baseline_gru_mse: float
    baseline_ewma_mse: float
    relative_post_shift_gain_vs_gru: float
    relative_post_shift_gain_vs_ewma: float
    adaptation_steps: float
    gru_adaptation_steps: float
    ewma_adaptation_steps: float
    steps_per_sec: float
    train_steps_per_sec: float
    seconds_per_step: float
    elapsed_seconds: float
    peak_mem_gb: float
    pruned_total: float
    myelinated_total: float
    myelinated_experts_end: int
    alive_experts_end: int


def _safe_value(value: float, default: float = 0.0) -> float:
    return float(value) if math.isfinite(value) else float(default)


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    finite = [float(v) for v in values if math.isfinite(v)]
    if not finite:
        return float(default)
    return float(statistics.mean(finite))


def _safe_pstdev(values: List[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(v)]
    if len(finite) <= 1:
        return 0.0
    return float(statistics.pstdev(finite))


def _steps_for_task(task: str, synthetic_steps: int, real_steps: int) -> int:
    return real_steps if task == "task_real" else synthetic_steps


def _run_task(
    *,
    task: str,
    seed: int,
    device: torch.device,
    dim: int,
    experts: int,
    top_k: int,
    batch_size: int,
    synthetic_steps: int,
    real_steps: int,
    csv_path: str,
    fuse_window: int,
    sleep_interval: int,
) -> TaskResult:
    torch.manual_seed(seed)
    steps = _steps_for_task(task, synthetic_steps=synthetic_steps, real_steps=real_steps)
    cfg = StreamConfig(
        feature_dim=dim,
        batch_size=batch_size,
        regime_len=500,
        noise_std=0.05,
        dt=0.02,
        seed=seed,
    )

    model = SiliconSynapse(
        feature_dim=dim,
        num_experts=experts,
        top_k=top_k,
        dt_default=cfg.dt,
        sleep_interval=sleep_interval,
        fuse_window=fuse_window,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    state = model.init_state(batch_size=batch_size, device=device)
    liquid = run_liquid_task(
        model=model,
        optimizer=optimizer,
        state=state,
        stream_name=task,
        steps=steps,
        cfg=cfg,
        device=device,
        csv_path=csv_path,
    )

    gru_errors, gru_regimes = run_gru_baseline(
        feature_dim=dim,
        steps=steps,
        cfg=cfg,
        device=device,
        seed=seed,
        stream_name=task,
        csv_path=csv_path,
    )
    ewma_errors, ewma_regimes = run_ewma_baseline(
        feature_dim=dim,
        steps=steps,
        cfg=cfg,
        seed=seed,
        stream_name=task,
        csv_path=csv_path,
    )

    liquid_mse = _safe_mean(liquid.mse_list, default=1e6)
    gru_mse = _safe_mean(gru_errors, default=1e6)
    ewma_mse = _safe_mean(ewma_errors, default=1e6)

    liquid_post = _post_shift_error(liquid.mse_list, liquid.regimes, window=80)
    gru_post = _post_shift_error(gru_errors, gru_regimes, window=80)
    ewma_post = _post_shift_error(ewma_errors, ewma_regimes, window=80)

    gain_vs_gru = _safe_value((gru_post - liquid_post) / max(gru_post, 1e-8), default=-1.0)
    gain_vs_ewma = _safe_value((ewma_post - liquid_post) / max(ewma_post, 1e-8), default=-1.0)

    return TaskResult(
        seed=seed,
        task=task,
        mse=liquid_mse,
        baseline_gru_mse=gru_mse,
        baseline_ewma_mse=ewma_mse,
        relative_post_shift_gain_vs_gru=gain_vs_gru,
        relative_post_shift_gain_vs_ewma=gain_vs_ewma,
        adaptation_steps=_safe_value(_adaptation_steps(liquid.mse_list, liquid.regimes, threshold=0.02, window=200), default=201.0),
        gru_adaptation_steps=_safe_value(
            _adaptation_steps(gru_errors, gru_regimes, threshold=0.02, window=200), default=201.0
        ),
        ewma_adaptation_steps=_safe_value(
            _adaptation_steps(ewma_errors, ewma_regimes, threshold=0.02, window=200), default=201.0
        ),
        steps_per_sec=liquid.steps_per_sec,
        train_steps_per_sec=liquid.steps_per_sec,
        seconds_per_step=liquid.seconds_per_step,
        elapsed_seconds=liquid.elapsed_seconds,
        peak_mem_gb=_peak_memory_gb(),
        pruned_total=liquid.pruned_total,
        myelinated_total=liquid.myelinated_total,
        myelinated_experts_end=int(liquid.state.myelinated_mask.sum().item() if liquid.state.myelinated_mask is not None else 0),
        alive_experts_end=int(liquid.state.alive_mask.sum().item()),
    )


def _run_structural_phase(
    *,
    seed: int,
    device: torch.device,
    dim: int,
    experts: int,
    top_k: int,
    batch_size: int,
    structural_steps: int,
    fuse_window: int,
    sleep_interval: int,
) -> Dict[str, float]:
    torch.manual_seed(seed + 1000)
    cfg = StreamConfig(
        feature_dim=dim,
        batch_size=batch_size,
        regime_len=500,
        noise_std=0.05,
        dt=0.02,
        seed=seed + 1000,
    )
    model = SiliconSynapse(
        feature_dim=dim,
        num_experts=experts,
        top_k=top_k,
        dt_default=cfg.dt,
        sleep_interval=sleep_interval,
        fuse_window=fuse_window,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    state = model.init_state(batch_size=batch_size, device=device)
    run = run_liquid_task(
        model=model,
        optimizer=optimizer,
        state=state,
        stream_name="task_c",
        steps=structural_steps,
        cfg=cfg,
        device=device,
    )

    myelinated_end = int(run.state.myelinated_mask.sum().item() if run.state.myelinated_mask is not None else 0)
    alive_end = int(run.state.alive_mask.sum().item())
    return {
        "seed": float(seed),
        "pruned_total": float(run.pruned_total),
        "myelinated_total": float(run.myelinated_total),
        "myelinated_experts_end": float(myelinated_end),
        "alive_experts_end": float(alive_end),
        "self_optimized": float((run.pruned_total > 0.0) and (run.myelinated_total > 0.0 or myelinated_end > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust multi-seed benchmark for ultra-liquid local prototype.")
    parser.add_argument("--seeds", type=str, default="7,11,19")
    parser.add_argument("--synthetic-steps", type=int, default=1500)
    parser.add_argument("--real-steps", type=int, default=2000)
    parser.add_argument("--structural-steps", type=int, default=12000)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sleep-interval", type=int, default=1000)
    parser.add_argument("--fuse-window", type=int, default=10000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--csv-path", type=str, default="StarAge/reports/smoke_run_quick/raw_records.csv")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = _resolve_device(args.device)

    tasks = ["task_a", "task_b", "task_c", "task_real"]
    all_results: List[TaskResult] = []
    for seed in seeds:
        for task in tasks:
            result = _run_task(
                task=task,
                seed=seed,
                device=device,
                dim=args.dim,
                experts=args.experts,
                top_k=args.top_k,
                batch_size=args.batch_size,
                synthetic_steps=args.synthetic_steps,
                real_steps=args.real_steps,
                csv_path=args.csv_path,
                fuse_window=args.fuse_window,
                sleep_interval=args.sleep_interval,
            )
            all_results.append(result)

    structural_runs = [
        _run_structural_phase(
            seed=seed,
            device=device,
            dim=args.dim,
            experts=args.experts,
            top_k=args.top_k,
            batch_size=args.batch_size,
            structural_steps=args.structural_steps,
            fuse_window=args.fuse_window,
            sleep_interval=args.sleep_interval,
        )
        for seed in seeds
    ]

    gains_gru = [r.relative_post_shift_gain_vs_gru for r in all_results]
    gains_ewma = [r.relative_post_shift_gain_vs_ewma for r in all_results]
    steps_s = [r.train_steps_per_sec for r in all_results]
    sec_per_step = [r.seconds_per_step for r in all_results]
    peak_mem = [r.peak_mem_gb for r in all_results]
    win_rate_gru = sum(g > 0.0 for g in gains_gru) / max(1, len(gains_gru))
    win_rate_ewma = sum(g > 0.0 for g in gains_ewma) / max(1, len(gains_ewma))
    structural_consistency = sum(int(r["self_optimized"]) for r in structural_runs) / max(1, len(structural_runs))

    per_task_summary = {}
    for task in tasks:
        task_rows = [r for r in all_results if r.task == task]
        task_gains = [r.relative_post_shift_gain_vs_gru for r in task_rows]
        per_task_summary[task] = {
            "mean_gain_vs_gru": _safe_mean(task_gains),
            "win_rate_vs_gru": float(sum(g > 0.0 for g in task_gains) / max(1, len(task_gains))),
            "mean_train_steps_per_sec": _safe_mean([r.train_steps_per_sec for r in task_rows]),
            "mean_steps_per_sec": _safe_mean([r.train_steps_per_sec for r in task_rows]),
            "mean_seconds_per_step": _safe_mean([r.seconds_per_step for r in task_rows]),
            "mean_peak_mem_gb": _safe_mean([r.peak_mem_gb for r in task_rows]),
            "mean_mse": _safe_mean([r.mse for r in task_rows], default=1e6),
        }

    summary = {
        "config": {
            "seeds": seeds,
            "tasks": tasks,
            "synthetic_steps": args.synthetic_steps,
            "real_steps": args.real_steps,
            "structural_steps": args.structural_steps,
            "experts": args.experts,
            "dim": args.dim,
            "top_k": args.top_k,
            "fuse_window": args.fuse_window,
            "sleep_interval": args.sleep_interval,
            "device": str(device),
            "csv_path": args.csv_path,
        },
        "aggregate": {
            "mean_gain_vs_gru": _safe_mean(gains_gru),
            "std_gain_vs_gru": _safe_pstdev(gains_gru),
            "win_rate_vs_gru": float(win_rate_gru),
            "mean_gain_vs_ewma": _safe_mean(gains_ewma),
            "win_rate_vs_ewma": float(win_rate_ewma),
            "mean_train_steps_per_sec": _safe_mean(steps_s),
            "mean_steps_per_sec": _safe_mean(steps_s),
            "mean_seconds_per_step": _safe_mean(sec_per_step),
            "mean_peak_mem_gb": _safe_mean(peak_mem),
            "structural_self_optimization_rate": float(structural_consistency),
        },
        "per_task": per_task_summary,
        "structural_runs": structural_runs,
        "rows": [r.__dict__ for r in all_results],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
