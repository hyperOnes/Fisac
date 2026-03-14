from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from environment_stream import RuleSwitchStream, StreamConfig, make_stream
from silicon_synapse import SiliconSynapse, SynapseState, default_device


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return default_device()
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but not available.")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _peak_memory_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and KiB on Linux.
    if sys.platform == "darwin":
        return float(rss) / (1024.0**3)
    return float(rss * 1024) / (1024.0**3)


def _adaptation_steps(errors: List[float], regimes: List[int], threshold: float = 0.02, window: int = 200) -> float:
    shifts = [i for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1]]
    if not shifts:
        return float("nan")
    recoveries: List[int] = []
    for shift in shifts:
        recovered = window + 1
        end = min(len(errors), shift + window)
        for idx in range(shift, end):
            if errors[idx] <= threshold:
                recovered = idx - shift + 1
                break
        recoveries.append(recovered)
    return float(statistics.mean(recoveries))


def _post_shift_error(errors: List[float], regimes: List[int], window: int = 80) -> float:
    shifts = [i for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1]]
    if not shifts:
        return float(statistics.mean(errors))
    values: List[float] = []
    for shift in shifts:
        end = min(len(errors), shift + window)
        values.extend(errors[shift:end])
    return float(statistics.mean(values)) if values else float(statistics.mean(errors))


class EWMABaseline:
    def __init__(self, feature_dim: int, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.value: torch.Tensor | None = None
        self.feature_dim = feature_dim

    def step(self, x_t: torch.Tensor, y_t: torch.Tensor) -> float:
        if self.value is None:
            self.value = x_t.detach().clone()
        pred = self.value
        mse = F.mse_loss(pred, y_t).item()
        self.value = self.alpha * y_t.detach() + (1.0 - self.alpha) * self.value
        return float(mse)


class GRUBaseline(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRUCell(feature_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, feature_dim)
        self.hidden_dim = hidden_dim

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, x_t: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_next = self.gru(x_t, hidden)
        return self.readout(hidden_next), hidden_next


@dataclass
class LiquidRunResult:
    mse_list: List[float]
    regimes: List[int]
    state: SynapseState
    steps_per_sec: float
    seconds_per_step: float
    elapsed_seconds: float
    pruned_total: float
    myelinated_total: float


def run_liquid_task(
    model: SiliconSynapse,
    optimizer: torch.optim.Optimizer,
    state: SynapseState,
    stream_name: str,
    steps: int,
    cfg: StreamConfig,
    device: torch.device,
    csv_path: str | None = None,
) -> LiquidRunResult:
    stream = make_stream(stream_name, cfg, csv_path=csv_path)
    mse_list: List[float] = []
    regimes: List[int] = []
    pruned_total = 0.0
    myelinated_total = 0.0

    start = time.perf_counter()
    for _ in range(steps):
        x_t, y_t, meta = stream.next_batch()
        _, state, info = model.online_step(
            x_t=x_t.to(device),
            y_target=y_t.to(device),
            state=state,
            dt=cfg.dt,
            optimizer=optimizer,
        )
        mse_list.append(info["mse"])
        regimes.append(meta["regime_id"])
        pruned_total += info.get("pruned_now", 0.0)
        myelinated_total += info.get("myelinated_now", 0.0)
    elapsed = max(time.perf_counter() - start, 1e-6)
    steps_per_sec = float(steps / elapsed)
    return LiquidRunResult(
        mse_list=mse_list,
        regimes=regimes,
        state=state,
        steps_per_sec=steps_per_sec,
        seconds_per_step=float(elapsed / max(1, steps)),
        elapsed_seconds=float(elapsed),
        pruned_total=pruned_total,
        myelinated_total=myelinated_total,
    )


def run_gru_baseline(
    feature_dim: int,
    steps: int,
    cfg: StreamConfig,
    device: torch.device,
    seed: int,
    stream_name: str = "task_a",
    csv_path: str | None = None,
) -> Tuple[List[float], List[int]]:
    torch.manual_seed(seed + 1)
    stream = make_stream(stream_name, cfg, csv_path=csv_path)
    model = GRUBaseline(feature_dim=feature_dim, hidden_dim=feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    hidden = model.init_state(batch_size=cfg.batch_size, device=device)

    errors: List[float] = []
    regimes: List[int] = []
    for _ in range(steps):
        x_t, y_t, meta = stream.next_batch()
        x_t = x_t.to(device)
        y_t = y_t.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred, hidden = model(x_t, hidden)
        loss = F.mse_loss(pred, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        hidden = hidden.detach()
        errors.append(float(loss.item()))
        regimes.append(meta["regime_id"])
    return errors, regimes


def run_ewma_baseline(
    feature_dim: int,
    steps: int,
    cfg: StreamConfig,
    seed: int,
    stream_name: str = "task_a",
    csv_path: str | None = None,
) -> Tuple[List[float], List[int]]:
    torch.manual_seed(seed + 2)
    stream = make_stream(stream_name, cfg, csv_path=csv_path)
    ewma = EWMABaseline(feature_dim=feature_dim)
    errors: List[float] = []
    regimes: List[int] = []
    for _ in range(steps):
        x_t, y_t, meta = stream.next_batch()
        errors.append(ewma.step(x_t=x_t, y_t=y_t))
        regimes.append(meta["regime_id"])
    return errors, regimes


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ultra-liquid online adaptation.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", type=str, default="task_a", choices=["task_a", "task_b", "task_c", "task_real"])
    parser.add_argument("--csv-path", type=str, default="StarAge/reports/smoke_run_quick/raw_records.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    cfg = StreamConfig(feature_dim=args.dim, batch_size=16, regime_len=500, noise_std=0.05, dt=0.02, seed=args.seed)

    model = SiliconSynapse(
        feature_dim=args.dim,
        num_experts=args.experts,
        top_k=4,
        dt_default=cfg.dt,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    state = model.init_state(batch_size=cfg.batch_size, device=device)

    task_a = run_liquid_task(
        model=model,
        optimizer=optimizer,
        state=state,
        stream_name=args.task,
        steps=args.steps,
        cfg=cfg,
        device=device,
        csv_path=args.csv_path,
    )
    task_b = run_liquid_task(
        model=model,
        optimizer=optimizer,
        state=task_a.state,
        stream_name="task_b",
        steps=args.steps,
        cfg=cfg,
        device=device,
    )

    gru_errors, gru_regimes = run_gru_baseline(
        feature_dim=args.dim,
        steps=args.steps,
        cfg=cfg,
        device=device,
        seed=args.seed,
        stream_name=args.task,
        csv_path=args.csv_path,
    )
    ewma_errors, ewma_regimes = run_ewma_baseline(
        feature_dim=args.dim,
        steps=args.steps,
        cfg=cfg,
        seed=args.seed,
        stream_name=args.task,
        csv_path=args.csv_path,
    )

    liquid_mse = float(statistics.mean(task_a.mse_list))
    liquid_adapt = _adaptation_steps(task_a.mse_list, task_a.regimes, threshold=0.02, window=200)
    gru_mse = float(statistics.mean(gru_errors))
    ewma_mse = float(statistics.mean(ewma_errors))

    liquid_post = _post_shift_error(task_a.mse_list, task_a.regimes, window=80)
    gru_post = _post_shift_error(gru_errors, gru_regimes, window=80)
    liquid_post_long = _post_shift_error(task_a.mse_list, task_a.regimes, window=200)
    gru_post_long = _post_shift_error(gru_errors, gru_regimes, window=200)
    relative_gain_vs_gru = float((gru_post - liquid_post) / max(gru_post, 1e-8))
    relative_gain_vs_gru_long = float((gru_post_long - liquid_post_long) / max(gru_post_long, 1e-8))

    metrics = {
        "mse": liquid_mse,
        "adaptation_steps": liquid_adapt,
        "train_steps_per_sec": task_a.steps_per_sec,
        "steps_per_sec": task_a.steps_per_sec,
        "seconds_per_step": task_a.seconds_per_step,
        "elapsed_seconds": task_a.elapsed_seconds,
        "peak_mem_gb": _peak_memory_gb(),
        "alive_experts": int(task_b.state.alive_mask.sum().item()),
        "myelinated_experts": int(task_b.state.myelinated_mask.sum().item() if task_b.state.myelinated_mask is not None else 0),
        "pruned_total": task_a.pruned_total + task_b.pruned_total,
        "myelinated_total": task_a.myelinated_total + task_b.myelinated_total,
        "task_b_mse": float(statistics.mean(task_b.mse_list)),
        "baseline_gru_mse": gru_mse,
        "baseline_ewma_mse": ewma_mse,
        "relative_post_shift_gain_vs_gru": relative_gain_vs_gru,
        "relative_post_shift_gain_vs_gru_long": relative_gain_vs_gru_long,
        "gru_adaptation_steps": _adaptation_steps(gru_errors, gru_regimes, threshold=0.02, window=200),
        "ewma_adaptation_steps": _adaptation_steps(ewma_errors, ewma_regimes, threshold=0.02, window=200),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
