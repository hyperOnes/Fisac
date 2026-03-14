from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict

import torch

from environment_stream import StreamConfig, make_stream
from silicon_synapse import SiliconSynapse, SynapseState, default_device


@dataclass
class SleepCycleConfig:
    steps: int = 5000
    log_every: int = 250
    dt: float = 0.02
    task: str = "task_a"
    feature_dim: int = 64
    num_experts: int = 128
    top_k: int = 4
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 7


def run_sleep_cycle(config: SleepCycleConfig) -> Dict[str, float]:
    torch.manual_seed(config.seed)
    device = default_device()

    model = SiliconSynapse(
        feature_dim=config.feature_dim,
        num_experts=config.num_experts,
        top_k=config.top_k,
        dt_default=config.dt,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    stream_cfg = StreamConfig(
        feature_dim=config.feature_dim,
        batch_size=config.batch_size,
        dt=config.dt,
        seed=config.seed,
    )
    stream = make_stream(task=config.task, config=stream_cfg)
    state: SynapseState = model.init_state(batch_size=config.batch_size, device=device)

    mse_sum = 0.0
    pruned_total = 0.0
    myelinated_total = 0.0

    for step in range(1, config.steps + 1):
        x_t, y_t, _ = stream.next_batch()
        x_t = x_t.to(device)
        y_t = y_t.to(device)

        _, state, info = model.online_step(
            x_t=x_t,
            y_target=y_t,
            state=state,
            dt=config.dt,
            optimizer=optimizer,
        )
        mse_sum += info["mse"]
        pruned_total += info["pruned_now"]
        myelinated_total += info["myelinated_now"]

        if step % config.log_every == 0:
            print(
                f"step={step:>6} mse={info['mse']:.6f} "
                f"alive={int(info['alive_experts'])}/{config.num_experts} "
                f"myelinated={int(info['myelinated_experts'])} "
                f"pruned_now={int(info['pruned_now'])}"
            )

    result = {
        "avg_mse": mse_sum / config.steps,
        "alive_experts": float(state.alive_mask.sum().item()),
        "myelinated_experts": float(state.myelinated_mask.sum().item() if state.myelinated_mask is not None else 0.0),
        "pruned_total": pruned_total,
        "myelinated_total": myelinated_total,
        "steps": float(config.steps),
    }
    return result


def _parse_args() -> SleepCycleConfig:
    parser = argparse.ArgumentParser(description="Run online replay-resolve-prune loop for SiliconSynapse.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--task", type=str, default="task_a", choices=["task_a", "task_b"])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    return SleepCycleConfig(
        steps=args.steps,
        log_every=args.log_every,
        dt=args.dt,
        task=args.task,
        feature_dim=args.dim,
        num_experts=args.experts,
        top_k=args.top_k,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    print(json.dumps({"config": asdict(cfg)}, indent=2))
    summary = run_sleep_cycle(cfg)
    print(json.dumps(summary, indent=2))
