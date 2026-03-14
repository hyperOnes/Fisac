from __future__ import annotations

import argparse
import json

import torch

from environment_stream import StreamConfig, NonStationarySineStream
from silicon_synapse import FusedMyelinatedNode, LiquidNode, SiliconSynapse, default_device


def run_myelination_scan(
    warmup_steps: int = 3000,
    feature_dim: int = 64,
    num_experts: int = 128,
    top_k: int = 4,
    batch_size: int = 16,
    dt: float = 0.02,
    seed: int = 7,
) -> dict:
    torch.manual_seed(seed)
    device = default_device()
    model = SiliconSynapse(
        feature_dim=feature_dim,
        num_experts=num_experts,
        top_k=top_k,
        dt_default=dt,
        fuse_window=max(200, warmup_steps // 4),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    stream = NonStationarySineStream(
        StreamConfig(feature_dim=feature_dim, batch_size=batch_size, dt=dt, seed=seed)
    )
    state = model.init_state(batch_size=batch_size, device=device)

    for _ in range(warmup_steps):
        x_t, y_t, _ = stream.next_batch()
        _, state, _ = model.online_step(
            x_t=x_t.to(device),
            y_target=y_t.to(device),
            state=state,
            dt=dt,
            optimizer=optimizer,
        )

    # Lower thresholds only for the scan demo so fusion can be observed quickly.
    model.fuse_usage_threshold = min(model.fuse_usage_threshold, 0.02)
    model.fuse_error_var_threshold = max(model.fuse_error_var_threshold, 1.0)
    model.fuse_window = 1
    scan_info = model.myelinate(state)

    fused_count = sum(isinstance(ex, FusedMyelinatedNode) for ex in model.experts)
    liquid_count = sum(isinstance(ex, LiquidNode) for ex in model.experts)
    return {
        "myelinated_now": scan_info["myelinated"],
        "fused_count": fused_count,
        "liquid_count": liquid_count,
        "alive_experts": int(state.alive_mask.sum().item()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run explicit myelination scan for SiliconSynapse.")
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_myelination_scan(
        warmup_steps=args.warmup_steps,
        feature_dim=args.dim,
        num_experts=args.experts,
        top_k=args.top_k,
        batch_size=args.batch_size,
        dt=args.dt,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
