from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Tuple

import torch

from silicon_synapse import SiliconSynapse, default_device


@dataclass
class TelemetryConfig:
    feature_dim: int = 64
    batch_size: int = 1
    dt: float = 0.02
    noise_std: float = 0.03
    ambient_temp: float = 18.0
    base_hrv: float = 45.0
    base_skin_temp: float = 32.5
    seed: int = 7


class VitruviusTelemetryStream:
    def __init__(self, cfg: TelemetryConfig) -> None:
        self.cfg = cfg
        self.t = 0.0
        self.rng = torch.Generator()
        self.rng.manual_seed(cfg.seed)

    def _build_vector(self, hrv: float, skin_temp: float) -> torch.Tensor:
        x = torch.randn(self.cfg.batch_size, self.cfg.feature_dim, generator=self.rng) * self.cfg.noise_std
        x[:, 0] = hrv
        x[:, 1] = skin_temp
        x[:, 2] = self.cfg.ambient_temp
        x[:, 3] = math.sin(self.t * 0.75)
        x[:, 4] = math.cos(self.t * 0.25)
        return x

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        self.t += self.cfg.dt
        breathing = math.sin(self.t * 2.0) * 5.0
        sympathetic_load = math.cos(self.t * 0.5) * 0.3

        hrv = self.cfg.base_hrv + breathing + torch.randn((), generator=self.rng).item() * 2.0
        skin_temp = self.cfg.base_skin_temp + sympathetic_load + torch.randn((), generator=self.rng).item() * 0.05

        x_t = self._build_vector(hrv=hrv, skin_temp=skin_temp)

        t_next = self.t + self.cfg.dt
        next_hrv = self.cfg.base_hrv + math.sin(t_next * 2.0) * 5.0
        next_skin = self.cfg.base_skin_temp + math.cos(t_next * 0.5) * 0.3
        y_t = self._build_vector(hrv=next_hrv, skin_temp=next_skin)
        return x_t, y_t, {"hrv": hrv, "skin_temp": skin_temp}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live telemetry demo for SiliconSynapse.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = default_device()

    model = SiliconSynapse(feature_dim=args.dim, num_experts=args.experts, top_k=args.top_k, dt_default=args.dt).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    state = model.init_state(batch_size=1, device=device)
    stream = VitruviusTelemetryStream(
        TelemetryConfig(feature_dim=args.dim, batch_size=1, dt=args.dt, seed=args.seed)
    )

    for step in range(1, args.steps + 1):
        x_t, y_t, meta = stream.sample()
        pred, state, info = model.online_step(
            x_t=x_t.to(device),
            y_target=y_t.to(device),
            state=state,
            dt=args.dt,
            optimizer=optimizer,
        )
        pump_velocity_command = torch.tanh(pred[0, 0]).item() * 100.0
        print(
            f"step={step:>4} "
            f"HRV={meta['hrv']:>6.2f}ms "
            f"skin={meta['skin_temp']:>5.2f}C "
            f"pump={pump_velocity_command:>7.2f}% "
            f"mse={info['mse']:.5f}"
        )
        time.sleep(args.sleep)
