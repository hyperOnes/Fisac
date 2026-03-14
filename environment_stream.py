from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader, IterableDataset


@dataclass
class StreamConfig:
    feature_dim: int = 64
    regime_len: int = 500
    noise_std: float = 0.05
    batch_size: int = 16
    dt: float = 0.02
    seed: int = 7


class NonStationarySineStream:
    """
    Task A stream: two sine waves with abrupt regime shifts every regime_len.
    """

    def __init__(self, config: StreamConfig) -> None:
        self.cfg = config
        self.step_idx = 0
        self.rng = torch.Generator()
        self.rng.manual_seed(config.seed)
        self._regimes = [
            (0.7, 1.8, 1.0, 0.55),
            (1.2, 2.7, 0.9, 0.50),
            (0.4, 1.1, 1.1, 0.45),
            (1.5, 3.2, 0.8, 0.35),
        ]

    def _regime(self, step: int) -> Tuple[float, float, float, float]:
        return self._regimes[(step // self.cfg.regime_len) % len(self._regimes)]

    def _signal(self, t: torch.Tensor, regime: Tuple[float, float, float, float]) -> torch.Tensor:
        f1, f2, a1, a2 = regime
        return a1 * torch.sin(2.0 * math.pi * f1 * t) + a2 * torch.sin(2.0 * math.pi * f2 * t)

    def _pack_features(self, signal: torch.Tensor, t: torch.Tensor, regime_id: int) -> torch.Tensor:
        bsz = signal.shape[0]
        x = torch.randn(bsz, self.cfg.feature_dim, generator=self.rng) * self.cfg.noise_std
        x[:, 0] = signal
        x[:, 1] = torch.cos(2.0 * math.pi * t)
        x[:, 2] = torch.sin(4.0 * math.pi * t)
        x[:, 3] = float(regime_id) / max(1, len(self._regimes) - 1)
        x[:, 4] = x[:, 0] * x[:, 1]
        return x

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        regime_now = self._regime(self.step_idx)
        regime_next = self._regime(self.step_idx + 1)
        regime_id_now = (self.step_idx // self.cfg.regime_len) % len(self._regimes)
        regime_id_next = ((self.step_idx + 1) // self.cfg.regime_len) % len(self._regimes)

        base_t = torch.full((self.cfg.batch_size,), self.step_idx * self.cfg.dt)
        jitter = (torch.rand(self.cfg.batch_size, generator=self.rng) - 0.5) * self.cfg.dt
        t_now = base_t + jitter
        t_next = t_now + self.cfg.dt

        signal_now = self._signal(t_now, regime_now) + torch.randn(self.cfg.batch_size, generator=self.rng) * self.cfg.noise_std
        signal_next = self._signal(t_next, regime_next) + torch.randn(self.cfg.batch_size, generator=self.rng) * self.cfg.noise_std

        x_t = self._pack_features(signal=signal_now, t=t_now, regime_id=regime_id_now)
        y_t = self._pack_features(signal=signal_next, t=t_next, regime_id=regime_id_next)
        self.step_idx += 1
        return x_t, y_t, {"step": self.step_idx, "regime_id": regime_id_now}


class RuleSwitchStream:
    """
    Task B stream: latent rule switching with abrupt regime changes.
    """

    def __init__(self, config: StreamConfig) -> None:
        self.cfg = config
        self.step_idx = 0
        self.rng = torch.Generator()
        self.rng.manual_seed(config.seed + 1234)
        self._rules = ("sum", "mul", "xor")

    def _rule(self, step: int) -> str:
        return self._rules[(step // self.cfg.regime_len) % len(self._rules)]

    def _rule_apply(self, x: torch.Tensor, rule: str) -> torch.Tensor:
        y = torch.zeros_like(x)
        if rule == "sum":
            y[:, 0] = x[:, 0] + 0.5 * x[:, 1]
            y[:, 1] = x[:, 1] - 0.2 * x[:, 2]
        elif rule == "mul":
            y[:, 0] = x[:, 0] * x[:, 1]
            y[:, 1] = x[:, 1] * x[:, 2]
        else:
            b0 = (x[:, 0] > 0).float()
            b1 = (x[:, 1] > 0).float()
            y[:, 0] = (b0 != b1).float() * 2.0 - 1.0
            y[:, 1] = (x[:, 2] > 0).float() * 2.0 - 1.0

        y[:, 2:] = 0.9 * x[:, 2:] + 0.1 * torch.tanh(x[:, :-2].mean(dim=1, keepdim=True))
        return y

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        rule = self._rule(self.step_idx)
        rule_id = (self.step_idx // self.cfg.regime_len) % len(self._rules)
        x = torch.randn(self.cfg.batch_size, self.cfg.feature_dim, generator=self.rng)
        x += torch.randn(self.cfg.batch_size, self.cfg.feature_dim, generator=self.rng) * self.cfg.noise_std
        y = self._rule_apply(x, rule) + torch.randn(self.cfg.batch_size, self.cfg.feature_dim, generator=self.rng) * (
            self.cfg.noise_std * 0.5
        )
        self.step_idx += 1
        return x, y, {"step": self.step_idx, "regime_id": rule_id}


class ChaoticRegimeStream:
    """
    Task C stream: harder non-stationary process with abrupt/continuous drift,
    heteroskedastic noise, and nonlinear regime-specific transition rules.
    """

    def __init__(self, config: StreamConfig) -> None:
        self.cfg = config
        self.step_idx = 0
        self.rng = torch.Generator()
        self.rng.manual_seed(config.seed + 4242)
        self.state = torch.randn(config.batch_size, 3, generator=self.rng) * 0.5

    def _regime(self, step: int) -> int:
        return (step // self.cfg.regime_len) % 4

    def _transition(self, x: torch.Tensor, regime: int) -> torch.Tensor:
        if regime == 0:
            # Quasi-periodic oscillator with mild damping.
            next_x = 0.88 * x
            next_x[:, 0] += 0.25 * torch.sin(1.7 * x[:, 1])
            next_x[:, 1] += 0.18 * torch.cos(1.2 * x[:, 2])
            next_x[:, 2] += 0.12 * torch.sin(1.5 * x[:, 0])
        elif regime == 1:
            # Logistic-chaotic coupling.
            next_x = 0.75 * x
            next_x[:, 0] += 0.95 * x[:, 0] * (1.0 - torch.tanh(x[:, 0]) ** 2)
            next_x[:, 1] += 0.65 * x[:, 0] * x[:, 2]
            next_x[:, 2] += 0.55 * torch.sin(2.3 * x[:, 1])
        elif regime == 2:
            # Piecewise saturating dynamics.
            next_x = 0.65 * x
            next_x[:, 0] += torch.where(x[:, 1] > 0.0, 0.7 * x[:, 1], -0.3 * x[:, 1])
            next_x[:, 1] += torch.where(x[:, 2] > 0.0, -0.5 * x[:, 2], 0.8 * x[:, 2])
            next_x[:, 2] += 0.45 * torch.tanh(2.0 * x[:, 0] * x[:, 1])
        else:
            # Fast mean-reverting with jumps.
            next_x = 0.55 * x
            jumps = (torch.rand(x.size(0), generator=self.rng) < 0.08).float().unsqueeze(1)
            jump_noise = torch.randn_like(x, generator=self.rng) * 0.75
            next_x += jumps * jump_noise
            next_x[:, 0] += 0.3 * torch.sin(3.1 * x[:, 2])
            next_x[:, 1] += 0.25 * torch.cos(2.7 * x[:, 0])
            next_x[:, 2] += 0.2 * torch.sin(2.1 * x[:, 1])
        return next_x

    def _pack(self, latent: torch.Tensor, regime: int) -> torch.Tensor:
        bsz = latent.size(0)
        x = torch.zeros(bsz, self.cfg.feature_dim)
        x[:, :3] = latent
        x[:, 3] = float(regime) / 3.0
        x[:, 4] = latent[:, 0] * latent[:, 1]
        x[:, 5] = latent[:, 1] * latent[:, 2]
        x[:, 6] = torch.sin(latent[:, 0])
        x[:, 7] = torch.cos(latent[:, 1])
        if self.cfg.feature_dim > 8:
            x[:, 8:] = torch.randn(bsz, self.cfg.feature_dim - 8, generator=self.rng) * self.cfg.noise_std
        return x

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        regime_now = self._regime(self.step_idx)
        regime_next = self._regime(self.step_idx + 1)
        noise_scale = self.cfg.noise_std * (1.0 + 0.5 * regime_now)

        x_now = self.state
        next_latent = self._transition(x_now, regime=regime_now)
        next_latent += torch.randn_like(next_latent, generator=self.rng) * noise_scale

        x_t = self._pack(x_now, regime=regime_now)
        y_t = self._pack(next_latent, regime=regime_next)

        self.state = next_latent
        self.step_idx += 1
        return x_t, y_t, {"step": self.step_idx, "regime_id": regime_now}


class CSVForecastStream:
    """
    Streams numeric columns from a local CSV as a non-stationary short-horizon forecast.
    Inputs stay grounded in real rows, while the target applies abrupt regime-specific
    operators to emulate live telemetry distribution shifts.
    """

    def __init__(self, config: StreamConfig, csv_path: str | Path, noise_std: float = 0.0) -> None:
        self.cfg = config
        self.step_idx = 0
        self.rng = torch.Generator()
        self.rng.manual_seed(config.seed + 9876)
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")
        self.noise_std = noise_std

        self.data, self.regimes = self._load_csv(self.csv_path)
        if self.data.size(0) < 3:
            raise ValueError(f"CSV stream requires at least 3 rows, got {self.data.size(0)}")
        self.core_dim = min(8, self.data.size(1))
        self.bucket_block = max(16, self.cfg.regime_len // 4)
        self.bucket_indices = []
        for b in range(4):
            idx = torch.nonzero((self.regimes % 4) == b, as_tuple=False).flatten()
            if idx.numel() < 2:
                idx = torch.arange(0, self.data.size(0) - 1, dtype=torch.long)
            self.bucket_indices.append(idx)
        self.bucket_slot_pos = [
            torch.randint(0, idx.numel(), (self.cfg.batch_size,), generator=self.rng, dtype=torch.long)
            for idx in self.bucket_indices
        ]

    def _load_csv(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"CSV has no data rows: {path}")

        headers = list(rows[0].keys())
        numeric_cols = []
        for h in headers:
            ok = True
            for row in rows[: min(512, len(rows))]:
                try:
                    float(row[h])
                except (TypeError, ValueError):
                    ok = False
                    break
            if ok:
                numeric_cols.append(h)
        if not numeric_cols:
            raise ValueError(f"No numeric columns found in CSV: {path}")

        numeric = []
        regimes = []
        tissue_lookup: Dict[str, int] = {}
        for row in rows:
            vec = [float(row[h]) for h in numeric_cols]
            numeric.append(vec)

            # Build a regime ID from available categorical/metadata fields.
            tissue = row.get("tissue", "unknown")
            if tissue not in tissue_lookup:
                tissue_lookup[tissue] = len(tissue_lookup)
            tissue_id = tissue_lookup[tissue]
            age = float(row.get("donor_age", 0.0))
            age_bin = int(age // 10.0)
            regimes.append(tissue_id * 16 + age_bin)

        x = torch.tensor(numeric, dtype=torch.float32)
        finite = torch.isfinite(x)
        if not torch.all(finite):
            # Replace non-finite entries with per-column finite means.
            finite_count = finite.sum(dim=0).clamp(min=1)
            finite_sum = torch.where(finite, x, torch.zeros_like(x)).sum(dim=0)
            col_mean = finite_sum / finite_count
            x = torch.where(finite, x, col_mean.unsqueeze(0))
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp(min=1e-4)
        x = (x - mean) / std
        return x, torch.tensor(regimes, dtype=torch.long)

    def _pack(self, batch_rows: torch.Tensor, row_idx: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        del regime
        bsz, cols = batch_rows.shape
        x = torch.zeros(bsz, self.cfg.feature_dim)
        n = min(cols, self.cfg.feature_dim)
        x[:, :n] = batch_rows[:, :n]
        if self.cfg.feature_dim > n:
            idx_norm = row_idx.float() / max(1.0, float(self.data.size(0) - 1))
            x[:, n] = idx_norm if n < self.cfg.feature_dim else 0.0
            if n + 1 < self.cfg.feature_dim:
                x[:, n + 1] = torch.sin(2.0 * math.pi * idx_norm)
            if n + 2 < self.cfg.feature_dim:
                x[:, n + 2] = torch.cos(2.0 * math.pi * idx_norm)
        if self.noise_std > 0:
            x += torch.randn_like(x, generator=self.rng) * self.noise_std
        return x

    def _regime_transform(self, x_rows: torch.Tensor, y_rows: torch.Tensor, bucket: int) -> torch.Tensor:
        # Mostly-local dynamics (short-horizon forecasting) plus abrupt regime operators.
        out = 0.9 * x_rows + 0.1 * y_rows
        core = x_rows[:, : self.core_dim]
        x_roll = torch.roll(core, shifts=1, dims=1)

        if bucket == 0:
            out[:, : self.core_dim] = out[:, : self.core_dim] + 0.08 * x_roll
        elif bucket == 1:
            split = max(1, self.core_dim // 2)
            perm = torch.cat([core[:, split:], core[:, :split]], dim=1)
            out[:, : self.core_dim] = out[:, : self.core_dim] - 0.12 * perm + 0.05 * x_roll
        elif bucket == 2:
            out[:, : self.core_dim] = out[:, : self.core_dim] + 0.15 * torch.tanh(1.2 * core) + 0.06 * core * x_roll
        else:
            out[:, : self.core_dim] = out[:, : self.core_dim] + 0.10 * torch.sign(core) - 0.05 * x_roll
        return out

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        bucket = (self.step_idx // self.bucket_block) % 4
        ids = self.bucket_indices[bucket]
        pos = self.bucket_slot_pos[bucket]
        idx = ids[pos]
        idx_next = ids[(pos + 1) % ids.numel()]
        self.bucket_slot_pos[bucket] = (pos + 1) % ids.numel()

        x_rows = self.data[idx]
        y_rows = self.data[idx_next]
        y_rows = self._regime_transform(x_rows=x_rows, y_rows=y_rows, bucket=bucket)
        x_t = self._pack(x_rows, row_idx=idx, regime=self.regimes[idx])
        y_t = self._pack(y_rows, row_idx=idx_next, regime=self.regimes[idx_next])

        regime_now = int(bucket)
        self.step_idx += 1
        return x_t, y_t, {"step": self.step_idx, "regime_id": regime_now}


class _ContinuousDataset(IterableDataset):
    def __init__(self, stream: NonStationarySineStream | RuleSwitchStream | ChaoticRegimeStream | CSVForecastStream) -> None:
        super().__init__()
        self.stream = stream

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]]:
        while True:
            yield self.stream.next_batch()


def build_dataloader(
    stream: NonStationarySineStream | RuleSwitchStream | ChaoticRegimeStream | CSVForecastStream,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = _ContinuousDataset(stream=stream)
    return DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=pin_memory)


def make_stream(
    task: str,
    config: Optional[StreamConfig] = None,
    csv_path: str | Path | None = None,
) -> NonStationarySineStream | RuleSwitchStream | ChaoticRegimeStream | CSVForecastStream:
    config = config or StreamConfig()
    if task.lower() in {"task_a", "sine", "forecast"}:
        return NonStationarySineStream(config=config)
    if task.lower() in {"task_b", "rule", "switch"}:
        return RuleSwitchStream(config=config)
    if task.lower() in {"task_c", "chaotic", "hard"}:
        return ChaoticRegimeStream(config=config)
    if task.lower() in {"task_real", "real_csv", "csv"}:
        if csv_path is None:
            raise ValueError("csv_path is required for task_real streams.")
        return CSVForecastStream(config=config, csv_path=csv_path)
    raise ValueError(f"Unknown task '{task}'. Use one of: task_a, task_b, task_c, task_real.")


if __name__ == "__main__":
    cfg = StreamConfig()
    stream = NonStationarySineStream(cfg)
    loader = build_dataloader(stream)
    for idx, (x_t, y_t, meta) in enumerate(loader):
        print(f"step={meta['step']} regime={meta['regime_id']} x={tuple(x_t.shape)} y={tuple(y_t.shape)}")
        if idx >= 2:
            break
