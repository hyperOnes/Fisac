from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class CursorDatasetTensors:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor


class CursorWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.size(0))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def load_mouse_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.genfromtxt(path, delimiter=",", names=True)
    if raw.size == 0:
        raise ValueError(f"No samples in {path}")
    t_ms = np.asarray(raw["timestamp_ms"], dtype=np.float64)
    x = np.asarray(raw["x"], dtype=np.float64)
    y = np.asarray(raw["y"], dtype=np.float64)
    order = np.argsort(t_ms)
    return t_ms[order], x[order], y[order]


def resample_to_hz(t_ms: np.ndarray, x: np.ndarray, y: np.ndarray, hz: float = 60.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if t_ms.size < 4:
        raise ValueError("Need at least 4 samples to resample")
    dt = 1000.0 / hz
    t_new = np.arange(t_ms[0], t_ms[-1], dt, dtype=np.float64)
    x_new = np.interp(t_new, t_ms, x)
    y_new = np.interp(t_new, t_ms, y)
    return t_new, x_new, y_new


def build_features(t_ms: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dt_s = np.diff(t_ms, prepend=t_ms[0]) / 1000.0
    dt_s[dt_s <= 1e-6] = 1.0 / 60.0
    vx = np.diff(x, prepend=x[0]) / dt_s
    vy = np.diff(y, prepend=y[0]) / dt_s

    x_n = (x - np.min(x)) / max(1e-6, (np.max(x) - np.min(x)))
    y_n = (y - np.min(y)) / max(1e-6, (np.max(y) - np.min(y)))
    vx_n = vx / max(1e-6, np.percentile(np.abs(vx), 95))
    vy_n = vy / max(1e-6, np.percentile(np.abs(vy), 95))
    feats = np.stack([x_n, y_n, vx_n, vy_n], axis=1).astype(np.float32)
    return np.clip(feats, -4.0, 4.0)


def make_windows(
    features: np.ndarray,
    sequence_length: int = 30,
    horizon_start: int = 15,
    horizon_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    xy = features[:, :2]
    max_h = horizon_start + horizon_points - 1
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    start = sequence_length - 1
    end = features.shape[0] - max_h
    for i in range(start, end):
        seq = features[i - sequence_length + 1 : i + 1]
        tgt = np.stack([xy[i + horizon_start + k] for k in range(horizon_points)], axis=0)
        xs.append(seq)
        ys.append(tgt)
    if not xs:
        raise ValueError("Not enough samples after windowing")
    return np.stack(xs, axis=0).astype(np.float32), np.stack(ys, axis=0).astype(np.float32)


def split_tensors(x: torch.Tensor, y: torch.Tensor, train_ratio: float = 0.7, val_ratio: float = 0.15) -> CursorDatasetTensors:
    n = x.size(0)
    i_train = max(1, int(n * train_ratio))
    i_val = max(i_train + 1, int(n * (train_ratio + val_ratio)))
    i_val = min(i_val, n - 1)

    train_x = x[:i_train]
    train_y = y[:i_train]
    val_x = x[i_train:i_val]
    val_y = y[i_train:i_val]
    test_x = x[i_val:]
    test_y = y[i_val:]

    mean = train_x.mean(dim=(0, 1))
    std = train_x.std(dim=(0, 1)).clamp(min=1e-4)

    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    return CursorDatasetTensors(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        feature_mean=mean,
        feature_std=std,
    )


def build_from_csv(
    csv_path: Path,
    sequence_length: int = 30,
    horizon_start: int = 15,
    horizon_points: int = 5,
) -> CursorDatasetTensors:
    t_ms, x, y = load_mouse_csv(csv_path)
    t_new, x_new, y_new = resample_to_hz(t_ms=t_ms, x=x, y=y, hz=60.0)
    feats = build_features(t_ms=t_new, x=x_new, y=y_new)
    win_x, win_y = make_windows(
        feats,
        sequence_length=sequence_length,
        horizon_start=horizon_start,
        horizon_points=horizon_points,
    )
    return split_tensors(torch.from_numpy(win_x), torch.from_numpy(win_y))
