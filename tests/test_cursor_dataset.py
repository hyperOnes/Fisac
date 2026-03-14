from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from organic_cursor.dataset import build_features, build_from_csv, make_windows


def test_cursor_window_shapes() -> None:
    features = np.random.randn(100, 4).astype(np.float32)
    x, y = make_windows(features, sequence_length=30, horizon_start=15, horizon_points=5)
    assert x.shape == (52, 30, 4)
    assert y.shape == (52, 5, 2)
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()


def test_build_from_csv_outputs_finite_tensors(tmp_path) -> None:
    csv_path = Path(tmp_path) / "mouse.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("timestamp_ms,x,y,event\n")
        t = 0
        for i in range(1200):
            x = 400 + np.sin(i / 30.0) * 120
            y = 300 + np.cos(i / 40.0) * 90
            fh.write(f"{t},{x:.3f},{y:.3f},move\n")
            t += 16

    ds = build_from_csv(csv_path, sequence_length=30, horizon_start=15, horizon_points=5)
    assert ds.train_x.ndim == 3
    assert ds.train_x.shape[1:] == (30, 4)
    assert ds.train_y.shape[1:] == (5, 2)
    assert torch.isfinite(ds.train_x).all()
    assert torch.isfinite(ds.train_y).all()
    assert torch.isfinite(ds.feature_mean).all()
    assert torch.isfinite(ds.feature_std).all()


def test_build_features_normalization_finite() -> None:
    t = np.arange(0, 1000, 16, dtype=np.float64)
    x = np.linspace(10.0, 500.0, t.size)
    y = np.linspace(20.0, 400.0, t.size)
    feats = build_features(t_ms=t, x=x, y=y)
    assert feats.shape == (t.size, 4)
    assert np.isfinite(feats).all()
