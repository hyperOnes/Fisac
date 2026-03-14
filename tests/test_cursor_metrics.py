from __future__ import annotations

from collections import deque

import torch

from chat_api.config import Settings
from organic_cursor.app import _CursorSession, _PendingPrediction


class _NoWeightPredictor:
    has_any_weights = False
    has_gru_weights = False
    has_liquid_weights = False

    def predict(self, seq: torch.Tensor) -> None:  # pragma: no cover - never called in these tests
        raise AssertionError("predict should not be called in no-weight mode")


class _FailingPredictor:
    has_any_weights = True
    has_gru_weights = True
    has_liquid_weights = True

    def predict(self, seq: torch.Tensor) -> None:
        raise RuntimeError("synthetic inference failure")


class _BrokenFilter:
    def update_and_forecast(self, **kwargs) -> torch.Tensor:
        raise RuntimeError("synthetic baseline failure")


def test_compute_k250_from_dt_ema() -> None:
    settings = Settings(cursor_predict_horizon_start=15, cursor_predict_points=5)
    session = _CursorSession(settings=settings, predictor=_NoWeightPredictor())

    session._dt_sample_ms_ema = 16.67
    assert session._compute_k_250() == 0

    session._dt_sample_ms_ema = 14.0
    # round(250 / 14) = 18 -> 18 - 15 = 3
    assert session._compute_k_250() == 3

    session._dt_sample_ms_ema = 22.0
    # round(250 / 22) = 11 -> 11 - 15 = -4 -> clamp to 0
    assert session._compute_k_250() == 0


def test_sample_index_alignment_scores_expected_k() -> None:
    settings = Settings(cursor_predict_horizon_start=2, cursor_predict_points=5)
    session = _CursorSession(settings=settings, predictor=_NoWeightPredictor())

    gru_path = torch.tensor(
        [
            [0.10, 0.10],
            [0.20, 0.20],
            [0.30, 0.30],
            [0.40, 0.40],
            [0.50, 0.50],
        ],
        dtype=torch.float32,
    )
    liquid_path = gru_path.clone()
    liquid_path[:, 0] += 0.01

    pending = _PendingPrediction(
        base_sample_index=10,
        k_250=1,
        gru_path=gru_path,
        liquid_path=liquid_path,
        kalman_path=liquid_path.clone(),
        abg_path=liquid_path.clone(),
    )
    session._pending = deque([pending])

    width = 1000
    height = 500

    # k=-1, should not score yet.
    session.sample_index = 11
    session._update_error_metrics(actual_xy=torch.tensor([0.10, 0.10], dtype=torch.float32), w=width, h=height)
    assert session._mse_initialized is False

    # Score k=0..4 with exact GRU targets.
    for sample_index, point in [(12, 0.10), (13, 0.20), (14, 0.30), (15, 0.40), (16, 0.50)]:
        session.sample_index = sample_index
        session._update_error_metrics(actual_xy=torch.tensor([point, point], dtype=torch.float32), w=width, h=height)

    assert session._mse_initialized is True
    assert len(session._pending) == 0
    assert session._fde250_gru_px_ema is not None
    assert session._fde250_liquid_px_ema is not None
    assert session._fde250_kalman_px_ema is not None
    assert session._fde250_abg_px_ema is not None
    assert session._fde250_gru_px_ema < 1e-5
    assert session._fde250_liquid_px_ema > session._fde250_gru_px_ema
    assert session._fde250_kalman_px_ema > session._fde250_gru_px_ema
    assert session._fde250_abg_px_ema > session._fde250_gru_px_ema
    assert session._ade_gru_px_ema is not None
    assert session._ade_liquid_px_ema is not None
    assert session._ade_kalman_px_ema is not None
    assert session._ade_abg_px_ema is not None
    assert session._ade_liquid_px_ema > session._ade_gru_px_ema
    assert session._ade_kalman_px_ema > session._ade_gru_px_ema
    assert session._ade_abg_px_ema > session._ade_gru_px_ema
    assert session._fde_final_gru_px_ema is not None
    assert session._fde_final_liquid_px_ema is not None
    assert session._fde_final_kalman_px_ema is not None
    assert session._fde_final_abg_px_ema is not None


def _run_session_with_bias(bias_px: float) -> tuple[float, float, float, float]:
    settings = Settings(
        cursor_sequence_length=12,
        cursor_predict_horizon_start=15,
        cursor_predict_points=5,
        cursor_infer_every_n_frames=1,
        cursor_dev_controls=True,
    )
    session = _CursorSession(settings=settings, predictor=_NoWeightPredictor())

    out: dict[str, object] | None = None
    for i in range(90):
        payload: dict[str, object] = {
            "type": "frame",
            "sample_index": i,
            "x": 320.0,
            "y": 220.0,
            "t_ms": 1000 + i * 16,
            "viewport_w": 1440,
            "viewport_h": 900,
        }
        if abs(bias_px) > 1e-6:
            payload["debug_gru_bias_px"] = bias_px
        out = session.ingest(payload)

    assert out is not None
    metrics = out["metrics"]  # type: ignore[index]
    return (
        float(metrics["fde250_gru_px_ema"]),  # type: ignore[index]
        float(metrics["fde250_liquid_px_ema"]),  # type: ignore[index]
        float(metrics["fde250_kalman_px_ema"]),  # type: ignore[index]
        float(metrics["fde250_abg_px_ema"]),  # type: ignore[index]
    )


def test_bias_injection_increases_gru_fde250() -> None:
    base_gru, base_liquid, base_kalman, base_abg = _run_session_with_bias(0.0)
    biased_gru, biased_liquid, biased_kalman, biased_abg = _run_session_with_bias(20.0)

    assert biased_gru > base_gru + 10.0
    assert abs(biased_liquid - base_liquid) < 0.5
    assert abs(biased_kalman - base_kalman) < 0.5
    assert abs(biased_abg - base_abg) < 0.5


def test_ingest_recovers_when_baselines_and_inference_throw() -> None:
    settings = Settings(
        cursor_sequence_length=8,
        cursor_predict_horizon_start=15,
        cursor_predict_points=5,
        cursor_infer_every_n_frames=1,
    )
    session = _CursorSession(settings=settings, predictor=_FailingPredictor())
    session._kalman_baseline = _BrokenFilter()  # type: ignore[assignment]
    session._abg_baseline = _BrokenFilter()  # type: ignore[assignment]

    out: dict[str, object] | None = None
    for i in range(24):
        out = session.ingest(
            {
                "type": "frame",
                "sample_index": i,
                "x": float(320 + i * 3),
                "y": float(240 + i * 2),
                "t_ms": 1000 + i * 16,
                "viewport_w": 1440,
                "viewport_h": 900,
            }
        )

    assert out is not None
    assert out["type"] == "pred"
    assert len(out["gru_path"]) == settings.cursor_predict_points  # type: ignore[index]
    assert len(out["liquid_path"]) == settings.cursor_predict_points  # type: ignore[index]
    assert len(out["kalman_path"]) == settings.cursor_predict_points  # type: ignore[index]
    assert len(out["abg_path"]) == settings.cursor_predict_points  # type: ignore[index]
