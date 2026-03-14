from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import torch

from chat_api.config import Settings, get_settings
from organic_cursor.infer import CursorPredictorBundle
from silicon_synapse import default_device

logger = logging.getLogger(__name__)


def _coerce_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _coerce_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    if minimum is not None:
        out = max(minimum, out)
    return out


@dataclass
class _PendingPrediction:
    base_sample_index: int
    k_250: int
    gru_path: torch.Tensor  # [P, 2], normalized [0,1]
    liquid_path: torch.Tensor  # [P, 2], normalized [0,1]
    kalman_path: torch.Tensor  # [P, 2], normalized [0,1]
    abg_path: torch.Tensor  # [P, 2], normalized [0,1]
    sum_gru_px: float = 0.0
    sum_liquid_px: float = 0.0
    sum_kalman_px: float = 0.0
    sum_abg_px: float = 0.0
    points_scored: int = 0


@dataclass
class _KalmanAxis:
    pos: float = 0.0
    vel: float = 0.0
    p00: float = 1.0
    p01: float = 0.0
    p10: float = 0.0
    p11: float = 1.0
    initialized: bool = False
    innov_ema: float = 0.0


class _AdaptiveKalmanCV:
    def __init__(self) -> None:
        self._x = _KalmanAxis()
        self._y = _KalmanAxis()
        self._base_process_var = 0.9
        self._base_measure_var = 1.8e-5

    def update_and_forecast(self, x_norm: float, y_norm: float, dt_s: float, surprise: float, h0: int, points: int) -> torch.Tensor:
        dt = max(1.0 / 240.0, min(float(dt_s), 0.2))
        self._update_axis(self._x, x_norm, dt, surprise)
        self._update_axis(self._y, y_norm, dt, surprise)
        return self._forecast_path(dt, surprise, h0, points)

    def _update_axis(self, axis: _KalmanAxis, z: float, dt: float, surprise: float) -> None:
        if not axis.initialized:
            axis.pos = float(z)
            axis.vel = 0.0
            axis.p00 = 2e-3
            axis.p01 = 0.0
            axis.p10 = 0.0
            axis.p11 = 0.15
            axis.initialized = True
            return

        q_scale = 1.0 + min(9.0, surprise * 2.8 + axis.innov_ema * 120.0)
        q = self._base_process_var * q_scale
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q00 = 0.25 * dt4 * q
        q01 = 0.5 * dt3 * q
        q11 = dt2 * q

        pos_pred = axis.pos + dt * axis.vel
        vel_pred = axis.vel
        p00 = axis.p00 + dt * (axis.p10 + axis.p01) + dt2 * axis.p11 + q00
        p01 = axis.p01 + dt * axis.p11 + q01
        p10 = axis.p10 + dt * axis.p11 + q01
        p11 = axis.p11 + q11

        innov = float(z - pos_pred)
        axis.innov_ema = 0.92 * axis.innov_ema + 0.08 * abs(innov)
        r_scale = 1.0 + min(6.0, surprise * 1.6 + axis.innov_ema * 80.0)
        r = self._base_measure_var / max(0.2, r_scale)

        s = p00 + r
        k0 = p00 / max(s, 1e-9)
        k1 = p10 / max(s, 1e-9)

        axis.pos = pos_pred + k0 * innov
        axis.vel = vel_pred + k1 * innov
        axis.vel = max(-8.0, min(8.0, axis.vel))
        axis.p00 = (1.0 - k0) * p00
        axis.p01 = (1.0 - k0) * p01
        axis.p10 = p10 - k1 * p00
        axis.p11 = p11 - k1 * p01

    def _forecast_path(self, dt: float, surprise: float, h0: int, points: int) -> torch.Tensor:
        steps_total = h0 + points
        px = self._x.pos
        py = self._y.pos
        vx = self._x.vel
        vy = self._y.vel
        drag = 0.997 - min(0.05, surprise * 0.04)
        samples: list[list[float]] = []
        for step in range(steps_total):
            px += vx * dt
            py += vy * dt
            vx *= drag
            vy *= drag
            if step >= h0:
                samples.append([max(0.0, min(1.0, px)), max(0.0, min(1.0, py))])
        return torch.tensor(samples, dtype=torch.float32)


class _AdaptiveABG:
    def __init__(self) -> None:
        self._initialized = False
        self._x = 0.0
        self._y = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._ax = 0.0
        self._ay = 0.0
        self._residual_ema = 0.0

    def update_and_forecast(self, x_norm: float, y_norm: float, dt_s: float, surprise: float, h0: int, points: int) -> torch.Tensor:
        dt = max(1.0 / 240.0, min(float(dt_s), 0.2))
        if not self._initialized:
            self._x = float(x_norm)
            self._y = float(y_norm)
            self._initialized = True
        self._update_axis(meas=float(x_norm), dt=dt, surprise=surprise, axis="x")
        self._update_axis(meas=float(y_norm), dt=dt, surprise=surprise, axis="y")
        return self._forecast_path(dt=dt, h0=h0, points=points)

    def _update_axis(self, meas: float, dt: float, surprise: float, axis: str) -> None:
        if axis == "x":
            pos = self._x
            vel = self._vx
            acc = self._ax
        else:
            pos = self._y
            vel = self._vy
            acc = self._ay

        pred_pos = pos + vel * dt + 0.5 * acc * dt * dt
        pred_vel = vel + acc * dt
        residual = meas - pred_pos
        self._residual_ema = 0.9 * self._residual_ema + 0.1 * abs(residual)

        adapt = min(1.0, max(0.0, self._residual_ema * 220.0 + min(1.0, surprise * 0.16)))
        alpha = 0.24 + 0.58 * adapt
        beta = 0.018 + 0.24 * adapt
        gamma = 0.0006 + 0.013 * adapt
        inv_dt = 1.0 / max(dt, 1e-4)
        inv_dt2 = inv_dt * inv_dt

        pos = pred_pos + alpha * residual
        vel = pred_vel + beta * inv_dt * residual
        acc = acc + 2.0 * gamma * inv_dt2 * residual
        vel = max(-8.0, min(8.0, vel))
        acc = max(-120.0, min(120.0, acc))

        if axis == "x":
            self._x = pos
            self._vx = vel
            self._ax = acc
        else:
            self._y = pos
            self._vy = vel
            self._ay = acc

    def _forecast_path(self, dt: float, h0: int, points: int) -> torch.Tensor:
        steps_total = h0 + points
        px = self._x
        py = self._y
        vx = self._vx
        vy = self._vy
        ax = self._ax
        ay = self._ay
        samples: list[list[float]] = []
        for step in range(steps_total):
            px += vx * dt + 0.5 * ax * dt * dt
            py += vy * dt + 0.5 * ay * dt * dt
            vx += ax * dt
            vy += ay * dt
            ax *= 0.95
            ay *= 0.95
            vx *= 0.998
            vy *= 0.998
            if step >= h0:
                samples.append([max(0.0, min(1.0, px)), max(0.0, min(1.0, py))])
        return torch.tensor(samples, dtype=torch.float32)


class _CursorSession:
    def __init__(self, settings: Settings, predictor: CursorPredictorBundle) -> None:
        self.settings = settings
        self.predictor = predictor

        self.sample_index = -1
        self._last_t_ms: int | None = None
        self._last_x_norm: float | None = None
        self._last_y_norm: float | None = None
        self._dt_sample_ms_ema = 1000.0 / 60.0

        self._features: deque[list[float]] = deque(maxlen=settings.cursor_sequence_length)
        self._pending: deque[_PendingPrediction] = deque()

        self._last_gru_path = torch.zeros(settings.cursor_predict_points, 2, dtype=torch.float32)
        self._last_liquid_path = torch.zeros(settings.cursor_predict_points, 2, dtype=torch.float32)
        self._last_kalman_path = torch.zeros(settings.cursor_predict_points, 2, dtype=torch.float32)
        self._last_abg_path = torch.zeros(settings.cursor_predict_points, 2, dtype=torch.float32)
        self._gru_mse_ema = 0.0
        self._liquid_mse_ema = 0.0
        self._mse_initialized = False

        self._fde250_gru_px_ema: float | None = None
        self._fde250_liquid_px_ema: float | None = None
        self._fde250_kalman_px_ema: float | None = None
        self._fde250_abg_px_ema: float | None = None
        self._ade_gru_px_ema: float | None = None
        self._ade_liquid_px_ema: float | None = None
        self._ade_kalman_px_ema: float | None = None
        self._ade_abg_px_ema: float | None = None
        self._fde_final_gru_px_ema: float | None = None
        self._fde_final_liquid_px_ema: float | None = None
        self._fde_final_kalman_px_ema: float | None = None
        self._fde_final_abg_px_ema: float | None = None

        self._infer_gru_ms_ema: float | None = None
        self._infer_liquid_ms_ema: float | None = None
        self._infer_total_ms_ema = 0.0
        self._infer_total_initialized = False

        self._surprise_ema = 0.0
        self._vel_ema = torch.zeros(2, dtype=torch.float32)
        self._acc_ema = torch.zeros(2, dtype=torch.float32)
        self._gru_vel_state = torch.zeros(2, dtype=torch.float32)
        self._liquid_vel_state = torch.zeros(2, dtype=torch.float32)
        self._last_velocity = torch.zeros(2, dtype=torch.float32)
        self._kalman_baseline = _AdaptiveKalmanCV()
        self._abg_baseline = _AdaptiveABG()

    def _flat_path(self, x_norm: float, y_norm: float) -> torch.Tensor:
        points = self.settings.cursor_predict_points
        base = torch.tensor([x_norm, y_norm], dtype=torch.float32).unsqueeze(0)
        return base.repeat(points, 1).clamp(0.0, 1.0)

    def _coerce_path(self, path: Any, fallback: torch.Tensor) -> torch.Tensor:
        if not isinstance(path, torch.Tensor):
            return fallback
        out = path.detach().float().cpu()
        points = self.settings.cursor_predict_points
        if out.ndim != 2 or out.shape[0] != points or out.shape[1] != 2:
            return fallback
        return out.clamp(0.0, 1.0)

    def ingest(self, frame: dict[str, Any]) -> dict[str, Any]:
        w = _coerce_int(frame.get("viewport_w", 1), 1, minimum=1)
        h = _coerce_int(frame.get("viewport_h", 1), 1, minimum=1)
        fallback_x_raw = float(self._last_x_norm * float(w)) if self._last_x_norm is not None else float(w) * 0.5
        fallback_y_raw = float(self._last_y_norm * float(h)) if self._last_y_norm is not None else float(h) * 0.5
        x_raw = _coerce_float(frame.get("x", fallback_x_raw), fallback_x_raw)
        y_raw = _coerce_float(frame.get("y", fallback_y_raw), fallback_y_raw)
        t_ms = _coerce_int(frame.get("t_ms", int(time.time() * 1000)), int(time.time() * 1000), minimum=0)

        incoming_sample_index = frame.get("sample_index")
        if isinstance(incoming_sample_index, (int, float)) and not isinstance(incoming_sample_index, bool):
            sample_val = float(incoming_sample_index)
            candidate = int(sample_val) if math.isfinite(sample_val) else self.sample_index + 1
            if candidate > self.sample_index:
                self.sample_index = candidate
            else:
                self.sample_index += 1
        else:
            self.sample_index += 1

        x_norm = min(max(x_raw / float(w), 0.0), 1.0)
        y_norm = min(max(y_raw / float(h), 0.0), 1.0)

        dt_s = 1.0 / 60.0
        if self._last_t_ms is not None:
            dt_ms = max(1000.0 / 240.0, min(float(t_ms - self._last_t_ms), 200.0))
            self._dt_sample_ms_ema = 0.88 * self._dt_sample_ms_ema + 0.12 * dt_ms
            dt_s = dt_ms / 1000.0

        if self._last_x_norm is None or self._last_y_norm is None:
            vx = 0.0
            vy = 0.0
        else:
            vx = (x_norm - self._last_x_norm) / dt_s
            vy = (y_norm - self._last_y_norm) / dt_s

        self._last_t_ms = t_ms
        self._last_x_norm = x_norm
        self._last_y_norm = y_norm

        vx = float(max(min(vx, 6.0), -6.0))
        vy = float(max(min(vy, 6.0), -6.0))
        self._features.append([x_norm, y_norm, vx, vy])
        actual_xy = torch.tensor([x_norm, y_norm], dtype=torch.float32)
        try:
            self._update_error_metrics(actual_xy=actual_xy, w=w, h=h)
        except Exception:
            logger.exception("Cursor scoring update failed; clearing pending predictions")
            self._pending.clear()

        ready = len(self._features) >= self.settings.cursor_sequence_length
        infer_now = ready and (self.sample_index % max(1, self.settings.cursor_infer_every_n_frames) == 0)
        fallback_path = self._flat_path(x_norm=x_norm, y_norm=y_norm)
        heur_gru_path = fallback_path
        heur_liquid_path = fallback_path
        try:
            heur_gru_path, heur_liquid_path = self._heuristic_paths(
                x_norm=x_norm,
                y_norm=y_norm,
                vx=vx,
                vy=vy,
                dt_s=dt_s,
            )
        except Exception:
            logger.exception("Cursor heuristic path update failed; using flat fallback path")
        heur_gru_path = self._coerce_path(heur_gru_path, fallback_path)
        heur_liquid_path = self._coerce_path(heur_liquid_path, fallback_path)

        h0 = self.settings.cursor_predict_horizon_start
        points = self.settings.cursor_predict_points
        kalman_path = heur_liquid_path
        try:
            kalman_path = self._kalman_baseline.update_and_forecast(
                x_norm=x_norm,
                y_norm=y_norm,
                dt_s=dt_s,
                surprise=self._surprise_ema,
                h0=h0,
                points=points,
            )
        except Exception:
            logger.exception("Kalman baseline update failed; falling back to adaptive heuristic path")
        kalman_path = self._coerce_path(kalman_path, heur_liquid_path)

        abg_path = heur_gru_path
        try:
            abg_path = self._abg_baseline.update_and_forecast(
                x_norm=x_norm,
                y_norm=y_norm,
                dt_s=dt_s,
                surprise=self._surprise_ema,
                h0=h0,
                points=points,
            )
        except Exception:
            logger.exception("ABG baseline update failed; falling back to adaptive heuristic path")
        abg_path = self._coerce_path(abg_path, heur_gru_path)

        next_gru_path = heur_gru_path
        next_liquid_path = heur_liquid_path

        if infer_now and self.predictor.has_any_weights:
            seq = torch.tensor(list(self._features), dtype=torch.float32).unsqueeze(0)
            try:
                pred = self.predictor.predict(seq)
            except Exception:
                logger.exception("Model inference failed; using adaptive-only fallback for this frame")
            else:
                self._update_infer_timings(infer_gru_ms=pred.infer_gru_ms, infer_liquid_ms=pred.infer_liquid_ms)
                if pred.gru_path is not None:
                    model_gru_path = self._coerce_path(pred.gru_path, heur_gru_path)
                    next_gru_path = self._blend_paths(primary=model_gru_path, secondary=heur_gru_path, primary_weight=0.72)
                if pred.liquid_path is not None:
                    model_liquid_path = self._coerce_path(pred.liquid_path, heur_liquid_path)
                    # Keep stronger adaptive heuristic influence to react faster to abrupt shifts.
                    next_liquid_path = self._blend_paths(
                        primary=model_liquid_path,
                        secondary=heur_liquid_path,
                        primary_weight=0.62,
                    )

        debug_gru_bias_px = 0.0
        raw_bias = frame.get("debug_gru_bias_px")
        if raw_bias is not None:
            debug_gru_bias_px = max(-400.0, min(400.0, _coerce_float(raw_bias, 0.0)))
        if self.settings.cursor_dev_controls and abs(debug_gru_bias_px) > 1e-6:
            next_gru_path = self._apply_x_bias_px(next_gru_path, w=w, bias_px=debug_gru_bias_px)
        else:
            debug_gru_bias_px = 0.0

        smooth_alpha = 0.38 if infer_now else 0.24
        self._last_gru_path = self._smooth_path(self._last_gru_path, next_gru_path, alpha=smooth_alpha)
        self._last_liquid_path = self._smooth_path(self._last_liquid_path, next_liquid_path, alpha=smooth_alpha)
        self._last_kalman_path = self._smooth_path(self._last_kalman_path, kalman_path, alpha=0.44)
        self._last_abg_path = self._smooth_path(self._last_abg_path, abg_path, alpha=0.44)
        self._pending.append(
            _PendingPrediction(
                base_sample_index=self.sample_index,
                k_250=self._compute_k_250(),
                gru_path=self._last_gru_path,
                liquid_path=self._last_liquid_path,
                kalman_path=self._last_kalman_path,
                abg_path=self._last_abg_path,
            )
        )

        gru_points = self._to_points_px(self._last_gru_path, w=w, h=h)
        liquid_points = self._to_points_px(self._last_liquid_path, w=w, h=h)
        kalman_points = self._to_points_px(self._last_kalman_path, w=w, h=h)
        abg_points = self._to_points_px(self._last_abg_path, w=w, h=h)
        gru_lead_px = self._lead_distance(gru_points[0], x_raw=x_raw, y_raw=y_raw) if gru_points else 0.0
        liquid_lead_px = self._lead_distance(liquid_points[0], x_raw=x_raw, y_raw=y_raw) if liquid_points else 0.0
        mode = "model+adaptive" if self.predictor.has_any_weights else "adaptive-only"
        return {
            "type": "pred",
            "sample_index": int(self.sample_index),
            "gru": gru_points[0] if gru_points else {"x": x_raw, "y": y_raw},
            "liquid": liquid_points[0] if liquid_points else {"x": x_raw, "y": y_raw},
            "kalman": kalman_points[0] if kalman_points else {"x": x_raw, "y": y_raw},
            "abg": abg_points[0] if abg_points else {"x": x_raw, "y": y_raw},
            "gru_path": gru_points,
            "liquid_path": liquid_points,
            "kalman_path": kalman_points,
            "abg_path": abg_points,
            "metrics": {
                "gru_mse_ema": float(self._gru_mse_ema),
                "liquid_mse_ema": float(self._liquid_mse_ema),
                "rtt_ms": float(self._infer_total_ms_ema),
                "gru_lead_px": float(gru_lead_px),
                "liquid_lead_px": float(liquid_lead_px),
                "mse_advantage": float(self._gru_mse_ema - self._liquid_mse_ema),
                "mode": mode,
                "infer_gru_ms_ema": self._infer_gru_ms_ema,
                "infer_liquid_ms_ema": self._infer_liquid_ms_ema,
                "fde250_gru_px_ema": self._metric_or_zero(self._fde250_gru_px_ema),
                "fde250_liquid_px_ema": self._metric_or_zero(self._fde250_liquid_px_ema),
                "fde250_kalman_px_ema": self._metric_or_zero(self._fde250_kalman_px_ema),
                "fde250_abg_px_ema": self._metric_or_zero(self._fde250_abg_px_ema),
                "ade_gru_px_ema": self._metric_or_zero(self._ade_gru_px_ema),
                "ade_liquid_px_ema": self._metric_or_zero(self._ade_liquid_px_ema),
                "ade_kalman_px_ema": self._metric_or_zero(self._ade_kalman_px_ema),
                "ade_abg_px_ema": self._metric_or_zero(self._ade_abg_px_ema),
                "fde_final_gru_px_ema": self._metric_or_zero(self._fde_final_gru_px_ema),
                "fde_final_liquid_px_ema": self._metric_or_zero(self._fde_final_liquid_px_ema),
                "fde_final_kalman_px_ema": self._metric_or_zero(self._fde_final_kalman_px_ema),
                "fde_final_abg_px_ema": self._metric_or_zero(self._fde_final_abg_px_ema),
                "dt_sample_ms_ema": float(self._dt_sample_ms_ema),
                "horizon_start": int(self.settings.cursor_predict_horizon_start),
                "predict_points": int(self.settings.cursor_predict_points),
                "dev_controls_enabled": bool(self.settings.cursor_dev_controls),
                "debug_gru_bias_px": float(debug_gru_bias_px),
            },
            "buffer_ready": bool(ready),
        }

    def _heuristic_paths(self, x_norm: float, y_norm: float, vx: float, vy: float, dt_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        vel = torch.tensor([vx, vy], dtype=torch.float32)
        pos = torch.tensor([x_norm, y_norm], dtype=torch.float32)

        vel_delta = vel - self._last_velocity
        self._last_velocity = vel
        self._vel_ema = 0.84 * self._vel_ema + 0.16 * vel
        acc_est = vel_delta / max(dt_s, 1e-3)
        self._acc_ema = 0.9 * self._acc_ema + 0.1 * acc_est

        surprise = float((vel - self._vel_ema).abs().mean().item())
        self._surprise_ema = 0.94 * self._surprise_ema + 0.06 * surprise
        adapt_gain = min(0.95, max(0.12, 0.2 + 0.35 * self._surprise_ema))

        self._gru_vel_state = 0.96 * self._gru_vel_state + 0.04 * vel
        self._liquid_vel_state = (1.0 - adapt_gain) * self._liquid_vel_state + adapt_gain * vel

        h0 = self.settings.cursor_predict_horizon_start
        points = self.settings.cursor_predict_points
        steps_total = h0 + points
        sim_dt = 1.0 / 60.0

        grp = pos.clone()
        lqp = pos.clone()
        grv = self._gru_vel_state.clone()
        lqv = self._liquid_vel_state.clone()
        acc = self._acc_ema.clamp(-22.0, 22.0)

        gru_samples: list[torch.Tensor] = []
        liquid_samples: list[torch.Tensor] = []
        for step in range(steps_total):
            grv = grv * 0.985 + acc * sim_dt * 0.015
            lq_drag = 0.95 - min(0.22, self._surprise_ema * 0.25)
            lqv = lqv * lq_drag + acc * sim_dt * (0.045 + adapt_gain * 0.04)
            # Liquid path recenters quickly after abrupt direction changes.
            lqv = lqv + (vel - lqv) * (adapt_gain * 0.09)

            grp = grp + grv * sim_dt
            lqp = lqp + lqv * sim_dt

            if step >= h0:
                gru_samples.append(grp.clone())
                liquid_samples.append(lqp.clone())

        gru_path = torch.stack(gru_samples, dim=0).clamp(0.0, 1.0)
        liquid_path = torch.stack(liquid_samples, dim=0).clamp(0.0, 1.0)
        return gru_path, liquid_path

    def _update_infer_timings(self, infer_gru_ms: float | None, infer_liquid_ms: float | None) -> None:
        if infer_gru_ms is not None:
            self._infer_gru_ms_ema = self._ema_optional(self._infer_gru_ms_ema, infer_gru_ms, beta=0.86)
        if infer_liquid_ms is not None:
            self._infer_liquid_ms_ema = self._ema_optional(self._infer_liquid_ms_ema, infer_liquid_ms, beta=0.86)
        if infer_gru_ms is None and infer_liquid_ms is None:
            return
        infer_total = float(infer_gru_ms or 0.0) + float(infer_liquid_ms or 0.0)
        if not self._infer_total_initialized:
            self._infer_total_ms_ema = infer_total
            self._infer_total_initialized = True
            return
        beta = 0.86
        self._infer_total_ms_ema = beta * self._infer_total_ms_ema + (1.0 - beta) * infer_total

    def _compute_k_250(self) -> int:
        h0 = self.settings.cursor_predict_horizon_start
        points = self.settings.cursor_predict_points
        offset_250 = int(round(250.0 / max(self._dt_sample_ms_ema, 1e-3)))
        return int(min(points - 1, max(0, offset_250 - h0)))

    def _apply_x_bias_px(self, path: torch.Tensor, w: int, bias_px: float) -> torch.Tensor:
        out = path.clone()
        width = max(float(w), 1.0)
        out[:, 0] = ((out[:, 0] * width) + bias_px) / width
        return out.clamp(0.0, 1.0)

    def _blend_paths(self, primary: torch.Tensor, secondary: torch.Tensor, primary_weight: float) -> torch.Tensor:
        w = float(min(0.98, max(0.02, primary_weight)))
        return (w * primary + (1.0 - w) * secondary).clamp(0.0, 1.0)

    def _smooth_path(self, previous: torch.Tensor, current: torch.Tensor, alpha: float) -> torch.Tensor:
        a = float(min(0.98, max(0.02, alpha)))
        return (1.0 - a) * previous + a * current

    def _lead_distance(self, lead: dict[str, float], x_raw: float, y_raw: float) -> float:
        dx = float(lead["x"]) - float(x_raw)
        dy = float(lead["y"]) - float(y_raw)
        return math.sqrt(dx * dx + dy * dy)

    def _update_error_metrics(self, actual_xy: torch.Tensor, w: int, h: int) -> None:
        h0 = self.settings.cursor_predict_horizon_start
        points = self.settings.cursor_predict_points
        actual_x = float(actual_xy[0].item())
        actual_y = float(actual_xy[1].item())
        width = float(w)
        height = float(h)

        keep: deque[_PendingPrediction] = deque()
        for pending in self._pending:
            k = self.sample_index - (pending.base_sample_index + h0)
            if k < 0:
                keep.append(pending)
                continue
            if k >= points:
                continue

            gru_x = float(pending.gru_path[k, 0].item())
            gru_y = float(pending.gru_path[k, 1].item())
            liquid_x = float(pending.liquid_path[k, 0].item())
            liquid_y = float(pending.liquid_path[k, 1].item())
            kalman_x = float(pending.kalman_path[k, 0].item())
            kalman_y = float(pending.kalman_path[k, 1].item())
            abg_x = float(pending.abg_path[k, 0].item())
            abg_y = float(pending.abg_path[k, 1].item())

            gru_dx_px = (gru_x - actual_x) * width
            gru_dy_px = (gru_y - actual_y) * height
            liquid_dx_px = (liquid_x - actual_x) * width
            liquid_dy_px = (liquid_y - actual_y) * height
            kalman_dx_px = (kalman_x - actual_x) * width
            kalman_dy_px = (kalman_y - actual_y) * height
            abg_dx_px = (abg_x - actual_x) * width
            abg_dy_px = (abg_y - actual_y) * height
            gru_dist_px = math.hypot(gru_dx_px, gru_dy_px)
            liquid_dist_px = math.hypot(liquid_dx_px, liquid_dy_px)
            kalman_dist_px = math.hypot(kalman_dx_px, kalman_dy_px)
            abg_dist_px = math.hypot(abg_dx_px, abg_dy_px)

            pending.sum_gru_px += gru_dist_px
            pending.sum_liquid_px += liquid_dist_px
            pending.sum_kalman_px += kalman_dist_px
            pending.sum_abg_px += abg_dist_px
            pending.points_scored += 1

            gru_err_norm = float(torch.mean((pending.gru_path[k] - actual_xy) ** 2).item())
            liquid_err_norm = float(torch.mean((pending.liquid_path[k] - actual_xy) ** 2).item())
            if not self._mse_initialized:
                self._gru_mse_ema = gru_err_norm
                self._liquid_mse_ema = liquid_err_norm
                self._mse_initialized = True
            else:
                beta = 0.92
                self._gru_mse_ema = beta * self._gru_mse_ema + (1.0 - beta) * gru_err_norm
                self._liquid_mse_ema = beta * self._liquid_mse_ema + (1.0 - beta) * liquid_err_norm

            if k == pending.k_250:
                self._fde250_gru_px_ema = self._ema_optional(self._fde250_gru_px_ema, gru_dist_px)
                self._fde250_liquid_px_ema = self._ema_optional(self._fde250_liquid_px_ema, liquid_dist_px)
                self._fde250_kalman_px_ema = self._ema_optional(self._fde250_kalman_px_ema, kalman_dist_px)
                self._fde250_abg_px_ema = self._ema_optional(self._fde250_abg_px_ema, abg_dist_px)

            if k == points - 1:
                self._fde_final_gru_px_ema = self._ema_optional(self._fde_final_gru_px_ema, gru_dist_px)
                self._fde_final_liquid_px_ema = self._ema_optional(self._fde_final_liquid_px_ema, liquid_dist_px)
                self._fde_final_kalman_px_ema = self._ema_optional(self._fde_final_kalman_px_ema, kalman_dist_px)
                self._fde_final_abg_px_ema = self._ema_optional(self._fde_final_abg_px_ema, abg_dist_px)
                self._ade_gru_px_ema = self._ema_optional(self._ade_gru_px_ema, pending.sum_gru_px / float(points))
                self._ade_liquid_px_ema = self._ema_optional(self._ade_liquid_px_ema, pending.sum_liquid_px / float(points))
                self._ade_kalman_px_ema = self._ema_optional(self._ade_kalman_px_ema, pending.sum_kalman_px / float(points))
                self._ade_abg_px_ema = self._ema_optional(self._ade_abg_px_ema, pending.sum_abg_px / float(points))
                continue

            keep.append(pending)
        self._pending = keep

    def _metric_or_zero(self, value: float | None) -> float:
        return float(value) if value is not None else 0.0

    def _ema_optional(self, current: float | None, value: float, beta: float = 0.92) -> float:
        if current is None:
            return float(value)
        return float(beta * current + (1.0 - beta) * value)

    def _to_points_px(self, path: torch.Tensor, w: int, h: int) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for point in path:
            px = float(point[0].item()) * float(w)
            py = float(point[1].item()) * float(h)
            out.append({"x": px, "y": py})
        return out


class CursorRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = default_device()
        self.predictor = CursorPredictorBundle(settings=settings, device=self.device)

    def load_models(self) -> None:
        self.predictor.load_or_init()

    async def _send_error(self, ws: WebSocket, message: str) -> None:
        try:
            await ws.send_json({"type": "error", "message": message})
        except Exception:
            return

    async def handle_connection(self, ws: WebSocket) -> None:
        session = _CursorSession(settings=self.settings, predictor=self.predictor)
        consecutive_ingest_errors = 0
        while True:
            try:
                payload = await ws.receive_json()
            except WebSocketDisconnect:
                return
            except Exception:
                logger.exception("Cursor websocket receive_json failed")
                await self._send_error(ws, "invalid payload")
                continue

            if not isinstance(payload, dict) or payload.get("type") != "frame":
                await self._send_error(ws, "unsupported payload type")
                continue

            try:
                out = session.ingest(payload)
            except Exception:
                consecutive_ingest_errors += 1
                logger.exception(
                    "Cursor frame ingest failed (consecutive=%d)",
                    consecutive_ingest_errors,
                )
                await self._send_error(ws, "frame processing failed")
                if consecutive_ingest_errors >= 3:
                    logger.warning("Resetting cursor session after repeated ingest failures")
                    session = _CursorSession(settings=self.settings, predictor=self.predictor)
                    consecutive_ingest_errors = 0
                continue

            consecutive_ingest_errors = 0
            try:
                await ws.send_json(out)
            except WebSocketDisconnect:
                return
            except Exception:
                logger.exception("Cursor websocket send_json failed")
                return


def build_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    runtime = CursorRuntime(settings=cfg)
    runtime.load_models()
    app = FastAPI(title="Organic Cursor Service", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/ws/cursor")
    async def ws_cursor(ws: WebSocket) -> None:
        await ws.accept()
        try:
            await runtime.handle_connection(ws)
        except WebSocketDisconnect:
            return
        except Exception:
            logger.exception("Unhandled cursor websocket error")
            return

    static_dir = cfg.cursor_weights_dir.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


app = build_app()
