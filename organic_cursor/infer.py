from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from chat_api.config import Settings
from organic_cursor.models import CursorGRU, CursorLiquid


@dataclass
class CursorPredictions:
    gru_path: torch.Tensor | None  # [5,2]
    liquid_path: torch.Tensor | None  # [5,2]
    infer_gru_ms: float | None
    infer_liquid_ms: float | None


class CursorPredictorBundle:
    def __init__(self, settings: Settings, device: torch.device) -> None:
        self.settings = settings
        self.device = device
        self.sequence_length = settings.cursor_sequence_length
        self.future_points = settings.cursor_predict_points

        self.gru = CursorGRU(input_dim=4, hidden_dim=settings.cursor_hidden_dim, future_points=self.future_points).to(device)
        self.liquid = CursorLiquid(input_dim=4, future_points=self.future_points).to(device)
        self.feature_mean = torch.zeros(4, device=device)
        self.feature_std = torch.ones(4, device=device)
        self.has_gru_weights = False
        self.has_liquid_weights = False

    def load_or_init(self) -> None:
        weights_dir = self.settings.cursor_weights_dir
        gru_path = weights_dir / self.settings.cursor_gru_weights
        liquid_path = weights_dir / self.settings.cursor_liquid_weights

        if gru_path.exists():
            payload = torch.load(gru_path, map_location=self.device)
            self.gru.load_state_dict(payload["model_state"])
            self.feature_mean = payload.get("feature_mean", self.feature_mean).to(self.device)
            self.feature_std = payload.get("feature_std", self.feature_std).to(self.device)
            self.has_gru_weights = True
        if liquid_path.exists():
            payload = torch.load(liquid_path, map_location=self.device)
            self.liquid.load_state_dict(payload["model_state"])
            # Keep shared normalization from whichever checkpoint has it.
            self.feature_mean = payload.get("feature_mean", self.feature_mean).to(self.device)
            self.feature_std = payload.get("feature_std", self.feature_std).to(self.device)
            self.has_liquid_weights = True

        self.gru.eval()
        self.liquid.eval()

    @property
    def has_any_weights(self) -> bool:
        return bool(self.has_gru_weights or self.has_liquid_weights)

    @torch.no_grad()
    def predict(self, seq: torch.Tensor) -> CursorPredictions:
        # seq: [1, T, 4]
        seq = seq.to(self.device)
        seq = (seq - self.feature_mean.view(1, 1, -1)) / self.feature_std.view(1, 1, -1).clamp(min=1e-4)

        gru: torch.Tensor | None = None
        liquid: torch.Tensor | None = None
        infer_gru_ms: float | None = None
        infer_liquid_ms: float | None = None

        if self.has_gru_weights:
            started = time.perf_counter()
            gru = self.gru(seq)[0]
            infer_gru_ms = (time.perf_counter() - started) * 1000.0

        if self.has_liquid_weights:
            started = time.perf_counter()
            liquid = self.liquid(seq)[0]
            infer_liquid_ms = (time.perf_counter() - started) * 1000.0

        return CursorPredictions(
            gru_path=gru,
            liquid_path=liquid,
            infer_gru_ms=infer_gru_ms,
            infer_liquid_ms=infer_liquid_ms,
        )
