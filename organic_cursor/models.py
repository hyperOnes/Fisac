from __future__ import annotations

import torch
import torch.nn as nn

from silicon_synapse import SiliconSynapse


class CursorGRU(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, future_points: int = 5) -> None:
        super().__init__()
        self.future_points = future_points
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_points * 2),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, 4]
        seq_out, _ = self.gru(x_seq)
        last = seq_out[:, -1, :]
        pred = self.head(last)
        # Predict normalized screen coordinates in [0, 1].
        return torch.sigmoid(pred.view(x_seq.size(0), self.future_points, 2))


class CursorLiquid(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        future_points: int = 5,
        num_experts: int = 32,
        top_k: int = 2,
        dt: float = 0.02,
    ) -> None:
        super().__init__()
        self.future_points = future_points
        self.dt = dt
        self.core = SiliconSynapse(
            feature_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            dt_default=dt,
            sleep_interval=10_000,
            fuse_window=100_000,
        )
        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, future_points * 2),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, 4]
        batch, seq_len, _ = x_seq.shape
        state = self.core.init_state(batch_size=batch, device=x_seq.device)
        y_t = x_seq[:, -1, :]
        for t in range(seq_len):
            y_t, state, _ = self.core.forward(x_t=x_seq[:, t, :], state=state, dt=self.dt, train_mode=self.training)
        pred = self.head(y_t)
        # Predict normalized screen coordinates in [0, 1].
        return torch.sigmoid(pred.view(batch, self.future_points, 2))
