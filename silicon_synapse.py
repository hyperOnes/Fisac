from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TernaryWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        # Per-output-channel scaling keeps quantization stable across rows.
        scale = weight.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        quantized = torch.clamp(torch.round(weight / scale), -1.0, 1.0)
        return quantized * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        # Straight-through estimator.
        return grad_output


class TernaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias: Optional[nn.Parameter]
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def quantized_weight(self) -> torch.Tensor:
        return TernaryWeight.apply(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quantized_weight(), self.bias)


class LiquidNode(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        # softplus(raw_tau) + 1e-3 ensures strictly positive time constants.
        self.raw_tau = nn.Parameter(torch.zeros(feature_dim))
        self.ternary_transform = TernaryLinear(feature_dim, feature_dim, bias=False)

    def tau(self) -> torch.Tensor:
        return F.softplus(self.raw_tau) + 1e-3

    def alpha(self, dt: float) -> torch.Tensor:
        return torch.clamp(torch.as_tensor(dt, device=self.raw_tau.device) / self.tau(), 0.0, 1.0)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        structural_signal = torch.tanh(self.ternary_transform(x) + hidden_state)
        alpha = self.alpha(dt).unsqueeze(0)
        return hidden_state + alpha * (structural_signal - hidden_state)


class FusedMyelinatedNode(nn.Module):
    def __init__(self, liquid_node: LiquidNode, fixed_dt: float = 0.02) -> None:
        super().__init__()
        with torch.no_grad():
            static_weight = liquid_node.ternary_transform.quantized_weight().detach().clone()
            alpha = liquid_node.alpha(fixed_dt).detach().clone()
        self.static_weight = nn.Parameter(static_weight, requires_grad=False)
        self.static_alpha = nn.Parameter(alpha, requires_grad=False)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        del dt
        structural_signal = torch.tanh(F.linear(x, self.static_weight) + hidden_state)
        return hidden_state + self.static_alpha.unsqueeze(0) * (structural_signal - hidden_state)


class LiquidMoERouter(nn.Module):
    def __init__(
        self,
        num_experts: int,
        feature_dim: int,
        top_k: int = 4,
        novelty_scale: float = 1.0,
        usage_penalty_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.novelty_scale = novelty_scale
        self.usage_penalty_scale = usage_penalty_scale
        self.gate = nn.Linear(feature_dim, num_experts)

    def forward(
        self,
        x: torch.Tensor,
        prediction_error_ema: torch.Tensor,
        usage_penalty: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = self.gate(x)
        novelty_boost = self.novelty_scale * prediction_error_ema.unsqueeze(-1) * (1.0 - usage_penalty).unsqueeze(0)
        logits = logits + novelty_boost - self.usage_penalty_scale * usage_penalty.unsqueeze(0)

        dead_mask = ~alive_mask.unsqueeze(0)
        logits = logits.masked_fill(dead_mask, torch.finfo(logits.dtype).min)
        probs = F.softmax(logits, dim=-1)

        alive_count = int(alive_mask.sum().item())
        if alive_count == 0:
            raise RuntimeError("No alive experts available for routing.")
        k = min(self.top_k, alive_count)

        top_weights, top_indices = torch.topk(probs, k=k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return {
            "logits": logits,
            "routing_probs": probs,
            "top_weights": top_weights,
            "top_indices": top_indices,
        }


class ReplayBuffer:
    def __init__(self, capacity: int = 4096, recent_window: int = 1024) -> None:
        self.capacity = capacity
        self.recent_window = recent_window
        self._buffer: Deque[Tuple[torch.Tensor, torch.Tensor, float]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, x: torch.Tensor, y: torch.Tensor, priority: float) -> None:
        self._buffer.append((x.detach().cpu(), y.detach().cpu(), float(priority)))

    def sample(self, size: int, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self._buffer:
            return None
        if len(self._buffer) > self.recent_window:
            entries = list(self._buffer)[-self.recent_window :]
        else:
            entries = list(self._buffer)

        size = min(size, len(entries))
        priorities = torch.tensor([p for _, _, p in entries], dtype=torch.float32)
        priorities = torch.nan_to_num(priorities, nan=1e-4, posinf=1.0, neginf=1e-4).clamp(min=1e-6)
        total = priorities.sum()
        if float(total.item()) <= 0.0:
            probs = torch.full_like(priorities, 1.0 / float(len(priorities)))
        else:
            probs = priorities / total
        choice = torch.multinomial(probs, num_samples=size, replacement=False)
        xs, ys = [], []
        for idx in choice.tolist():
            x_i, y_i, _ = entries[idx]
            xs.append(x_i)
            ys.append(y_i)
        return torch.stack(xs).to(device), torch.stack(ys).to(device)


@dataclass
class SynapseState:
    hidden: torch.Tensor  # [B, E, D]
    health: torch.Tensor  # [E]
    alive_mask: torch.Tensor  # [E] bool
    usage_ema: torch.Tensor  # [E]
    step: int = 0
    prediction_error_ema: Optional[torch.Tensor] = None  # [B]
    prune_streak: Optional[torch.Tensor] = None  # [E]
    expert_error_ema: Optional[torch.Tensor] = None  # [E]
    expert_error_var: Optional[torch.Tensor] = None  # [E]
    myelinated_mask: Optional[torch.Tensor] = None  # [E] bool
    fuse_streak: Optional[torch.Tensor] = None  # [E]

    def clone(self) -> "SynapseState":
        def _clone(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if x is None else x.detach().clone()

        return SynapseState(
            hidden=self.hidden.detach().clone(),
            health=self.health.detach().clone(),
            alive_mask=self.alive_mask.detach().clone(),
            usage_ema=self.usage_ema.detach().clone(),
            step=self.step,
            prediction_error_ema=_clone(self.prediction_error_ema),
            prune_streak=_clone(self.prune_streak),
            expert_error_ema=_clone(self.expert_error_ema),
            expert_error_var=_clone(self.expert_error_var),
            myelinated_mask=_clone(self.myelinated_mask),
            fuse_streak=_clone(self.fuse_streak),
        )

    def ensure_internal(self, batch_size: int, num_experts: int, device: torch.device) -> None:
        if self.prediction_error_ema is None:
            self.prediction_error_ema = torch.zeros(batch_size, device=device)
        if self.prune_streak is None:
            self.prune_streak = torch.zeros(num_experts, device=device)
        if self.expert_error_ema is None:
            self.expert_error_ema = torch.zeros(num_experts, device=device)
        if self.expert_error_var is None:
            self.expert_error_var = torch.zeros(num_experts, device=device)
        if self.myelinated_mask is None:
            self.myelinated_mask = torch.zeros(num_experts, dtype=torch.bool, device=device)
        if self.fuse_streak is None:
            self.fuse_streak = torch.zeros(num_experts, device=device)


class SiliconSynapse(nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        num_experts: int = 128,
        top_k: int = 4,
        dt_default: float = 0.02,
        health_init: float = 10.0,
        health_boost: float = 1.0,
        max_health: float = 100.0,
        health_step_decay: float = 0.999,
        decay_factor: float = 0.92,
        prune_threshold: float = 2.0,
        sleep_interval: int = 1000,
        prune_patience: int = 3,
        usage_floor: float = 0.01,
        underuse_penalty_scale: float = 0.25,
        fuse_usage_threshold: float = 0.85,
        fuse_window: int = 10000,
        fuse_error_var_threshold: float = 0.0025,
        usage_ema_decay: float = 0.99,
        error_ema_decay: float = 0.98,
        replay_capacity: int = 4096,
        replay_recent_window: int = 1024,
        replay_batch: int = 32,
        replay_interval: int = 4,
        replay_weight: float = 0.02,
        load_balance_weight: float = 0.001,
        stability_weight: float = 0.001,
        adaptation_gain: float = 8.0,
        adaptation_gain_cap: float = 2.0,
        novelty_scale: float = 1.0,
        usage_penalty_scale: float = 1.0,
        health_prediction_gain_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.dt_default = dt_default

        self.health_init = health_init
        self.health_boost = health_boost
        self.max_health = max_health
        self.health_step_decay = health_step_decay
        self.decay_factor = decay_factor
        self.prune_threshold = prune_threshold
        self.sleep_interval = sleep_interval
        self.prune_patience = prune_patience
        self.usage_floor = usage_floor
        self.underuse_penalty_scale = underuse_penalty_scale
        self.fuse_usage_threshold = fuse_usage_threshold
        self.fuse_window = fuse_window
        self.fuse_error_var_threshold = fuse_error_var_threshold
        self.usage_ema_decay = usage_ema_decay
        self.error_ema_decay = error_ema_decay
        self.replay_batch = replay_batch
        self.replay_interval = replay_interval
        self.replay_weight = replay_weight
        self.load_balance_weight = load_balance_weight
        self.stability_weight = stability_weight
        self.adaptation_gain = adaptation_gain
        self.adaptation_gain_cap = adaptation_gain_cap
        self.health_prediction_gain_scale = health_prediction_gain_scale

        self.experts = nn.ModuleList([LiquidNode(feature_dim) for _ in range(num_experts)])
        self.router = LiquidMoERouter(
            num_experts=num_experts,
            feature_dim=feature_dim,
            top_k=top_k,
            novelty_scale=novelty_scale,
            usage_penalty_scale=usage_penalty_scale,
        )
        self.fast_adapter = nn.Linear(feature_dim, feature_dim)
        self.readout = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity, recent_window=replay_recent_window)

    def init_state(self, batch_size: int, device: torch.device) -> SynapseState:
        state = SynapseState(
            hidden=torch.zeros(batch_size, self.num_experts, self.feature_dim, device=device),
            health=torch.full((self.num_experts,), self.health_init, device=device),
            alive_mask=torch.ones(self.num_experts, dtype=torch.bool, device=device),
            usage_ema=torch.zeros(self.num_experts, device=device),
            step=0,
        )
        state.ensure_internal(batch_size=batch_size, num_experts=self.num_experts, device=device)
        return state

    def _usage_penalty(self, usage_ema: torch.Tensor) -> torch.Tensor:
        total = usage_ema.sum()
        if float(total.item()) <= 0.0:
            return torch.zeros_like(usage_ema)
        return usage_ema / total.clamp(min=1e-6)

    def _load_balance_loss(self, routing_probs: torch.Tensor, alive_mask: torch.Tensor) -> torch.Tensor:
        alive_dist = alive_mask.float()
        alive_dist = alive_dist / alive_dist.sum().clamp(min=1.0)
        routed = routing_probs.mean(dim=0)
        return F.mse_loss(routed, alive_dist)

    def _compute_expert_statistics(
        self,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        sample_errors: torch.Tensor,
        num_experts: int,
    ) -> Dict[str, torch.Tensor]:
        device = top_indices.device
        flat_idx = top_indices.reshape(-1)
        flat_weights = top_weights.reshape(-1)
        repeated_err = sample_errors.unsqueeze(1).expand_as(top_weights).reshape(-1)

        usage_mass = torch.zeros(num_experts, device=device)
        usage_mass.scatter_add_(0, flat_idx, torch.ones_like(flat_weights))
        usage_mass = usage_mass / max(1.0, float(top_indices.size(0) * top_indices.size(1)))

        weighted_error = torch.zeros(num_experts, device=device)
        weight_mass = torch.zeros(num_experts, device=device)
        weighted_error.scatter_add_(0, flat_idx, repeated_err * flat_weights)
        weight_mass.scatter_add_(0, flat_idx, flat_weights)
        mean_error = weighted_error / weight_mass.clamp(min=1e-6)

        return {
            "usage_mass": usage_mass,
            "mean_error": mean_error,
            "weight_mass": weight_mass,
        }

    def forward(
        self,
        x_t: torch.Tensor,
        state: SynapseState,
        dt: float,
        train_mode: bool = True,
    ) -> Tuple[torch.Tensor, SynapseState, Dict[str, torch.Tensor]]:
        del train_mode
        batch_size, feature_dim = x_t.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {feature_dim}.")
        if state.hidden.shape != (batch_size, self.num_experts, self.feature_dim):
            raise ValueError("State hidden tensor does not match [B, E, D].")

        next_state = state.clone()
        next_state.ensure_internal(batch_size=batch_size, num_experts=self.num_experts, device=x_t.device)

        usage_penalty = self._usage_penalty(next_state.usage_ema)
        route = self.router(
            x=x_t,
            prediction_error_ema=next_state.prediction_error_ema,
            usage_penalty=usage_penalty,
            alive_mask=next_state.alive_mask,
        )
        top_indices = route["top_indices"]
        top_weights = route["top_weights"]
        routing_probs = route["routing_probs"]

        output_hidden = torch.zeros(batch_size, self.feature_dim, device=x_t.device, dtype=x_t.dtype)
        new_hidden = next_state.hidden.clone()

        active_experts = torch.unique(top_indices)
        for expert_idx in active_experts.tolist():
            expert = self.experts[expert_idx]
            chosen_mask = top_indices.eq(expert_idx)
            chosen_batch = chosen_mask.any(dim=1)
            if not torch.any(chosen_batch):
                continue

            x_sub = x_t[chosen_batch]
            h_sub = next_state.hidden[chosen_batch, expert_idx, :]
            out_sub = expert(x_sub, h_sub, dt)

            weight_sub = (top_weights * chosen_mask.float()).sum(dim=1)[chosen_batch]
            output_hidden[chosen_batch] += out_sub * weight_sub.unsqueeze(-1)
            new_hidden[chosen_batch, expert_idx, :] = out_sub

        next_state.hidden = new_hidden
        # Residual + fast plastic adapter + expert path for fast online adaptation.
        y_pred = x_t + self.fast_adapter(x_t) + self.readout(output_hidden)

        # Activation is fraction of batch that selected each expert at least once (range [0, 1]).
        selected_once = F.one_hot(top_indices, num_classes=self.num_experts).amax(dim=1).float()
        expert_activation = selected_once.mean(dim=0)
        next_state.usage_ema = self.usage_ema_decay * next_state.usage_ema + (1.0 - self.usage_ema_decay) * expert_activation

        next_state.health = torch.where(
            next_state.alive_mask,
            next_state.health + self.health_boost * expert_activation,
            next_state.health,
        )
        next_state.health = torch.where(
            next_state.alive_mask,
            next_state.health * self.health_step_decay,
            next_state.health,
        )
        next_state.health = next_state.health.clamp(min=0.0, max=self.max_health)
        next_state.step += 1

        aux = {
            "routing_probs": routing_probs,
            "top_indices": top_indices,
            "top_weights": top_weights,
            "expert_activation": expert_activation,
        }
        return y_pred, next_state, aux

    def _predict_loss(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_errors = ((y_pred - y_target) ** 2).mean(dim=1)
        return sample_errors.mean(), sample_errors

    def online_step(
        self,
        x_t: torch.Tensor,
        y_target: torch.Tensor,
        state: SynapseState,
        dt: float,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[torch.Tensor, SynapseState, Dict[str, float]]:
        self.train()
        optimizer.zero_grad(set_to_none=True)

        y_pred, next_state, aux = self.forward(x_t=x_t, state=state, dt=dt, train_mode=True)
        pred_loss, sample_errors = self._predict_loss(y_pred=y_pred, y_target=y_target)
        load_loss = self._load_balance_loss(aux["routing_probs"], next_state.alive_mask)
        # Streaming mode uses one-step updates; detach previous state to avoid BPTT across steps.
        stability_loss = ((next_state.hidden - state.hidden.detach()) ** 2).mean()
        surprise = sample_errors.detach().mean()
        adapt_boost = 1.0 + torch.clamp(surprise * self.adaptation_gain, 0.0, self.adaptation_gain_cap)
        boosted_pred_loss = pred_loss * adapt_boost
        loss = boosted_pred_loss + self.load_balance_weight * load_loss + self.stability_weight * stability_loss

        replay_loss = torch.zeros((), device=x_t.device)
        replay_weight = self.replay_weight
        replay_batch = None
        if next_state.step % self.replay_interval == 0:
            replay_batch = self.replay_buffer.sample(size=self.replay_batch, device=x_t.device)
        if replay_batch is not None:
            replay_x, replay_y = replay_batch
            replay_state = self.init_state(batch_size=replay_x.size(0), device=x_t.device)
            replay_pred, _, _ = self.forward(x_t=replay_x, state=replay_state, dt=dt, train_mode=True)
            replay_loss, _ = self._predict_loss(y_pred=replay_pred, y_target=replay_y)
            # Down-weight replay when surprise is high so shift adaptation is not slowed by stale regimes.
            replay_weight = float((self.replay_weight * torch.clamp(1.0 - 8.0 * surprise, 0.0, 1.0)).item())
            loss = loss + replay_weight * replay_loss

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        old_error = state.prediction_error_ema if state.prediction_error_ema is not None else torch.zeros_like(sample_errors)
        next_state.prediction_error_ema = self.error_ema_decay * old_error + (1.0 - self.error_ema_decay) * sample_errors.detach()

        stats = self._compute_expert_statistics(
            top_indices=aux["top_indices"],
            top_weights=aux["top_weights"],
            sample_errors=sample_errors.detach(),
            num_experts=self.num_experts,
        )

        mean_error = stats["mean_error"]
        expert_gain = torch.clamp(next_state.expert_error_ema - mean_error, min=0.0)
        next_state.health = torch.where(
            next_state.alive_mask,
            next_state.health + self.health_prediction_gain_scale * expert_gain,
            next_state.health,
        )

        delta = mean_error - next_state.expert_error_ema
        next_state.expert_error_ema = self.error_ema_decay * next_state.expert_error_ema + (1.0 - self.error_ema_decay) * mean_error
        next_state.expert_error_var = self.error_ema_decay * next_state.expert_error_var + (1.0 - self.error_ema_decay) * (delta**2)

        for x_i, y_i, err_i in zip(x_t.detach(), y_target.detach(), sample_errors.detach()):
            self.replay_buffer.add(x=x_i, y=y_i, priority=float(err_i.item() + 1e-4))

        sleep_info: Dict[str, float] = {"ran": 0.0, "pruned": 0.0}
        if next_state.step % self.sleep_interval == 0:
            sleep_info = self.sleep_cycle(next_state)

        myelin_info = self.myelinate(next_state)
        next_state.hidden = next_state.hidden.detach()
        if next_state.prediction_error_ema is not None:
            next_state.prediction_error_ema = next_state.prediction_error_ema.detach()
        if next_state.expert_error_ema is not None:
            next_state.expert_error_ema = next_state.expert_error_ema.detach()
        if next_state.expert_error_var is not None:
            next_state.expert_error_var = next_state.expert_error_var.detach()
        info = {
            "loss": float(loss.item()),
            "pred_loss": float(pred_loss.item()),
            "boosted_pred_loss": float(boosted_pred_loss.item()),
            "adapt_boost": float(adapt_boost.item()),
            "load_loss": float(load_loss.item()),
            "stability_loss": float(stability_loss.item()),
            "replay_loss": float(replay_loss.item()) if replay_batch is not None else 0.0,
            "replay_weight": float(replay_weight) if replay_batch is not None else 0.0,
            "mse": float(sample_errors.mean().item()),
            "alive_experts": float(next_state.alive_mask.sum().item()),
            "myelinated_experts": float(next_state.myelinated_mask.sum().item() if next_state.myelinated_mask is not None else 0),
            "sleep_ran": float(sleep_info.get("ran", 0.0)),
            "pruned_now": float(sleep_info.get("pruned", 0.0)),
            "myelinated_now": float(myelin_info.get("myelinated", 0.0)),
        }
        return y_pred.detach(), next_state, info

    def _prune_expert(self, idx: int) -> None:
        with torch.no_grad():
            self.router.gate.weight[idx].zero_()
            if self.router.gate.bias is not None:
                self.router.gate.bias[idx] = torch.finfo(self.router.gate.bias.dtype).min
            expert = self.experts[idx]
            if isinstance(expert, LiquidNode):
                expert.ternary_transform.weight.zero_()
                expert.raw_tau.zero_()
            elif isinstance(expert, FusedMyelinatedNode):
                expert.static_weight.zero_()
                expert.static_alpha.zero_()

    def sleep_cycle(self, state: SynapseState) -> Dict[str, float]:
        state.ensure_internal(batch_size=state.hidden.size(0), num_experts=self.num_experts, device=state.hidden.device)
        alive = state.alive_mask

        state.health = torch.where(alive, state.health * self.decay_factor, state.health)
        underuse = torch.clamp(self.usage_floor - state.usage_ema, min=0.0)
        state.health = torch.where(alive, state.health - self.underuse_penalty_scale * underuse, state.health)
        state.health = state.health.clamp(min=0.0, max=self.max_health)
        below = alive & (state.health < self.prune_threshold)
        state.prune_streak = torch.where(below, state.prune_streak + 1, torch.zeros_like(state.prune_streak))

        to_prune = below & (state.prune_streak >= float(self.prune_patience))
        prune_indices = torch.nonzero(to_prune, as_tuple=False).flatten().tolist()
        alive_indices = torch.nonzero(alive, as_tuple=False).flatten().tolist()
        if alive_indices and len(prune_indices) >= len(alive_indices):
            keep_idx = max(alive_indices, key=lambda idx: float(state.health[idx].item()))
            prune_indices = [idx for idx in prune_indices if idx != keep_idx]
        for idx in prune_indices:
            self._prune_expert(idx)
            state.alive_mask[idx] = False
            state.health[idx] = 0.0
            state.usage_ema[idx] = 0.0
            state.prune_streak[idx] = 0.0
            if state.myelinated_mask is not None:
                state.myelinated_mask[idx] = False
            if state.fuse_streak is not None:
                state.fuse_streak[idx] = 0.0

        return {"ran": 1.0, "pruned": float(len(prune_indices))}

    def myelinate(self, state: SynapseState) -> Dict[str, float]:
        state.ensure_internal(batch_size=state.hidden.size(0), num_experts=self.num_experts, device=state.hidden.device)
        usage_scale = state.usage_ema.max().clamp(min=1e-6)
        relative_usage = state.usage_ema / usage_scale
        # As pruning reduces the active pool, relax the gate slightly so stable
        # high-usage experts can still cross the fusion threshold.
        alive_ratio = state.alive_mask.float().mean().item()
        usage_threshold = max(0.5, self.fuse_usage_threshold * (0.5 + 0.5 * alive_ratio))
        usage_ok = relative_usage >= usage_threshold
        error_stable = state.expert_error_var <= self.fuse_error_var_threshold
        candidate = state.alive_mask & ~state.myelinated_mask & usage_ok & error_stable
        eligible = state.alive_mask & ~state.myelinated_mask & error_stable
        evidence = torch.where(eligible, relative_usage, torch.zeros_like(relative_usage))
        state.fuse_streak = torch.where(
            eligible,
            state.fuse_streak + evidence,
            torch.zeros_like(state.fuse_streak),
        )

        fuse_ready = candidate & (state.fuse_streak >= float(self.fuse_window))
        fuse_indices = torch.nonzero(fuse_ready, as_tuple=False).flatten().tolist()
        fused_count = 0
        for idx in fuse_indices:
            expert = self.experts[idx]
            if isinstance(expert, LiquidNode):
                self.experts[idx] = FusedMyelinatedNode(expert, fixed_dt=self.dt_default)
                state.myelinated_mask[idx] = True
                fused_count += 1
        return {"myelinated": float(fused_count)}


if __name__ == "__main__":
    torch.manual_seed(7)
    device = default_device()

    model = SiliconSynapse(feature_dim=64, num_experts=128, top_k=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    batch_size = 16
    state = model.init_state(batch_size=batch_size, device=device)
    x = torch.randn(batch_size, 64, device=device)
    y = torch.randn(batch_size, 64, device=device)

    pred, state, info = model.online_step(x_t=x, y_target=y, state=state, dt=0.02, optimizer=optimizer)
    print(
        {
            "pred_shape": tuple(pred.shape),
            "alive_experts": int(info["alive_experts"]),
            "myelinated_experts": int(info["myelinated_experts"]),
        }
    )
