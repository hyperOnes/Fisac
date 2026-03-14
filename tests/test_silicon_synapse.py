from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from environment_stream import CSVForecastStream, ChaoticRegimeStream, NonStationarySineStream, StreamConfig
from silicon_synapse import (
    FusedMyelinatedNode,
    LiquidMoERouter,
    LiquidNode,
    SiliconSynapse,
    TernaryWeight,
)


def test_ternary_quantizer_and_ste_gradient() -> None:
    weight = torch.randn(8, 4, requires_grad=True)
    quantized = TernaryWeight.apply(weight)

    scale = weight.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
    expected = torch.clamp(torch.round(weight.detach() / scale), -1.0, 1.0) * scale
    assert torch.allclose(quantized.detach(), expected)

    quantized.sum().backward()
    assert weight.grad is not None
    assert torch.allclose(weight.grad, torch.ones_like(weight))


def test_tau_positive_and_alpha_bounded() -> None:
    node = LiquidNode(feature_dim=16)
    tau = node.tau()
    alpha = node.alpha(dt=0.02)
    assert torch.all(tau > 0.0)
    assert torch.all(alpha >= 0.0)
    assert torch.all(alpha <= 1.0)


def test_router_never_selects_dead_experts() -> None:
    router = LiquidMoERouter(num_experts=6, feature_dim=4, top_k=3)
    x = torch.randn(5, 4)
    prediction_error_ema = torch.zeros(5)
    usage_penalty = torch.zeros(6)
    alive_mask = torch.tensor([True, False, True, False, True, True])

    out = router(
        x=x,
        prediction_error_ema=prediction_error_ema,
        usage_penalty=usage_penalty,
        alive_mask=alive_mask,
    )
    selected = out["top_indices"]
    assert not torch.isin(selected, torch.tensor([1, 3])).any()


def test_sleep_cycle_prunes_and_router_ignores_pruned_expert() -> None:
    model = SiliconSynapse(feature_dim=8, num_experts=6, top_k=2, sleep_interval=2, prune_patience=1)
    state = model.init_state(batch_size=2, device=torch.device("cpu"))

    prune_idx = 2
    state.health[prune_idx] = 0.0
    state.prune_streak[prune_idx] = float(model.prune_patience)

    info = model.sleep_cycle(state)
    assert info["pruned"] >= 1.0
    assert not state.alive_mask[prune_idx]
    assert torch.all(model.router.gate.weight[prune_idx] == 0.0)
    if model.router.gate.bias is not None:
        assert model.router.gate.bias[prune_idx].item() < -1e20

    x = torch.randn(2, 8)
    _, _, aux = model.forward(x_t=x, state=state, dt=0.02)
    assert not torch.isin(aux["top_indices"], torch.tensor([prune_idx])).any()


def test_myelination_fused_node_matches_liquid_node() -> None:
    dt = 0.02
    model = SiliconSynapse(feature_dim=8, num_experts=1, top_k=1, dt_default=dt, fuse_window=1)
    state = model.init_state(batch_size=1, device=torch.device("cpu"))

    liquid = model.experts[0]
    assert isinstance(liquid, LiquidNode)
    x = torch.randn(1, 8)
    h = torch.randn(1, 8)
    out_before = liquid(x, h, dt=dt).detach()

    model.fuse_usage_threshold = 0.0
    model.fuse_error_var_threshold = 1.0
    state.usage_ema[0] = 1.0
    state.expert_error_var[0] = 0.0
    state.fuse_streak[0] = 1.0
    state.myelinated_mask[0] = False
    state.alive_mask[0] = True

    model.myelinate(state)
    assert isinstance(model.experts[0], FusedMyelinatedNode)
    out_after = model.experts[0](x, h, dt=dt).detach()
    assert torch.allclose(out_before, out_after, atol=1e-6, rtol=1e-5)


def test_myelination_accumulates_evidence_across_brief_usage_dips() -> None:
    model = SiliconSynapse(feature_dim=8, num_experts=2, top_k=1, dt_default=0.02, fuse_window=2)
    state = model.init_state(batch_size=1, device=torch.device("cpu"))
    model.fuse_error_var_threshold = 1.0
    state.expert_error_var.zero_()

    state.usage_ema = torch.tensor([1.0, 0.1])
    info = model.myelinate(state)
    streak_after_hit = float(state.fuse_streak[0].item())
    assert info["myelinated"] == 0.0
    assert streak_after_hit > 0.0
    assert not state.myelinated_mask[0]

    state.usage_ema = torch.tensor([0.7, 1.0])
    info = model.myelinate(state)
    assert info["myelinated"] == 0.0
    assert float(state.fuse_streak[0].item()) > streak_after_hit
    assert not state.myelinated_mask[0]

    state.usage_ema = torch.tensor([1.0, 0.2])
    info = model.myelinate(state)
    assert info["myelinated"] >= 1.0
    assert state.myelinated_mask[0]


def test_online_step_updates_parameters_and_advances_state() -> None:
    model = SiliconSynapse(feature_dim=8, num_experts=8, top_k=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    state = model.init_state(batch_size=4, device=torch.device("cpu"))

    x_t = torch.randn(4, 8)
    y_t = torch.randn(4, 8)
    before = model.readout[0].weight.detach().clone()
    _, next_state, _ = model.online_step(
        x_t=x_t,
        y_target=y_t,
        state=state,
        dt=0.02,
        optimizer=optimizer,
    )
    assert next_state.step == state.step + 1
    assert not torch.allclose(before, model.readout[0].weight.detach())
    assert torch.isfinite(next_state.hidden).all()


def test_task_c_stream_shapes_and_regimes() -> None:
    stream = ChaoticRegimeStream(StreamConfig(feature_dim=12, batch_size=5, regime_len=20, noise_std=0.02, seed=3))
    x_t, y_t, meta = stream.next_batch()
    assert x_t.shape == (5, 12)
    assert y_t.shape == (5, 12)
    assert isinstance(meta["regime_id"], int)
    assert 0 <= meta["regime_id"] <= 3


def test_csv_stream_load_and_step(tmp_path: pytest.TempPathFactory) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "donor_age,leakage,mutation_reduction,tissue,viability\n"
        "25,0.31,1.45,blood_cd34,0.96\n"
        "45,0.28,1.62,muscle_satellite,0.93\n"
        "61,0.39,1.21,blood_cd34,0.91\n"
        "61,0.41,1.17,muscle_satellite,0.89\n",
        encoding="utf-8",
    )
    stream = CSVForecastStream(StreamConfig(feature_dim=10, batch_size=2, seed=9), csv_path=csv_path)
    x_t, y_t, meta = stream.next_batch()
    assert x_t.shape == (2, 10)
    assert y_t.shape == (2, 10)
    assert isinstance(meta["regime_id"], int)


def test_sleep_cycle_underuse_penalty_decreases_health() -> None:
    model = SiliconSynapse(feature_dim=8, num_experts=4, top_k=2, usage_floor=0.1, underuse_penalty_scale=1.0)
    state = model.init_state(batch_size=2, device=torch.device("cpu"))
    before = state.health.clone()
    state.usage_ema.zero_()
    model.sleep_cycle(state)
    assert torch.all(state.health <= before)


def test_benchmark_liquid_emits_required_json_keys() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out = subprocess.check_output(
        [
            sys.executable,
            "benchmark_liquid.py",
            "--task",
            "task_a",
            "--steps",
            "40",
            "--experts",
            "8",
            "--dim",
            "16",
            "--device",
            "cpu",
            "--seed",
            "3",
        ],
        cwd=repo_root,
        text=True,
    )
    metrics = json.loads(out)
    required = {
        "mse",
        "adaptation_steps",
        "train_steps_per_sec",
        "steps_per_sec",
        "seconds_per_step",
        "elapsed_seconds",
        "peak_mem_gb",
        "alive_experts",
        "myelinated_experts",
    }
    assert required.issubset(metrics.keys())


@pytest.mark.slow
def test_smoke_5k_steps_no_nan_divergence() -> None:
    model = SiliconSynapse(feature_dim=16, num_experts=16, top_k=2, sleep_interval=250)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    stream = NonStationarySineStream(StreamConfig(feature_dim=16, batch_size=4, regime_len=100, noise_std=0.03))
    state = model.init_state(batch_size=4, device=torch.device("cpu"))

    for _ in range(5000):
        x_t, y_t, _ = stream.next_batch()
        pred, state, _ = model.online_step(
            x_t=x_t,
            y_target=y_t,
            state=state,
            dt=0.02,
            optimizer=optimizer,
        )
        assert torch.isfinite(pred).all()
        assert torch.isfinite(state.hidden).all()
