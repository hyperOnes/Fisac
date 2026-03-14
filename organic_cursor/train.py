from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from organic_cursor.dataset import CursorWindowDataset, build_from_csv
from organic_cursor.models import CursorGRU, CursorLiquid


@dataclass
class TrainConfig:
    data_csv: Path
    weights_dir: Path
    epochs: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    hidden_dim: int = 64
    sequence_length: int = 30
    horizon_start: int = 15
    horizon_points: int = 5
    device: str = "cpu"


def _pick_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_train(model: torch.nn.Module, train_dl: DataLoader, val_dl: DataLoader, device: torch.device, cfg: TrainConfig) -> float:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(cfg.epochs):
        total = 0.0
        n = 0
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * x.size(0)
            n += int(x.size(0))
        train_mse = total / max(1, n)

        model.eval()
        with torch.no_grad():
            v_total = 0.0
            v_n = 0
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                v_total += float(loss_fn(pred, y).item()) * x.size(0)
                v_n += int(x.size(0))
            val_mse = v_total / max(1, v_n)
        model.train()

        print(f"epoch={epoch + 1:02d} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")
        if val_mse < best:
            best = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def _eval(model: torch.nn.Module, test_dl: DataLoader, device: torch.device) -> float:
    loss_fn = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        total = 0.0
        n = 0
        for x, y in test_dl:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            total += float(loss_fn(pred, y).item()) * x.size(0)
            n += int(x.size(0))
    return total / max(1, n)


def train_models(cfg: TrainConfig) -> None:
    ds = build_from_csv(
        csv_path=cfg.data_csv,
        sequence_length=cfg.sequence_length,
        horizon_start=cfg.horizon_start,
        horizon_points=cfg.horizon_points,
    )

    train_dl = DataLoader(CursorWindowDataset(ds.train_x, ds.train_y), batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(CursorWindowDataset(ds.val_x, ds.val_y), batch_size=cfg.batch_size, shuffle=False)
    test_dl = DataLoader(CursorWindowDataset(ds.test_x, ds.test_y), batch_size=cfg.batch_size, shuffle=False)

    device = _pick_device(cfg.device)
    cfg.weights_dir.mkdir(parents=True, exist_ok=True)

    gru = CursorGRU(input_dim=4, hidden_dim=cfg.hidden_dim, future_points=cfg.horizon_points)
    print("Training GRU...")
    _run_train(gru, train_dl, val_dl, device, cfg)
    gru_test = _eval(gru, test_dl, device)
    print(f"GRU test_mse={gru_test:.6f}")
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in gru.state_dict().items()},
            "feature_mean": ds.feature_mean,
            "feature_std": ds.feature_std,
            "test_mse": gru_test,
        },
        cfg.weights_dir / "gru_cursor.pt",
    )

    liquid = CursorLiquid(input_dim=4, future_points=cfg.horizon_points)
    print("Training Liquid...")
    _run_train(liquid, train_dl, val_dl, device, cfg)
    liquid_test = _eval(liquid, test_dl, device)
    print(f"Liquid test_mse={liquid_test:.6f}")
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in liquid.state_dict().items()},
            "feature_mean": ds.feature_mean,
            "feature_std": ds.feature_std,
            "test_mse": liquid_test,
        },
        cfg.weights_dir / "liquid_cursor.pt",
    )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train organic cursor GRU + Liquid predictors.")
    parser.add_argument("--data-csv", type=Path, default=Path("organic_cursor/data/mouse_capture.csv"))
    parser.add_argument("--weights-dir", type=Path, default=Path("organic_cursor/weights"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--horizon-start", type=int, default=15)
    parser.add_argument("--horizon-points", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    return TrainConfig(
        data_csv=args.data_csv,
        weights_dir=args.weights_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        sequence_length=args.sequence_length,
        horizon_start=args.horizon_start,
        horizon_points=args.horizon_points,
        device=args.device,
    )


def main() -> None:
    train_models(parse_args())


if __name__ == "__main__":
    main()
