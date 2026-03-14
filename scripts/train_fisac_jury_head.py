#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train lightweight jury head from candidate feature rows.")
    p.add_argument("--data", type=Path, default=Path("artifacts/reasoning_dataset.jsonl"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--out", type=Path, default=Path("artifacts/jury_head.pt"))
    p.add_argument("--version", type=str, default="jury_head_v1")
    return p.parse_args()


class JuryFeatureHead(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x))).squeeze(-1)


def _load_rows(path: Path) -> tuple[list[dict], list[str]]:
    rows = []
    feature_names: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            if not feature_names:
                feature_names = [str(x) for x in row.get("feature_names", [])]
    if not rows:
        raise RuntimeError("Empty dataset")
    if not feature_names:
        feature_names = [
            "coverage",
            "echo",
            "actionability",
            "coherence",
            "alignment",
            "evidence",
            "contradiction_penalty",
            "meta_penalty",
            "generic_penalty",
            "role_bonus",
            "role_mismatch_penalty",
            "length_quality",
            "score_hint",
        ]
    return rows, feature_names


def _feature_tensor(candidate: dict, feature_names: list[str]) -> torch.Tensor:
    feats = candidate.get("features", {})
    return torch.tensor([float(feats.get(name, 0.0)) for name in feature_names], dtype=torch.float32)


def main() -> None:
    args = _parse_args()
    rows, feature_names = _load_rows(args.data)
    dim = len(feature_names)
    model = JuryFeatureHead(dim=dim, hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        running = 0.0
        n = 0
        for row in rows:
            candidates = row["candidates"]
            winner = int(row["winner_index"])
            x = torch.stack([_feature_tensor(c, feature_names) for c in candidates], dim=0)
            logits = model(x)
            target = torch.tensor([winner], dtype=torch.long)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += float(loss.item())
            n += 1
        print(f"epoch={epoch + 1} loss={running / max(1, n):.4f}")

    # Collapse into linearized first-order scorer so runtime can use cheap dot-product.
    with torch.no_grad():
        # Approximate influence from first layer and output layer.
        w1 = model.fc1.weight.detach().cpu()  # [H, D]
        b1 = model.fc1.bias.detach().cpu()  # [H]
        w2 = model.fc2.weight.detach().cpu().reshape(-1)  # [H]
        b2 = float(model.fc2.bias.detach().cpu().item())
        relu_mask = (b1 > 0).float()
        eff_w = (w2 * relu_mask).unsqueeze(0) @ w1
        eff_w = eff_w.reshape(-1)
        eff_b = float((w2 * relu_mask * b1).sum().item() + b2)

    payload = {
        "version": args.version,
        "feature_names": feature_names,
        "weights": [float(x) for x in eff_w.tolist()],
        "bias": float(eff_b),
        "epochs": int(args.epochs),
        "hidden": int(args.hidden),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(args.out), "sidecar": str(sidecar)}, indent=2))


if __name__ == "__main__":
    main()
