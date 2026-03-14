#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a lightweight synthesis policy from winner candidates.")
    p.add_argument("--data", type=Path, default=Path("artifacts/reasoning_dataset.jsonl"))
    p.add_argument("--out", type=Path, default=Path("artifacts/synthesis_head.json"))
    p.add_argument("--version", type=str, default="synthesis_head_v1")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    transition_counts: Counter[str] = Counter()
    sentence_lengths: list[int] = []
    samples = 0

    with args.data.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = int(row["winner_index"])
            winner = row["candidates"][idx]["text"]
            low = winner.lower()
            if "phase" in low or "step" in low:
                transition_counts["phased"] += 1
            if "risk" in low or "failure" in low:
                transition_counts["risk"] += 1
            if "compare" in low or "versus" in low or "easier" in low:
                transition_counts["compare"] += 1
            sentence_lengths.append(max(1, winner.count(".") + winner.count("!") + winner.count("?")))
            samples += 1

    if not sentence_lengths:
        sentence_lengths = [4]
    avg_sentences = sum(sentence_lengths) / len(sentence_lengths)
    max_sentences = max(2, min(6, round(avg_sentences)))

    policy = {
        "version": args.version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "samples": samples,
        "transitions": transition_counts.most_common(),
        "max_sentences": int(max_sentences),
        "target_style": "concise_actionable",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(args.out), "samples": samples}, indent=2))


if __name__ == "__main__":
    main()
