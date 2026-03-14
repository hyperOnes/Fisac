#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


ROLES = [
    "direct_answer",
    "comparative_feasibility",
    "failure_modes",
    "implementation_sequence",
    "skeptical_counterexample",
]

PROVIDERS = ["gemini", "openai", "fisac_truth_reasoner"]
TOK_RE = re.compile(r"[a-z0-9']+")
STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
    "you",
    "your",
}

SCENARIOS = [
    {
        "prompt": "Assess feasibility: self-laying maglev track car at 100km/h.",
        "context": "Track deployment at speed risks jamming. Alignment drift amplifies magnetic instability.",
        "winner_role": "failure_modes",
    },
    {
        "prompt": "Compare dynamic maglev car versus crewed Venus mission this decade.",
        "context": "Venus mission requires thermal survival and launch windows. Ground mobility can be staged incrementally.",
        "winner_role": "comparative_feasibility",
    },
    {
        "prompt": "How exactly should prototype validation be sequenced?",
        "context": "Bench tests can isolate failure modes before full-speed trials. Abort criteria prevent cascading damage.",
        "winner_role": "implementation_sequence",
    },
    {
        "prompt": "Explain why this approach is likely to fail first.",
        "context": "Mechanical retraction under load causes fatigue and jams. Recovery procedures are often underdesigned.",
        "winner_role": "failure_modes",
    },
    {
        "prompt": "Give one decision now and one next step.",
        "context": "Decisions improve when options are ranked by risk-adjusted time-to-proof.",
        "winner_role": "direct_answer",
    },
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build reasoning dataset with candidate features for jury-head training.")
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("artifacts/reasoning_dataset.jsonl"))
    return p.parse_args()


def _tokens(text: str) -> list[str]:
    return TOK_RE.findall(text.lower())


def _overlap(a: list[str], b: list[str]) -> float:
    sa = {t for t in a if t and t not in STOP}
    sb = {t for t in b if t and t not in STOP}
    if not sa:
        sa = set(a)
    if not sb:
        sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sb))


def _echo(candidate: str, user: str) -> float:
    ct = set(_tokens(candidate))
    ut = set(_tokens(user))
    if not ct or not ut:
        return 0.0
    j = len(ct & ut) / max(1, len(ct | ut))
    return min(1.0, j)


def _actionability(tokens: list[str]) -> float:
    action = {"build", "test", "measure", "compare", "prototype", "validate", "rank", "choose", "phase", "step"}
    hits = sum(1 for t in tokens if t in action)
    return min(1.0, hits / 3.0)


def _evidence(tokens: list[str]) -> float:
    words = {"because", "therefore", "risk", "failure", "mitigation", "tradeoff", "threshold", "abort"}
    hits = sum(1 for t in tokens if t in words)
    return min(1.0, hits / 3.0)


def _contradiction(text: str) -> float:
    low = text.lower()
    pairs = [("easier", "harder"), ("feasible", "impossible"), ("safe", "unsafe"), ("always", "never")]
    hits = sum(1 for a, b in pairs if a in low and b in low)
    return min(1.0, hits / 2.0)


def _meta(text: str) -> float:
    low = text.lower()
    patterns = ["based on the available context", "user asks", "user wants", "previous assistant"]
    hits = sum(1 for p in patterns if p in low)
    return min(1.0, hits / 2.0)


def _length_quality(tokens: list[str]) -> float:
    n = len(tokens)
    if n < 12:
        return 0.1
    if 18 <= n <= 80:
        return 1.0
    if n < 18:
        return n / 18.0
    return max(0.1, 1.0 - ((n - 80) / 180.0))


def _coherence(candidate: str, context: str) -> float:
    return _overlap(_tokens(candidate), _tokens(context))


def _candidate_text(prompt: str, context: str, role: str, rng: random.Random) -> str:
    if role == "direct_answer":
        return (
            "Choose the lower-complexity path first. "
            "Rank options by time-to-proof under hard safety constraints."
        )
    if role == "comparative_feasibility":
        return (
            "The near-term winner is the option with shorter validation loops and fewer coupled unknowns. "
            "Compare by risk-adjusted engineering cycle time, then pick one."
        )
    if role == "failure_modes":
        return (
            "Top failure modes are alignment drift, retraction jamming, and thermal fatigue. "
            "Mitigate in that order with instrumented thresholds and automatic abort gates."
        )
    if role == "implementation_sequence":
        return (
            "Phase 1 bench isolation. Phase 2 low-speed closed-loop trials. "
            "Phase 3 incremental speed ramps with quantified pass/fail gates."
        )
    if rng.random() < 0.5:
        return (
            "Counterexample: if maintenance latency dominates, apparent feasibility collapses. "
            "Falsify this by measuring mean-time-to-recovery under realistic fault injection."
        )
    return (
        "Challenge the hidden assumption that coupling can be controlled at speed. "
        "If control lag exceeds threshold, the architecture is not production-viable."
    )


def _winner_score(prompt: str, context: str, role: str, text: str, scenario_role: str) -> tuple[float, dict[str, float]]:
    t = _tokens(text)
    features = {
        "coverage": float(_overlap(t, _tokens(prompt))),
        "echo": float(_echo(text, prompt)),
        "actionability": float(_actionability(t)),
        "coherence": float(_coherence(text, context)),
        "alignment": float((_overlap(t, _tokens(prompt)) + _overlap(t, _tokens(context))) / 2.0),
        "evidence": float(_evidence(t)),
        "contradiction_penalty": float(_contradiction(text)),
        "meta_penalty": float(_meta(text)),
        "generic_penalty": float(0.0 if len(t) >= 12 else 0.4),
        "role_bonus": float(1.0 if role == scenario_role else 0.0),
        "role_mismatch_penalty": float(0.0 if role == scenario_role else 0.3),
        "length_quality": float(_length_quality(t)),
        "score_hint": 0.18,
    }
    score = (
        0.34 * features["coverage"]
        + 0.18 * features["actionability"]
        + 0.12 * features["coherence"]
        + 0.20 * features["alignment"]
        + 0.10 * features["length_quality"]
        + 0.08 * features["evidence"]
        - 0.30 * features["contradiction_penalty"]
        - 0.45 * features["echo"]
        - 0.42 * features["meta_penalty"]
        - 0.34 * features["generic_penalty"]
        + 0.24 * features["role_bonus"]
        - 0.24 * features["role_mismatch_penalty"]
        + 0.12 * features["score_hint"]
    )
    return float(score), features


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        for _ in range(args.samples):
            scenario = rng.choice(SCENARIOS)
            prompt = scenario["prompt"]
            context = scenario["context"]
            scenario_role = scenario["winner_role"]
            roles = list(ROLES)
            rng.shuffle(roles)
            candidates = []
            scored = []
            for role in roles:
                provider = rng.choice(PROVIDERS)
                text = _candidate_text(prompt, context, role, rng)
                score, features = _winner_score(prompt, context, role, text, scenario_role)
                candidates.append(
                    {
                        "role": role,
                        "provider": provider,
                        "text": text,
                        "features": features,
                    }
                )
                scored.append(score)
            winner_index = max(range(len(scored)), key=lambda i: scored[i])
            row = {
                "prompt": prompt,
                "context": context,
                "candidates": candidates,
                "winner_index": int(winner_index),
                "winner_score": float(scored[winner_index]),
                "feature_names": list(candidates[0]["features"].keys()),
            }
            f.write(json.dumps(row) + "\n")

    print(json.dumps({"ok": True, "samples": args.samples, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
