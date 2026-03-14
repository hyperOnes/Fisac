from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from chat_api.config import Settings
from chat_api.db import init_db
from chat_api.repository import ChatRepository
from chat_api.services.chat_service import ChatService
from chat_api.services.context_window import ContextWindowPolicy
from chat_api.services.conversational_composer import ConversationalComposer
from chat_api.services.fiscal_text_bridge import FiscalTextBridge
from chat_api.services.summary_service import SummaryService
from chat_api.services.tool_service import ToolService


REGIMES: list[dict[str, object]] = [
    {
        "name": "mobility",
        "keywords": ["vehicle", "speed", "safety", "track", "control", "reliability"],
        "prompts": [
            "Design a safer high-speed ground mobility concept.",
            "What is the main failure mode for a deployable track system?",
            "How should I compare hover concepts against wheeled systems?",
            "Which control risks dominate at 120 km/h?",
        ],
    },
    {
        "name": "energy",
        "keywords": ["battery", "thermal", "efficiency", "power", "cooling", "range"],
        "prompts": [
            "How do we improve battery thermal stability under peak load?",
            "What tradeoff matters most for range versus performance?",
            "Suggest a compact power architecture for rapid prototyping.",
            "Which efficiency lever should be tested first?",
        ],
    },
    {
        "name": "product",
        "keywords": ["user", "feedback", "iteration", "prototype", "risk", "scope"],
        "prompts": [
            "What should I test first in a new product concept?",
            "How do I reduce scope without killing ambition?",
            "Give one practical way to learn faster from user feedback.",
            "What is the fastest prototype loop for high uncertainty?",
        ],
    },
    {
        "name": "systems",
        "keywords": ["latency", "memory", "throughput", "stability", "monitoring", "drift"],
        "prompts": [
            "How should I monitor drift in an online system?",
            "What is the first latency bottleneck to remove?",
            "How do I keep throughput high under changing load?",
            "What stability guard should be mandatory in production?",
        ],
    },
]


@dataclass
class RunMetrics:
    no_signal_rate: float
    echo_rate: float
    repeat_streak_max: int
    adaptation_turns_after_shift: float
    liquid_source_rate: float
    median_latency_ms: float
    internal_mse_mean: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark chat adaptation/coherence for liquid vs deterministic paths.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--regime-len", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _seed_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or [1, 2, 3, 4, 5]


def _build_prompt_stream(steps: int, regime_len: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    stream: list[tuple[str, str]] = []
    for idx in range(steps):
        regime = REGIMES[(idx // regime_len) % len(REGIMES)]
        prompt = rng.choice(regime["prompts"])  # type: ignore[arg-type]
        stream.append((str(regime["name"]), str(prompt)))
    return stream


def _norm_words(text: str) -> set[str]:
    return {tok for tok in "".join(c.lower() if c.isalnum() else " " for c in text).split() if tok}


def _is_echo(response: str, prompt: str) -> bool:
    a = _norm_words(response)
    b = _norm_words(prompt)
    if not a or not b:
        return False
    jaccard = len(a & b) / max(1, len(a | b))
    return jaccard >= 0.82


def _keyword_overlap(text: str, keywords: Iterable[str]) -> float:
    words = _norm_words(text)
    target = {k.lower() for k in keywords}
    if not words or not target:
        return 0.0
    return len(words & target) / max(1, len(target))


async def _run_turn(service: ChatService, conversation_id: str, run_id: str, prompt: str) -> None:
    async for _ in service.stream_reply(conversation_id=conversation_id, user_text=prompt, run_id=run_id):
        pass


def _run_mode(seed: int, prompts: list[tuple[str, str]], regime_len: int, mode: str) -> RunMetrics:
    with tempfile.TemporaryDirectory(prefix=f"fisac-chat-bench-{mode}-{seed}-") as td:
        db_path = Path(td) / "bench.db"
        settings = Settings(
            db_path=db_path,
            feature_dim=32,
            num_experts=32,
            top_k=4,
            checkpoint_every_turns=25,
            lifecycle_interval_turns=200,
            generation_backend=("liquid_native" if mode == "liquid" else "hybrid"),
            gemini_default_enabled=False,
            gemini_context_only=True,
            response_guard_enabled=True,
        )
        init_db(settings)
        repo = ChatRepository(settings)
        bridge = FiscalTextBridge(repo=repo, settings=settings)
        bridge.load_or_init()
        service = ChatService(
            repo=repo,
            bridge=bridge,
            context_policy=ContextWindowPolicy(keep_last_messages=20, summary_every_user_turns=6),
            summary_service=SummaryService(max_sentences=3),
            tool_service=ToolService(repo=repo),
            composer=ConversationalComposer(),
            gemini_client=None,
        )

        conv = service.create_conversation(f"bench-{mode}", gemini_enabled=False)
        responses: list[str] = []
        sources: list[str] = []
        latencies: list[float] = []
        mse_values: list[float] = []
        no_signal = 0
        echo_count = 0
        repeat_streak = 0
        repeat_max = 0
        prev_norm = ""

        for turn, (_, prompt) in enumerate(prompts):
            asyncio.run(_run_turn(service, conv.id, f"{mode}-{seed}-{turn}", prompt))
            latest = repo.list_messages(conv.id, limit=2)[-1]
            text = latest.content.strip()
            responses.append(text)
            sources.append(latest.generation_source or "")
            if latest.latency_ms is not None and math.isfinite(latest.latency_ms):
                latencies.append(float(latest.latency_ms))
            if latest.mse is not None and math.isfinite(latest.mse):
                mse_values.append(float(latest.mse))

            if text.lower() == "no better signal":
                no_signal += 1
            if _is_echo(text, prompt):
                echo_count += 1

            norm = " ".join(text.lower().split())
            if norm and norm == prev_norm:
                repeat_streak += 1
            else:
                repeat_streak = 1
            prev_norm = norm
            repeat_max = max(repeat_max, repeat_streak)

        adaptation_steps: list[int] = []
        for shift in range(regime_len, len(prompts), regime_len):
            regime_name = prompts[shift][0]
            regime = next(r for r in REGIMES if r["name"] == regime_name)
            horizon = min(len(prompts), shift + 12)
            found = 12
            for t in range(shift, horizon):
                overlap = _keyword_overlap(responses[t], regime["keywords"])  # type: ignore[arg-type]
                if overlap >= 0.15:
                    found = t - shift
                    break
            adaptation_steps.append(found)

        return RunMetrics(
            no_signal_rate=no_signal / max(1, len(prompts)),
            echo_rate=echo_count / max(1, len(prompts)),
            repeat_streak_max=repeat_max,
            adaptation_turns_after_shift=(statistics.mean(adaptation_steps) if adaptation_steps else 0.0),
            liquid_source_rate=sum(1 for s in sources if s == "liquid_native") / max(1, len(sources)),
            median_latency_ms=(statistics.median(latencies) if latencies else 0.0),
            internal_mse_mean=(statistics.mean(mse_values) if mse_values else 0.0),
        )


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "se": 0.0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "se": 0.0}
    std = statistics.stdev(values)
    return {"mean": statistics.mean(values), "std": std, "se": std / math.sqrt(len(values))}


def main() -> None:
    args = _parse_args()
    seeds = _seed_list(args.seeds)
    per_seed: list[dict[str, object]] = []

    for seed in seeds:
        prompts = _build_prompt_stream(args.steps, args.regime_len, seed)
        liquid = _run_mode(seed, prompts, args.regime_len, mode="liquid")
        deterministic = _run_mode(seed, prompts, args.regime_len, mode="deterministic")
        delta = {
            "no_signal_rate": deterministic.no_signal_rate - liquid.no_signal_rate,
            "echo_rate": deterministic.echo_rate - liquid.echo_rate,
            "repeat_streak_max": float(deterministic.repeat_streak_max - liquid.repeat_streak_max),
            "adaptation_turns_after_shift": deterministic.adaptation_turns_after_shift - liquid.adaptation_turns_after_shift,
            "liquid_source_rate": liquid.liquid_source_rate - deterministic.liquid_source_rate,
            "median_latency_ms": deterministic.median_latency_ms - liquid.median_latency_ms,
            "internal_mse_mean": deterministic.internal_mse_mean - liquid.internal_mse_mean,
        }
        per_seed.append(
            {
                "seed": seed,
                "liquid": asdict(liquid),
                "deterministic": asdict(deterministic),
                "delta": delta,
            }
        )

    metric_names = list(RunMetrics.__annotations__.keys())
    liquid_summary: dict[str, dict[str, float]] = {}
    deterministic_summary: dict[str, dict[str, float]] = {}
    delta_summary: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        liquid_values = [float(seed_row["liquid"][metric]) for seed_row in per_seed]  # type: ignore[index]
        det_values = [float(seed_row["deterministic"][metric]) for seed_row in per_seed]  # type: ignore[index]
        delta_values = [float(seed_row["delta"][metric]) for seed_row in per_seed]  # type: ignore[index]
        liquid_summary[metric] = _summary(liquid_values)
        deterministic_summary[metric] = _summary(det_values)
        delta_summary[metric] = _summary(delta_values)

    out = {
        "config": {
            "steps": args.steps,
            "regime_len": args.regime_len,
            "seeds": seeds,
            "paired_baseline": "deterministic",
        },
        "per_seed": per_seed,
        "summary": {
            "liquid": liquid_summary,
            "deterministic": deterministic_summary,
            "delta_det_minus_liquid": delta_summary,
        },
    }

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
