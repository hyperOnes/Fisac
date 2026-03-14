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

from chat_api.config import Settings
from chat_api.db import init_db
from chat_api.repository import ChatRepository
from chat_api.services.chat_service import ChatService
from chat_api.services.context_window import ContextWindowPolicy
from chat_api.services.conversational_composer import ConversationalComposer
from chat_api.services.fiscal_text_bridge import FiscalTextBridge
from chat_api.services.gemini_client import GeminiClient
from chat_api.services.openai_client import OpenAIClient
from chat_api.services.provider_pool import ProviderPool
from chat_api.services.summary_service import SummaryService
from chat_api.services.tool_service import ToolService


@dataclass
class Metrics:
    no_signal_rate: float
    echo_rate: float
    repeat_streak_max: int
    context_available_rate: float
    liquid_source_rate: float
    median_latency_ms: float
    winner_consistency: float
    expert_judgment_score: float


_PROMPTS = {
    "mobility": [
        "Assess feasibility: self-laying maglev track car at 100km/h.",
        "Give top three failure modes and why.",
        "What exact prototype sequence would you run first?",
        "Compare this versus a guideway pod approach.",
    ],
    "systems": [
        "How to monitor drift in an online adaptation engine?",
        "What is the first bottleneck to remove for latency?",
        "Give one concrete stability guardrail.",
        "Explain tradeoff between throughput and reliability.",
    ],
    "product": [
        "How should I scope v1 of a hard engineering product?",
        "What assumptions must be falsified first?",
        "What metrics define success for first 30 days?",
        "Give a practical weekly execution loop.",
    ],
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat reasoning benchmark for jury-mode Fisac")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--out", type=Path, default=Path("benchmark_runs/chat_reasoning_benchmark.json"))
    p.add_argument("--jury-mode", type=int, default=1, choices=[0, 1])
    p.add_argument("--external-context", type=int, default=1, choices=[0, 1])
    p.add_argument("--learned-heads", type=int, default=0, choices=[0, 1])
    p.add_argument("--jury-head-path", type=str, default="")
    p.add_argument("--synthesis-head-path", type=str, default="")
    return p.parse_args()


def _seed_list(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or [1, 2, 3, 4, 5]


def _tokenize(text: str) -> set[str]:
    return {tok for tok in "".join(c.lower() if c.isalnum() else " " for c in text).split() if tok}


def _is_echo(a: str, b: str) -> bool:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return False
    j = len(ta & tb) / max(1, len(ta | tb))
    return j > 0.85


def _make_stack(
    db_path: Path,
    *,
    jury_mode: bool,
    external_context: bool,
    learned_heads: bool,
    jury_head_path: str,
    synthesis_head_path: str,
) -> tuple[ChatRepository, ChatService]:
    settings = Settings(
        db_path=db_path,
        feature_dim=32,
        num_experts=64,
        top_k=4,
        checkpoint_every_turns=25,
        lifecycle_interval_turns=100,
        generation_backend="liquid_native",
        jury_mode=jury_mode,
        learned_heads_enabled=learned_heads,
        jury_head_path=jury_head_path,
        synthesis_head_path=synthesis_head_path,
        context_query_count=5,
        external_context_enabled=external_context,
    )
    init_db(settings)
    repo = ChatRepository(settings)
    bridge = FiscalTextBridge(repo=repo, settings=settings)
    bridge.load_or_init()
    pool = ProviderPool()
    gemini = GeminiClient(settings=settings, provider_pool=pool)
    openai = OpenAIClient(settings=settings, provider_pool=pool)
    service = ChatService(
        repo=repo,
        bridge=bridge,
        context_policy=ContextWindowPolicy(keep_last_messages=20, summary_every_user_turns=6),
        summary_service=SummaryService(max_sentences=3),
        tool_service=ToolService(repo=repo),
        composer=ConversationalComposer(),
        gemini_client=gemini,
        openai_client=openai,
    )
    return repo, service


async def _run_turn(service: ChatService, conversation_id: str, run_id: str, prompt: str) -> None:
    async for _ in service.stream_reply(conversation_id=conversation_id, user_text=prompt, run_id=run_id):
        pass


def _build_prompt_stream(steps: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    domains = list(_PROMPTS.keys())
    out: list[str] = []
    for i in range(steps):
        domain = domains[(i // 20) % len(domains)]
        out.append(rng.choice(_PROMPTS[domain]))
    return out


def _score_expert_judgment(text: str) -> float:
    words = _tokenize(text)
    if not words:
        return 0.0
    action_words = {"build", "test", "compare", "measure", "prototype", "validate", "risk", "failure"}
    structure_words = {"because", "therefore", "tradeoff", "constraint", "assumption", "evidence"}
    return min(1.0, (len(words & action_words) * 0.08) + (len(words & structure_words) * 0.06))


def _run_seed(
    seed: int,
    steps: int,
    *,
    jury_mode: bool,
    external_context: bool,
    learned_heads: bool,
    jury_head_path: str,
    synthesis_head_path: str,
) -> dict[str, object]:
    prompts = _build_prompt_stream(steps, seed)
    with tempfile.TemporaryDirectory(prefix=f"fisac-chat-reasoning-{seed}-") as td:
        db_path = Path(td) / "bench.db"
        repo, service = _make_stack(
            db_path,
            jury_mode=jury_mode,
            external_context=external_context,
            learned_heads=learned_heads,
            jury_head_path=jury_head_path,
            synthesis_head_path=synthesis_head_path,
        )
        conv = service.create_conversation(f"bench-{seed}", gemini_enabled=True)

        outputs: list[str] = []
        sources: list[str] = []
        latencies: list[float] = []
        context_ok = 0
        winner_keys: list[str] = []
        repeat_streak = 0
        repeat_max = 0
        prev = ""

        for i, prompt in enumerate(prompts):
            asyncio.run(_run_turn(service, conv.id, f"{seed}-{i}", prompt))
            msg = repo.list_messages(conv.id, limit=1)[-1]
            outputs.append(msg.content)
            sources.append(msg.generation_source or "")
            if msg.latency_ms is not None and math.isfinite(msg.latency_ms):
                latencies.append(float(msg.latency_ms))

            run_row = repo.get_run(f"{seed}-{i}")
            flags = (run_row["quality_flags"] if run_row is not None else "") or ""
            if "external_context_error" not in flags:
                context_ok += 1
            trace = service.get_decision_trace(f"{seed}-{i}")
            if trace is not None:
                winner_keys.append(f"{trace.get('winner_provider')}:{trace.get('winner_role')}:{trace.get('winner_index')}")

            norm = " ".join(msg.content.lower().split())
            if norm and norm == prev:
                repeat_streak += 1
            else:
                repeat_streak = 1
            repeat_max = max(repeat_max, repeat_streak)
            prev = norm

        # Replay for winner consistency
        conv2 = service.create_conversation(f"bench-replay-{seed}", gemini_enabled=True)
        winner_keys_2: list[str] = []
        for i, prompt in enumerate(prompts):
            asyncio.run(_run_turn(service, conv2.id, f"{seed}-r-{i}", prompt))
            trace = service.get_decision_trace(f"{seed}-r-{i}")
            if trace is not None:
                winner_keys_2.append(f"{trace.get('winner_provider')}:{trace.get('winner_role')}:{trace.get('winner_index')}")

        paired = min(len(winner_keys), len(winner_keys_2))
        consistency = (
            sum(1 for i in range(paired) if winner_keys[i] == winner_keys_2[i]) / paired if paired > 0 else 0.0
        )

        no_signal = sum(1 for o in outputs if o.strip().lower() == "no better signal")
        echo = sum(1 for o, p in zip(outputs, prompts) if _is_echo(o, p))
        judgment = statistics.mean(_score_expert_judgment(o) for o in outputs) if outputs else 0.0

        metrics = Metrics(
            no_signal_rate=no_signal / max(1, len(outputs)),
            echo_rate=echo / max(1, len(outputs)),
            repeat_streak_max=repeat_max,
            context_available_rate=context_ok / max(1, len(outputs)),
            liquid_source_rate=sum(1 for s in sources if s == "liquid_native") / max(1, len(sources)),
            median_latency_ms=statistics.median(latencies) if latencies else 0.0,
            winner_consistency=consistency,
            expert_judgment_score=judgment,
        )
        return {
            "seed": seed,
            "metrics": asdict(metrics),
        }


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
    jury_mode = bool(args.jury_mode)
    external_context = bool(args.external_context)
    learned_heads = bool(args.learned_heads)
    rows = [
        _run_seed(
            seed,
            args.steps,
            jury_mode=jury_mode,
            external_context=external_context,
            learned_heads=learned_heads,
            jury_head_path=args.jury_head_path,
            synthesis_head_path=args.synthesis_head_path,
        )
        for seed in seeds
    ]

    metric_names = list(Metrics.__annotations__.keys())
    summary = {}
    for name in metric_names:
        vals = [float(r["metrics"][name]) for r in rows]  # type: ignore[index]
        summary[name] = _summary(vals)

    result = {
        "config": {
            "steps": args.steps,
            "seeds": seeds,
            "jury_mode": jury_mode,
            "external_context": external_context,
            "learned_heads": learned_heads,
            "jury_head_path": args.jury_head_path,
            "synthesis_head_path": args.synthesis_head_path,
        },
        "rows": rows,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
