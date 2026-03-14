from __future__ import annotations

import asyncio
import sqlite3

from chat_api.config import Settings
from chat_api.db import init_db
from chat_api.repository import ChatRepository
from chat_api.services.chat_service import ChatService
from chat_api.services.context_window import ContextWindowPolicy
from chat_api.services.fiscal_text_bridge import FiscalTextBridge
from chat_api.services.gemini_client import GeminiAnswerResult, GeminiContextResult
from chat_api.services.openai_client import OpenAIAnswerResult, OpenAIContextResult
from chat_api.services.summary_service import SummaryService


class _GoodGemini:
    configured = True
    available = True

    async def extract_context(self, **_kwargs) -> GeminiContextResult:
        return GeminiContextResult(
            ok=True,
            context=(
                "Dynamic self-laid track at 100 km/h has high jamming risk. "
                "Existing wheeled EV supply chains are mature. "
                "Crewed Venus mission complexity remains extreme."
            ),
            attempts=1,
        )

    async def answer_candidate(self, *, role: str, **_kwargs) -> GeminiAnswerResult:
        candidates = {
            "direct_answer": (
                "The easier path is advanced wheeled EV development, not self-laid maglev. "
                "Self-laid track introduces moving infrastructure and severe failure risk at speed. "
                "Start with suspension and guidance upgrades before hover concepts."
            ),
            "comparative_feasibility": (
                "Near-term ranking is wheeled EV first, guided lane pod second, self-laid maglev third, "
                "crewed Venus mission last due to mission complexity."
            ),
            "failure_modes": (
                "Critical failures are segment misalignment, retraction jam, and thermal deformation under cyclic load."
            ),
            "implementation_sequence": (
                "Use phased validation: bench endurance, low-speed test lane, medium-speed fault injection, "
                "then high-speed safety-gated pilot."
            ),
            "skeptical_counterexample": (
                "If jam rate cannot stay below one per million cycles, road deployment is not viable."
            ),
        }
        return GeminiAnswerResult(ok=True, answer=candidates.get(role, candidates["direct_answer"]), role=role)


class _GoodOpenAI:
    configured = True
    available = True

    async def extract_context(self, **_kwargs) -> OpenAIContextResult:
        return OpenAIContextResult(ok=True, context="Maintenance burden dominates long-term viability.", attempts=1)

    async def answer_candidate(self, *, role: str, **_kwargs) -> OpenAIAnswerResult:
        return OpenAIAnswerResult(
            ok=True,
            answer="Pick the option with fewer moving subsystems and a faster certification path.",
            role=role,
        )


async def _run_turn(service: ChatService, conversation_id: str, run_id: str, text: str) -> str:
    chunks: list[str] = []
    async for evt in service.stream_reply(conversation_id=conversation_id, user_text=text, run_id=run_id):
        if evt.event == "token":
            chunks.append(evt.data["delta"])
    return "".join(chunks).strip()


def test_jury_mode_prefers_high_quality_external_candidate(tmp_path) -> None:
    settings = Settings(
        db_path=tmp_path / "jury_mode.db",
        feature_dim=16,
        num_experts=16,
        top_k=2,
        generation_backend="liquid_native",
        external_context_enabled=True,
        jury_mode=True,
    )
    init_db(settings)
    repo = ChatRepository(settings)
    bridge = FiscalTextBridge(repo=repo, settings=settings)
    service = ChatService(
        repo=repo,
        bridge=bridge,
        context_policy=ContextWindowPolicy(keep_last_messages=20, summary_every_user_turns=6),
        summary_service=SummaryService(max_sentences=2),
        gemini_client=_GoodGemini(),
        openai_client=_GoodOpenAI(),
    )

    conv = service.create_conversation("jury")
    output = asyncio.run(
        _run_turn(
            service,
            conv.id,
            "jury-run-1",
            "Maglev self-laid track at 100 km/h or crewed Venus mission: which is easier this decade?",
        )
    )
    assert output
    assert "wheeled ev" in output.lower()
    assert "based on the available context" not in output.lower()
    assert "user asks" not in output.lower()

    trace = service.get_decision_trace("jury-run-1")
    assert trace is not None
    assert int(trace.get("candidate_count", 0)) >= 3
    assert trace.get("winner_provider") in {"gemini", "openai", "fisac_truth_reasoner", "fisac_liquid_decode"}

    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT generation_source, candidate_count, winner_index, quality_flags FROM runs WHERE id = ?",
            ("jury-run-1",),
        ).fetchone()
    assert row is not None
    assert row[0] == "liquid_native"
    assert int(row[1]) >= 3
    assert row[2] is not None
