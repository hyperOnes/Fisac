from __future__ import annotations

import asyncio
import sqlite3

import pytest

pytest.skip("Legacy strict/golden lock assertions disabled after runtime reversion.", allow_module_level=True)

from chat_api.config import Settings
from chat_api.db import init_db
from chat_api.repository import ChatRepository
from chat_api.services.chat_service import ChatService
from chat_api.services.context_window import ContextWindowPolicy
from chat_api.services.conversational_composer import ConversationalComposer
from chat_api.services.fiscal_text_bridge import FiscalTextBridge
from chat_api.services.gemini_client import GeminiResult
from chat_api.services.summary_service import SummaryService


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.calls = 0
        self.configured = True
        self.available = True

    async def generate(self, **_kwargs) -> GeminiResult:
        self.calls += 1
        return GeminiResult(ok=True, text="gemini text", attempts=1)


async def _run_turn(service: ChatService, conversation_id: str, run_id: str, text: str) -> list[str]:
    events: list[str] = []
    async for evt in service.stream_reply(conversation_id=conversation_id, user_text=text, run_id=run_id):
        events.append(evt.event)
    return events


def test_golden_lock_bypasses_gemini_and_forces_conversation_toggle(tmp_path) -> None:
    settings = Settings(
        db_path=tmp_path / "golden_lock.db",
        feature_dim=16,
        num_experts=16,
        top_k=2,
        runtime_profile="golden_lock",
        force_deterministic_global=True,
        golden_disable_gemini=True,
        per_chat_hard_reset=True,
        max_active_conversation_models=4,
    )
    init_db(settings)
    repo = ChatRepository(settings)
    bridge = FiscalTextBridge(repo=repo, settings=settings)
    bridge.load_or_init()

    fake_gemini = _FakeGeminiClient()
    service = ChatService(
        repo=repo,
        bridge=bridge,
        context_policy=ContextWindowPolicy(keep_last_messages=20, summary_every_user_turns=6),
        summary_service=SummaryService(max_sentences=2),
        composer=ConversationalComposer(),
        gemini_client=fake_gemini,
    )

    conv = service.create_conversation("golden", gemini_enabled=True)
    assert conv.gemini_enabled is False

    events = asyncio.run(_run_turn(service, conv.id, "run-lock-1", "Explain"))
    assert "done" in events
    assert fake_gemini.calls == 0

    updated = service.update_conversation(conv.id, gemini_enabled=True)
    assert updated is not None
    assert updated.gemini_enabled is False

    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT generation_source, runtime_profile FROM runs WHERE id = ?",
            ("run-lock-1",),
        ).fetchone()
    assert row is not None
    assert row[0] in {"deterministic", "liquid_native"}
    assert row[1] == "golden_lock"
