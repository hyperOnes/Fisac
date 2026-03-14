from __future__ import annotations

from pathlib import Path

from chat_api.config import Settings
from chat_api.db import init_db
from chat_api.repository import ChatRepository
from chat_api.services.chat_service import ChatService
from chat_api.services.context_window import ContextWindowPolicy
from chat_api.services.conversational_composer import ConversationalComposer
from chat_api.services.fiscal_text_bridge import FiscalTextBridge
from chat_api.services.summary_service import SummaryService
from chat_api.services.tool_service import ToolService


def _make_service(tmp_db: Path) -> ChatService:
    settings = Settings(
        db_path=tmp_db,
        feature_dim=16,
        num_experts=16,
        top_k=2,
        jury_mode=True,
        external_context_enabled=True,
    )
    init_db(settings)
    repo = ChatRepository(settings)
    bridge = FiscalTextBridge(repo=repo, settings=settings)
    bridge.load_or_init()
    return ChatService(
        repo=repo,
        bridge=bridge,
        context_policy=ContextWindowPolicy(keep_last_messages=20, summary_every_user_turns=6),
        summary_service=SummaryService(max_sentences=3),
        tool_service=ToolService(repo=repo),
        composer=ConversationalComposer(),
    )


def test_context_merge_is_deterministic_and_dedupes(tmp_path: Path) -> None:
    svc = _make_service(tmp_path / "ctx_merge.db")
    user = "How do I test self laying maglev track safely?"
    chunks = [
        "Top risk is track retraction jam. Use phased speed ramps.",
        "Top risk is track retraction jam. Use phased speed ramps.",
        "Based on the available context, user asks for safety guidance.",
        "Define abort thresholds before speed escalation.",
    ]
    merged1 = svc._merge_context_chunks(chunks, user_text=user)
    merged2 = svc._merge_context_chunks(chunks, user_text=user)
    assert merged1 == merged2
    assert "user asks" not in merged1.lower()
    assert merged1.lower().count("track retraction jam") <= 1
    assert "abort thresholds" in merged1.lower()


def test_context_cache_key_includes_summary(tmp_path: Path) -> None:
    svc = _make_service(tmp_path / "ctx_cache.db")
    user = "Compare option A versus option B"
    key_a = svc._cache_key(user, "summary one")
    key_b = svc._cache_key(user, "summary two")
    assert key_a != key_b
