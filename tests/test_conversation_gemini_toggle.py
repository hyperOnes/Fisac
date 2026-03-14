from __future__ import annotations

import asyncio
import sqlite3

from chat_api.config import Settings
from chat_api.db import init_db
from chat_api.repository import ChatRepository
from chat_api.services.chat_service import ChatService
from chat_api.services.context_window import ContextWindowPolicy
from chat_api.services.conversational_composer import ConversationalComposer
from chat_api.services.fiscal_text_bridge import FiscalTextBridge
from chat_api.services.gemini_client import GeminiContextResult
from chat_api.services.openai_client import OpenAIContextResult
from chat_api.services.summary_service import SummaryService
from chat_api.services.tool_service import ToolService


class _FakeGeminiClient:
    def __init__(self, *, ok: bool = True, context: str = "constraints, tradeoffs", error: str | None = None) -> None:
        self._ok = ok
        self._context = context
        self._error = error
        self.context_calls = 0
        self.configured = True
        self.available = True

    async def extract_context(self, **_kwargs) -> GeminiContextResult:
        self.context_calls += 1
        return GeminiContextResult(ok=self._ok, context=self._context if self._ok else "", error=self._error, attempts=1)


class _FakeOpenAIClient:
    def __init__(self, *, ok: bool = True, context: str = "constraints, tradeoffs", error: str | None = None) -> None:
        self._ok = ok
        self._context = context
        self._error = error
        self.context_calls = 0
        self.configured = True
        self.available = True

    async def extract_context(self, **_kwargs) -> OpenAIContextResult:
        self.context_calls += 1
        return OpenAIContextResult(ok=self._ok, context=self._context if self._ok else "", error=self._error, attempts=1)


def _make_service(tmp_path, gemini_client: _FakeGeminiClient, openai_client: _FakeOpenAIClient | None = None):
    settings = Settings(
        db_path=tmp_path / "fiscal_chat_toggle.db",
        feature_dim=16,
        num_experts=16,
        top_k=2,
        checkpoint_every_turns=3,
        lifecycle_interval_turns=100,
        generation_backend="liquid_native",
        gemini_context_only=True,
        context_query_count=5,
    )
    init_db(settings)
    repo = ChatRepository(settings)
    bridge = FiscalTextBridge(repo=repo, settings=settings)
    bridge.load_or_init()
    service = ChatService(
        repo=repo,
        bridge=bridge,
        context_policy=ContextWindowPolicy(keep_last_messages=20, summary_every_user_turns=6),
        summary_service=SummaryService(max_sentences=2),
        tool_service=ToolService(repo=repo),
        composer=ConversationalComposer(),
        gemini_client=gemini_client,
        openai_client=openai_client,
    )
    return settings, repo, service


async def _run_turn(service: ChatService, conversation_id: str, run_id: str, text: str = "hello") -> list[str]:
    events: list[str] = []
    async for evt in service.stream_reply(conversation_id=conversation_id, user_text=text, run_id=run_id):
        events.append(evt.event)
    return events


def test_conversation_toggle_controls_gemini_context_usage(tmp_path) -> None:
    fake = _FakeGeminiClient(ok=True, context="mobility, reliability, failure modes")
    _, repo, service = _make_service(tmp_path, fake)
    conv = service.create_conversation("toggle", gemini_enabled=True)
    assert conv.gemini_enabled is True

    events_on = asyncio.run(_run_turn(service, conv.id, "r-on"))
    assert "done" in events_on
    assert 1 <= fake.context_calls <= 5

    updated = service.update_conversation(conv.id, gemini_enabled=False)
    assert updated is not None
    assert updated.gemini_enabled is False

    events_off = asyncio.run(_run_turn(service, conv.id, "r-off"))
    assert "done" in events_off
    assert 1 <= fake.context_calls <= 5

    # Repository default for new conversations is ON.
    conv2 = repo.create_conversation("default")
    assert conv2.gemini_enabled is True


def test_gemini_context_failure_keeps_liquid_generation_and_records_flag(tmp_path) -> None:
    fake = _FakeGeminiClient(ok=False, error="quota")
    settings, _, service = _make_service(tmp_path, fake)
    conv = service.create_conversation("context-fail", gemini_enabled=True)

    events = asyncio.run(_run_turn(service, conv.id, "r-context-fail"))
    assert "done" in events
    assert fake.context_calls == 5

    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT generation_source, error_code, quality_flags FROM runs WHERE id = ?",
            ("r-context-fail",),
        ).fetchone()
    assert row is not None
    assert row[0] == "liquid_native"
    assert row[1] is None
    assert "external_context_local_fallback" in (row[2] or "")


def test_liquid_source_persisted_even_when_gemini_context_used(tmp_path) -> None:
    fake = _FakeGeminiClient(ok=True, context="constraints, budget, timeline")
    settings, _, service = _make_service(tmp_path, fake)
    conv = service.create_conversation("liquid-source", gemini_enabled=True)

    events = asyncio.run(_run_turn(service, conv.id, "r-liquid", text="Explain tradeoffs"))
    assert "done" in events

    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT generation_source, generation_attempts, quality_flags FROM runs WHERE id = ?",
            ("r-liquid",),
        ).fetchone()
    assert row is not None
    assert row[0] == "liquid_native"
    assert int(row[1]) == 1
    assert "external_context_used" in (row[2] or "")


def test_context_probe_falls_back_to_openai_when_gemini_fails(tmp_path) -> None:
    fake_gemini = _FakeGeminiClient(ok=False, error="timeout")
    fake_openai = _FakeOpenAIClient(ok=True, context="mobility constraints, failure modes, safer alternatives")
    settings, _, service = _make_service(tmp_path, fake_gemini, fake_openai)
    conv = service.create_conversation("context-fallback", gemini_enabled=True)

    events = asyncio.run(_run_turn(service, conv.id, "r-fallback", text="compare options"))
    assert "done" in events
    assert 1 <= fake_gemini.context_calls <= 5
    assert 1 <= fake_openai.context_calls <= 5

    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT generation_source, error_code, quality_flags FROM runs WHERE id = ?",
            ("r-fallback",),
        ).fetchone()
    assert row is not None
    assert row[0] == "liquid_native"
    assert row[1] is None
    assert "external_context_used" in (row[2] or "")


def test_local_context_fallback_when_no_provider_configured(tmp_path) -> None:
    fake = _FakeGeminiClient(ok=True, context="unused")
    settings, _, service = _make_service(tmp_path, fake)
    # Disable all external providers at runtime for this scenario.
    service.gemini_client = None
    service.openai_client = None
    conv = service.create_conversation("local-fallback", gemini_enabled=True)

    events = asyncio.run(_run_turn(service, conv.id, "r-local-fallback", text="hello there"))
    assert "done" in events

    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT generation_source, error_code, quality_flags FROM runs WHERE id = ?",
            ("r-local-fallback",),
        ).fetchone()
    assert row is not None
    assert row[0] == "liquid_native"
    assert row[1] is None
    assert "external_context_local_fallback" in (row[2] or "")


def test_db_migration_adds_and_backfills_gemini_enabled(tmp_path) -> None:
    db_path = tmp_path / "legacy_schema.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("c1", "legacy", "2026-02-25T00:00:00Z", "2026-02-25T00:00:00Z"),
        )
        conn.commit()

    settings = Settings(db_path=db_path)
    init_db(settings)

    with sqlite3.connect(db_path) as conn:
        cols = conn.execute("PRAGMA table_info(conversations)").fetchall()
        names = {row[1] for row in cols}
        assert "gemini_enabled" in names
        value = conn.execute("SELECT gemini_enabled FROM conversations WHERE id = 'c1'").fetchone()
    assert value is not None
    assert int(value[0]) == 1


def test_db_migration_adds_runs_generation_columns(tmp_path) -> None:
    db_path = tmp_path / "legacy_runs.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                mse REAL NOT NULL,
                confidence REAL NOT NULL,
                pruned_now INTEGER NOT NULL,
                myelinated_now INTEGER NOT NULL,
                error_code TEXT NULL,
                error_message TEXT NULL
            )
            """
        )
        conn.commit()

    settings = Settings(db_path=db_path)
    init_db(settings)

    with sqlite3.connect(db_path) as conn:
        cols = conn.execute("PRAGMA table_info(runs)").fetchall()
        names = {row[1] for row in cols}
    assert "generation_source" in names
    assert "generation_attempts" in names
    assert "quality_flags" in names
    assert "output_chars" in names
