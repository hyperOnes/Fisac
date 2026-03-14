from __future__ import annotations

import asyncio
import math

import pytest

torch = pytest.importorskip("torch")

from chat_api.config import Settings
from chat_api.db import init_db
from chat_api.models import MessageRecord
from chat_api.repository import ChatRepository
from chat_api.services.chat_service import ChatService
from chat_api.services.context_window import ContextWindowPolicy
from chat_api.services.fiscal_text_bridge import FiscalTextBridge
from chat_api.services.sse import SSEEvent, format_sse
from chat_api.services.summary_service import SummaryService
from chat_api.services.tool_service import ToolService


def _make_stack(tmp_path):
    settings = Settings(
        db_path=tmp_path / "fiscal_chat_test.db",
        feature_dim=16,
        num_experts=16,
        top_k=2,
        checkpoint_every_turns=3,
        lifecycle_interval_turns=100,
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
    )
    return settings, repo, bridge, service


def test_text_encoder_determinism_and_bounds(tmp_path) -> None:
    _, _, bridge, _ = _make_stack(tmp_path)

    v1 = bridge.encode_text("Fiscal deterministic encoding")
    v2 = bridge.encode_text("Fiscal deterministic encoding")

    assert v1.shape == (1, 16)
    assert torch.equal(v1, v2)
    assert torch.all(v1 <= 1.0)
    assert torch.all(v1 >= -1.0)


def test_response_memory_ranking_selects_nearest(tmp_path) -> None:
    _, repo, bridge, _ = _make_stack(tmp_path)
    conv = repo.create_conversation("ranking")

    alpha_vec = bridge.encode_text("alpha signal")[0]
    beta_vec = bridge.encode_text("beta signal")[0]

    repo.add_response_memory(
        conversation_id=conv.id,
        user_text="alpha",
        assistant_text="Alpha response stays stable under constraints and ends with a clear recommendation for the next step.",
        user_vec=bridge._tensor_to_blob(alpha_vec),  # noqa: SLF001
        assistant_vec=bridge._tensor_to_blob(alpha_vec),  # noqa: SLF001
        confidence=0.9,
    )
    repo.add_response_memory(
        conversation_id=conv.id,
        user_text="beta",
        assistant_text="Beta response focuses on a different strategy while still giving concrete implementation guidance.",
        user_vec=bridge._tensor_to_blob(beta_vec),  # noqa: SLF001
        assistant_vec=bridge._tensor_to_blob(beta_vec),  # noqa: SLF001
        confidence=0.9,
    )

    ranked = bridge._rank_response_memory(conv.id, alpha_vec)  # noqa: SLF001
    assert ranked
    assert "Alpha response" in ranked[0][1]


def test_context_window_and_summary_policy() -> None:
    policy = ContextWindowPolicy(keep_last_messages=20, summary_every_user_turns=6)
    messages = [
        MessageRecord(
            id=str(i),
            conversation_id="c1",
            role="user" if i % 2 == 0 else "assistant",
            content=f"message {i}",
            status="complete",
            created_at="2026-02-25T00:00:00Z",
        )
        for i in range(30)
    ]

    trimmed = policy.trim_messages(messages)
    assert len(trimmed) == 20
    assert trimmed[0].id == "10"
    assert policy.should_refresh_summary(6)
    assert not policy.should_refresh_summary(5)

    summary = SummaryService(max_sentences=2).update_summary(trimmed)
    assert summary


def test_sse_formatter_emits_valid_frames() -> None:
    payload = format_sse(SSEEvent(event="token", data={"run_id": "r1", "delta": "abc", "index": 0})).decode("utf-8")
    assert payload.startswith("event: token\n")
    assert "data:" in payload
    assert payload.endswith("\n\n")


def test_chat_service_stream_and_persistence(tmp_path) -> None:
    _, repo, bridge, service = _make_stack(tmp_path)
    conv = service.create_conversation("stream")

    async def collect_events():
        out = []
        async for event in service.stream_reply(conversation_id=conv.id, user_text="hello fiscal", run_id="run-1"):
            out.append(event)
        return out

    events = asyncio.run(collect_events())
    event_names = [e.event for e in events]
    assert "ack" in event_names
    assert "done" in event_names
    assert "error" not in event_names

    messages = repo.list_messages(conv.id, limit=200)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[1].latency_ms is not None
    assert messages[1].confidence is not None
    assert messages[1].mse is not None
    assert messages[1].generation_source == "liquid_native"

    bridge.persist_all()
    snapshot = repo.get_state_snapshot(conv.id)
    assert snapshot is not None


def test_summary_refresh_every_6_user_turns(tmp_path) -> None:
    _, _, bridge, service = _make_stack(tmp_path)
    conv = service.create_conversation("summary")

    async def one_turn(i: int):
        async for _ in service.stream_reply(
            conversation_id=conv.id,
            user_text=f"user turn {i} with repeated salient phrase adaptation target",
            run_id=f"sum-{i}",
        ):
            pass

    for i in range(6):
        asyncio.run(one_turn(i))

    summary = bridge.get_summary(conv.id)
    assert summary.strip()


def test_tool_command_executes_and_persists(tmp_path) -> None:
    _, repo, _, service = _make_stack(tmp_path)
    conv = service.create_conversation("tooling")

    async def collect_events():
        out = []
        async for event in service.stream_reply(
            conversation_id=conv.id,
            user_text="/tool math.eval (2 + 3) * 4",
            run_id="tool-run-1",
        ):
            out.append(event)
        return out

    events = asyncio.run(collect_events())
    assert any(e.event == "done" for e in events)
    messages = repo.list_messages(conv.id, limit=50)
    assert messages[-1].role == "assistant"
    assert "Tool `math.eval` result" in messages[-1].content

    calls = repo.list_tool_calls(conv.id, limit=20)
    assert calls
    assert calls[-1].tool_name == "math.eval"
    assert calls[-1].ok == 1


def test_balanced_response_not_clipped_for_open_ended_prompt(tmp_path) -> None:
    _, repo, _, service = _make_stack(tmp_path)
    conv = service.create_conversation("depth")

    async def collect():
        out = []
        async for event in service.stream_reply(
            conversation_id=conv.id,
            user_text="What do you advise instead for a realistic vehicle concept and why?",
            run_id="depth-run-1",
        ):
            out.append(event)
        return out

    events = asyncio.run(collect())
    assert any(e.event == "done" for e in events)
    msg = [m for m in repo.list_messages(conv.id, limit=20) if m.role == "assistant"][-1]
    assert len(msg.content) >= 1
    assert msg.content.lower() == "no better signal" or msg.content.rstrip()[-1] in ".!?"


def test_regression_prompts_do_not_return_clipped_one_liners(tmp_path) -> None:
    _, repo, _, service = _make_stack(tmp_path)
    conv = service.create_conversation("regression")
    prompts = [
        "Explain",
        "Is it shit idea",
        "What do you advice instead?",
    ]

    async def one_turn(idx: int, prompt: str):
        async for _ in service.stream_reply(
            conversation_id=conv.id,
            user_text=prompt,
            run_id=f"reg-{idx}",
        ):
            pass

    for idx, prompt in enumerate(prompts):
        asyncio.run(one_turn(idx, prompt))

    assistant = [m for m in repo.list_messages(conv.id, limit=100) if m.role == "assistant"]
    assert len(assistant) == 3
    for msg in assistant:
        assert len(msg.content) >= 1
        assert msg.content.lower() == "no better signal" or msg.content.rstrip()[-1] in ".!?"


@pytest.mark.slow
def test_chat_soak_100_turns_no_nan(tmp_path) -> None:
    _, repo, _, service = _make_stack(tmp_path)
    conv = service.create_conversation("soak")

    async def run_turn(turn: int):
        events = []
        async for event in service.stream_reply(
            conversation_id=conv.id,
            user_text=f"turn {turn}: track adaptation",
            run_id=f"run-{turn}",
        ):
            events.append(event)
        return events

    for i in range(100):
        events = asyncio.run(run_turn(i))
        names = [e.event for e in events]
        assert "done" in names
        assert "error" not in names

    assistant_messages = [m for m in repo.list_messages(conv.id, limit=500) if m.role == "assistant"]
    assert len(assistant_messages) == 100
    for msg in assistant_messages:
        if msg.mse is not None:
            assert math.isfinite(msg.mse)


@pytest.mark.slow
def test_chat_latency_median_under_1500ms(tmp_path) -> None:
    _, repo, _, service = _make_stack(tmp_path)
    conv = service.create_conversation("latency")

    async def run_turn(turn: int):
        async for _ in service.stream_reply(
            conversation_id=conv.id,
            user_text=f"latency turn {turn}",
            run_id=f"lat-{turn}",
        ):
            pass

    for i in range(20):
        asyncio.run(run_turn(i))

    assistant = [m for m in repo.list_messages(conv.id, limit=200) if m.role == "assistant" and m.latency_ms is not None]
    assert assistant
    latencies = sorted(float(m.latency_ms) for m in assistant if m.latency_ms is not None)
    median = latencies[len(latencies) // 2]
    assert median < 1500.0
