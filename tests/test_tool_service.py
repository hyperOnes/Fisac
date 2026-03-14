from __future__ import annotations

import pytest

from chat_api.config import Settings
from chat_api.db import init_db
from chat_api.repository import ChatRepository
from chat_api.services.tool_service import ToolService


def _tool_stack(tmp_path):
    settings = Settings(db_path=tmp_path / "tool_test.db", feature_dim=16, num_experts=8, top_k=2)
    init_db(settings)
    repo = ChatRepository(settings)
    return ToolService(repo=repo), repo


def test_math_eval_safe_expression(tmp_path) -> None:
    service, _ = _tool_stack(tmp_path)
    result = service.call_tool("math.eval", {"expression": "(2+3)*4 - 5"})
    assert result.ok
    assert result.output["result"] == pytest.approx(15.0)


def test_math_eval_rejects_unsafe(tmp_path) -> None:
    service, _ = _tool_stack(tmp_path)
    result = service.call_tool("math.eval", {"expression": "__import__('os').system('echo hi')"})
    assert not result.ok


def test_context_search_hits(tmp_path) -> None:
    service, repo = _tool_stack(tmp_path)
    conv = repo.create_conversation("c")
    repo.create_message(conversation_id=conv.id, role="user", content="alpha beta gamma", status="complete")
    repo.create_message(conversation_id=conv.id, role="assistant", content="delta alpha", status="complete")

    result = service.call_tool("context.search", {"conversation_id": conv.id, "query": "alpha"})
    assert result.ok
    assert result.output["count"] == 2


def test_parse_tool_command_shortcuts(tmp_path) -> None:
    service, _ = _tool_stack(tmp_path)
    parsed = service.parse_tool_command("/tool math.eval 2+2")
    assert parsed is not None
    name, args = parsed
    assert name == "math.eval"
    assert args["expression"] == "2+2"
