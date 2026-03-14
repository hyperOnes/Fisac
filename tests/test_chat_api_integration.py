from __future__ import annotations

import importlib
import json
import os
import sqlite3
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


def _load_app_with_db(db_path: str):
    os.environ["FISCAL_CHAT_DB"] = db_path
    for mod in ["chat_api.main", "chat_api.config"]:
        if mod in sys.modules:
            del sys.modules[mod]
    import chat_api.main as main

    importlib.reload(main)
    return main


def _parse_sse_text(raw: str) -> list[tuple[str, dict]]:
    events = []
    for frame in raw.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        event = None
        data = None
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        if event and data is not None:
            events.append((event, data))
    return events


def test_fastapi_chat_roundtrip_and_restore(tmp_path) -> None:
    db_path = str(tmp_path / "chat_api_test.db")
    main = _load_app_with_db(db_path)

    with TestClient(main.app) as client:
        tools = client.get("/api/tools").json()["items"]
        assert any(t["name"] == "math.eval" for t in tools)
        direct_tool = client.post("/api/tools/call", json={"tool": "math.eval", "args": {"expression": "3*7"}}).json()
        assert direct_tool["ok"] is True
        assert direct_tool["output"]["result"] == 21.0

        created = client.post("/api/conversations", json={"title": "API Test"}).json()
        cid = created["id"]
        assert created["gemini_enabled"] is True

        toggled = client.patch(f"/api/conversations/{cid}", json={"gemini_enabled": False}).json()
        assert toggled["gemini_enabled"] is False

        resp = client.post(
            "/api/chat/respond",
            json={"conversation_id": cid, "message": "hello", "stream": True},
        )
        assert resp.status_code == 200
        events = _parse_sse_text(resp.text)
        names = [name for name, _ in events]
        assert "ack" in names
        assert "done" in names

        tool_resp = client.post(
            "/api/chat/respond",
            json={"conversation_id": cid, "message": "/tool math.eval 2+5", "stream": True},
        )
        assert tool_resp.status_code == 200
        tool_events = _parse_sse_text(tool_resp.text)
        assert "done" in [name for name, _ in tool_events]
        tool_calls = client.get(f"/api/conversations/{cid}/tool-calls").json()["items"]
        assert tool_calls
        assert tool_calls[-1]["tool_name"] == "math.eval"

        messages = client.get(f"/api/conversations/{cid}/messages").json()["items"]
        assert len(messages) >= 2
        assistant = [m for m in messages if m["role"] == "assistant"][-1]
        assert "generation_source" in assistant
        assert "generation_attempts" in assistant

    # Restart app and ensure history still present.
    main2 = _load_app_with_db(db_path)
    with TestClient(main2.app) as client2:
        listed = client2.get("/api/conversations").json()["items"]
        assert listed
        cid = listed[0]["id"]
        messages = client2.get(f"/api/conversations/{cid}/messages").json()["items"]
        assert messages

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM state_snapshots").fetchone()
        run = conn.execute(
            "SELECT generation_source, generation_attempts, output_chars FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    assert row is not None and row[0] >= 1
    assert run is not None
    assert run[0] in {"liquid_native", "deterministic", "gemini_raw"}
    assert int(run[1]) >= 1
    assert int(run[2]) >= 0
