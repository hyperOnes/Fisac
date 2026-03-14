from __future__ import annotations

import asyncio

from chat_api.config import Settings
from chat_api.services.gemini_client import GeminiClient


def _settings() -> Settings:
    return Settings(
        gemini_api_key="test-key",
        response_guard_enabled=True,
        response_depth_mode="balanced",
        response_min_words_balanced=70,
        response_min_sentences_balanced=3,
        response_regen_attempts=1,
    )


def test_quality_evaluator_passes_complete_balanced_text() -> None:
    client = GeminiClient(settings=_settings())
    text = (
        "Answer: This approach is feasible if you reduce scope to one deploy mechanism and test it at low speed first, "
        "while keeping interfaces modular so you can replace failing parts without redesigning the whole system. "
        "Why: The highest risk is mechanical reliability under vibration and weather, so you should validate durability "
        "before performance and monitor failure trends across repeated cycles to expose weak points. "
        "Next steps: Define one success metric, build a bench prototype, run repeated deployment cycles, document every "
        "failure mode, then scale velocity only after stability and maintenance effort stay within your constraints."
    )
    q = client.evaluate_output(text)
    assert q.is_valid is True
    assert "too_short" not in q.flags


def test_quality_evaluator_fails_short_or_incomplete_text() -> None:
    client = GeminiClient(settings=_settings())
    q = client.evaluate_output("Short answer and")
    assert q.is_valid is False
    assert "too_short" in q.flags
    assert "incomplete_tail" in q.flags


def test_generate_retries_once_when_first_output_fails_quality(monkeypatch) -> None:
    client = GeminiClient(settings=_settings())
    calls = {"count": 0}

    async def fake_request_once(_prompt: str):
        calls["count"] += 1
        if calls["count"] == 1:
            return "Too short and", None
        return (
            "Answer: A realistic route is to avoid track deployment and instead use high-speed active suspension with strict safety envelopes, "
            "because it keeps the architecture simpler and easier to harden under road variability. "
            "Why: It reduces moving parts and keeps failure modes manageable while still allowing performance improvements over time, "
            "especially if you validate controller behavior under abrupt maneuvers and degraded conditions. "
            "Next steps: Start with a low-speed prototype, validate control stability, instrument fault cases, "
            "and gate progression through measurable safety milestones before increasing target speed.",
            None,
        )

    monkeypatch.setattr(client, "_request_once", fake_request_once)
    result = asyncio.run(
        client.generate(
            user_text="Explain",
            deterministic_draft="Answer: Base draft. Why: Base reason. Next steps: Base next steps.",
            summary="",
            recent_messages=[],
        )
    )
    assert result.ok is True
    assert result.attempts == 2
    assert calls["count"] == 2
