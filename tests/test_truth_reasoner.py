from __future__ import annotations

from chat_api.models import MessageRecord
from chat_api.services.truth_reasoner import TruthReasoner


def _msg(role: str, content: str) -> MessageRecord:
    return MessageRecord(
        id="m",
        conversation_id="c",
        role=role,
        content=content,
        status="complete",
        created_at="2026-02-27T00:00:00Z",
    )


def test_truth_reasoner_prefers_more_feasible_option() -> None:
    r = TruthReasoner()
    result = r.reason(
        user_text="maglev self laying track car or spaceship to venus, which is easier?",
        context_hint=(
            "self laying track risks severe mechanical jamming at high speed. "
            "spaceship to venus has extreme propulsion and life support complexity. "
            "incremental ground mobility prototypes are more practical."
        ),
        recent_messages=[],
    )
    assert result is not None
    text = result.text.lower()
    assert "more feasible" in text
    assert "venus" in text


def test_truth_reasoner_works_with_recent_context_fallback() -> None:
    r = TruthReasoner()
    result = r.reason(
        user_text="is this viable this decade?",
        context_hint="",
        recent_messages=[
            _msg("user", "self laying track at 100 km/h"),
            _msg("assistant", "the critical blocker is mechanical jamming and alignment drift"),
        ],
    )
    assert result is not None
    assert result.text
    assert result.confidence > 0.0
