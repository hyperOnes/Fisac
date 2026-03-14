from __future__ import annotations

from chat_api.models import MessageRecord
from chat_api.services.conversational_composer import ComposeInput, ConversationalComposer


def test_composer_sanitizes_duplicate_leadins() -> None:
    composer = ConversationalComposer()
    raw = (
        "Here is what I infer: Here is what I infer: "
        "I processed your message and updated my internal state. "
        "I processed your message and updated my internal state."
    )
    cleaned = composer.sanitize_generated(raw)
    assert cleaned.lower().count("here is what i infer:") <= 1
    assert cleaned.lower().count("i processed your message and updated my internal state") <= 1


def test_composer_ignores_low_signal_bridge_text() -> None:
    composer = ConversationalComposer()
    out = composer.compose(
        ComposeInput(
            user_text="Give me options for deployment",
            bridge_text="I processed your message and updated my internal state.",
            summary="",
            recent_messages=[
                MessageRecord(
                    id="1",
                    conversation_id="c",
                    role="user",
                    content="Give me options",
                    status="complete",
                    created_at="2026-02-25T00:00:00Z",
                )
            ],
        )
    )
    assert "i processed your message and updated my internal state" not in out.lower()
    assert out.lower() == "no better signal"


def test_choose_generated_or_fallback_rejects_near_duplicate() -> None:
    composer = ConversationalComposer()
    deterministic = "Main tradeoff: latency versus memory. I can break this down with risks and implementation steps."
    generated = "Main tradeoff latency versus memory. I can break this down with risks and implementation steps."
    chosen = composer.choose_generated_or_fallback(generated=generated, deterministic=deterministic)
    assert chosen == composer.sanitize_generated(deterministic)


def test_composer_does_not_truncate_clean_tail() -> None:
    composer = ConversationalComposer()
    text = (
        "Answer: This is a complete sentence with a recommendation. "
        "Why: The constraints are clear enough to proceed. "
        "Next steps: Build a small prototype and test one metric first."
    )
    cleaned = composer.sanitize_generated(text)
    assert cleaned.endswith(".")
    assert "prototype" in cleaned.lower()


def test_is_incomplete_tail_flags_unfinished_output() -> None:
    composer = ConversationalComposer()
    assert composer.is_incomplete_tail("This is incomplete and")
    assert composer.is_incomplete_tail("Next steps:")
    assert not composer.is_incomplete_tail("This is complete.")
