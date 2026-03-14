from __future__ import annotations

from pathlib import Path

import torch

from chat_api.models import MessageRecord
from chat_api.services.jury_scorer import JuryScorer
from chat_api.services.reasoning_orchestrator import ReasoningCandidate


def _encode(text: str) -> torch.Tensor:
    vec = torch.zeros(32, dtype=torch.float32)
    for tok in text.lower().split():
        vec[abs(hash(tok)) % 32] += 1.0
    n = vec.norm().clamp(min=1e-6)
    return (vec / n).unsqueeze(0)


def test_jury_scorer_applies_contradiction_penalty() -> None:
    scorer = JuryScorer()
    candidates = [
        ReasoningCandidate(
            role="direct_answer",
            provider="x",
            text="This is feasible and impossible, easier and harder at the same time.",
            score_hint=0.1,
        ),
        ReasoningCandidate(
            role="direct_answer",
            provider="x",
            text="This is feasible with phased testing and clear abort thresholds.",
            score_hint=0.1,
        ),
    ]
    best, ranked = scorer.choose_best(
        user_text="Is it feasible?",
        candidates=candidates,
        recent_messages=[],
        query_vec=_encode("feasible phased testing")[0],
        encode_text=_encode,
    )
    assert best is not None
    assert ranked[0].index == 1
    assert ranked[0].contradiction_penalty <= ranked[1].contradiction_penalty


def test_jury_scorer_loads_learned_head(tmp_path: Path) -> None:
    head_path = tmp_path / "jury_head.pt"
    torch.save(
        {
            "version": "test_head_v1",
            "feature_names": [
                "coverage",
                "echo",
                "actionability",
                "coherence",
                "alignment",
                "evidence",
                "contradiction_penalty",
                "meta_penalty",
                "generic_penalty",
                "role_bonus",
                "role_mismatch_penalty",
                "length_quality",
                "score_hint",
            ],
            "weights": [0.05] * 13,
            "bias": 0.01,
        },
        head_path,
    )
    scorer = JuryScorer(learned_head_path=str(head_path), learned_heads_enabled=True)
    assert scorer.head_version == "test_head_v1"
    msg = MessageRecord(
        id="m1",
        conversation_id="c1",
        role="assistant",
        content="previous answer",
        status="complete",
        created_at="2026-01-01T00:00:00+00:00",
    )
    best, _ = scorer.choose_best(
        user_text="give implementation steps",
        candidates=[ReasoningCandidate(role="implementation_sequence", text="Phase 1 test. Phase 2 validate.", provider="x")],
        recent_messages=[msg],
        query_vec=_encode("phase test validate")[0],
        encode_text=_encode,
    )
    assert best is not None
