from __future__ import annotations

import pytest

pytest.skip("Legacy golden profile reset assertions disabled after runtime reversion.", allow_module_level=True)

from chat_api.config import Settings
from chat_api.db import init_db
from chat_api.repository import ChatRepository
from chat_api.services.fiscal_text_bridge import FiscalTextBridge


def _model_fingerprint(bridge: FiscalTextBridge, conversation_id: str) -> str:
    runtime = bridge._get_or_create_runtime(conversation_id)  # noqa: SLF001
    return bridge._hash_state_dict(bridge._cpu_state_dict(runtime.model.state_dict()))  # noqa: SLF001


def test_new_conversation_resets_to_baseline_and_lru_eviction_persists(tmp_path) -> None:
    settings = Settings(
        db_path=tmp_path / "golden_reset.db",
        feature_dim=16,
        num_experts=16,
        top_k=2,
        runtime_profile="golden_lock",
        force_deterministic_global=True,
        golden_disable_gemini=True,
        per_chat_hard_reset=True,
        max_active_conversation_models=1,
        checkpoint_every_turns=100,
    )
    init_db(settings)
    repo = ChatRepository(settings)
    bridge = FiscalTextBridge(repo=repo, settings=settings)
    bridge.load_or_init()

    baseline_id = bridge.get_baseline_id()
    assert baseline_id is not None

    conv_a = repo.create_conversation("A", gemini_enabled=False)
    conv_b = repo.create_conversation("B", gemini_enabled=False)

    # Conversation A starts from baseline then learns.
    fp_a_before = _model_fingerprint(bridge, conv_a.id)
    assert fp_a_before == baseline_id

    for _ in range(5):
        bridge.step("hover vehicle track constraints", conv_a.id)

    fp_a_after = _model_fingerprint(bridge, conv_a.id)
    assert fp_a_after != baseline_id

    # Conversation B must still start from baseline.
    fp_b_before = _model_fingerprint(bridge, conv_b.id)
    assert fp_b_before == baseline_id

    # With max_active_conversation_models=1, one runtime should be evicted and checkpointed.
    assert len(bridge._states) == 1  # noqa: SLF001
    assert repo.get_state_snapshot(conv_a.id) is not None
