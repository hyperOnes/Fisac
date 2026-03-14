from __future__ import annotations

import json
from pathlib import Path

from tools import recover_golden_window as rgw
from tools import replay_golden_window as replay


def _create_fallback_tree(root: Path) -> None:
    for rel in rgw.REQUIRED_FILES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"content for {rel}\n", encoding="utf-8")
    db = root / "chat_api" / "fiscal_chat.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"sqlite-placeholder")


def test_recover_candidates_uses_fallback_and_writes_manifest(tmp_path, monkeypatch) -> None:
    fallback = tmp_path / "fallback"
    output = tmp_path / "out"
    _create_fallback_tree(fallback)

    monkeypatch.setattr(rgw, "list_local_snapshots", lambda: [])
    bundles = rgw.recover_candidates(
        target_iso="2026-02-25T21:51:21+01:00",
        output_dir=output,
        fallback_source=fallback,
    )

    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle.source_kind == "fallback_backup"

    manifest = json.loads((bundle.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_kind"] == "fallback_backup"
    paths = {f["path"] for f in manifest["files"]}
    for rel in rgw.REQUIRED_FILES:
        assert str(rel) in paths

    sample = bundle.path / rgw.REQUIRED_FILES[0]
    h1 = rgw._sha256(sample)  # noqa: SLF001
    h2 = rgw._sha256(sample)  # noqa: SLF001
    assert h1 == h2


def test_replay_turn_eval_flags_template_labels() -> None:
    turn = replay._evaluate_turn(  # noqa: SLF001
        prompt="Explain",
        output="Answer: Why: Next steps: this is bad",
        latency_ms=150.0,
    )
    assert turn.label_ok is False
