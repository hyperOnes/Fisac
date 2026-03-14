# Golden Mode

## Goal
Reproduce and lock Fisac behavior close to the Feb 25, 2026 21:51-21:53 CET deterministic "golden window" style.

## Runtime Profile
`golden_lock` enforces:

1. Deterministic global mode (Gemini path bypassed).
2. Per-conversation hard reset from a fixed baseline checkpoint.
3. Max online learning within a conversation.
4. Conversation-local memory retrieval (no cross-chat fallback).
5. Confidence clamped to configured golden bounds.

## Baseline Checkpoint
Default path:

`/Users/sebastian/Fisac/artifacts/golden_recovery/best/golden_baseline.pt`

If the file is missing, the bridge creates one from current initialized model weights and computes a baseline hash.

## Recovery + Scoring Flow
Run:

```bash
./scripts/recover_and_score_golden.sh
```

This runs:

1. `tools/recover_golden_window.py` to create candidate bundles from Time Machine snapshots (or fallback backup).
2. `tools/replay_golden_window.py` to replay canonical prompts, score style-quality, and pick best candidate.
3. Copies best bundle to `artifacts/golden_recovery/best/` and writes `golden_baseline_manifest.json`.

## Status API
`GET /api/model/status` includes:

1. `runtime_profile`
2. `deterministic_forced`
3. `per_chat_reset_enabled`
4. `golden_baseline_loaded`
5. `golden_baseline_id`

## Rollback
To disable golden lock behavior, set:

```bash
export FISCAL_CHAT_RUNTIME_PROFILE=default
export FISCAL_CHAT_FORCE_DETERMINISTIC_GLOBAL=0
export FISCAL_CHAT_GOLDEN_DISABLE_GEMINI=0
```
