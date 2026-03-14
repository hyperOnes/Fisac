# Plan: Compare Fisac vs Mistral 7B on HellaSwag 10-shot (lm-eval-harness)

## Goal
Create a reproducible, private-capable benchmark flow that runs:

1. Mistral 7B on `hellaswag` with `num_fewshot=10`
2. Fisac on the same task/settings through an lm-eval model adapter
3. A paired comparison report with raw rows + summary statistics

## Important constraint
`lm-eval-harness` requires a model runtime backend. There is no "no-runtime" mode.

- Mistral 7B runtime options: `transformers` or `llama.cpp`.
- Fisac runtime option: PyTorch via a custom lm-eval model wrapper.

## Phase 1: Evaluation protocol lock

1. Pin exact package versions and evaluation settings.
2. Fix task settings: `hellaswag`, `num_fewshot=10`, deterministic seeds.
3. Define reporting metrics:
   - `acc`
   - `acc_norm`
   - run metadata (runtime/backend/model path/precision)

Deliverable:

- `lmeval/PROTOCOL.md`

## Phase 2: Offline-first environment and caching

1. Build setup script to pre-download:
   - model weights (Mistral 7B)
   - HellaSwag dataset
2. Support fully offline execution after setup:
   - `HF_HUB_OFFLINE=1`
   - `HF_DATASETS_OFFLINE=1`
   - `TRANSFORMERS_OFFLINE=1`
3. Add cache validation command before run.

Deliverables:

- `lmeval/prepare_offline.py`
- `lmeval/env_offline.sh`

## Phase 3: Mistral 7B runner

1. Implement baseline runner supporting:
   - HF backend (preferred comparability)
   - llama.cpp GGUF backend (practical local path)
2. Store raw lm-eval JSON output and normalized summary row.

Deliverable:

- `lmeval/run_mistral_hellaswag.py`

## Phase 4: Fisac lm-eval adapter

1. Implement custom lm-eval model class for Fisac:
   - required log-likelihood API path for multiple choice scoring
2. Freeze learning during eval (deterministic scoring mode).
3. Add adapter unit tests for shape, tokenization mapping, and determinism.

Deliverables:

- `lmeval/fisac_lm_adapter.py`
- `lmeval/run_fisac_hellaswag.py`
- `tests/test_fisac_lmeval_adapter.py`

## Phase 5: Paired comparison reporting

1. Collect per-run raw rows with schema:
   - seed, model, backend, task, fewshot, acc, acc_norm, runtime stats
2. Build paired deltas:
   - `delta_acc = acc_mistral - acc_fisac` (or inverse, consistently documented)
3. Add win-rate with Clopper-Pearson confidence intervals.

Deliverables:

- `lmeval/compare_hellaswag.py`
- `lmeval/results/paired_rows.json`
- `lmeval/results/paired_rows.csv`
- `lmeval/results/summary.md`

## Phase 6: One-command orchestration

1. Single entrypoint to run complete flow:
   - environment checks
   - cache validation
   - Mistral run
   - Fisac run
   - comparison report generation
2. Timestamped artifact directory for reproducibility.

Deliverable:

- `lmeval/run_all.sh`

## Validation and acceptance criteria

1. Same `task` and `num_fewshot` for both models.
2. Same seed set and prompt format where applicable.
3. All outputs written to timestamped directory.
4. Raw rows and summary are both generated.
5. Offline run works after cache prep.

## Risks and mitigations

1. **Risk:** Fisac is not a pretrained text LM.
   - **Mitigation:** mark benchmark as mechanical comparability, not capability parity.
2. **Risk:** Backend mismatch vs published Mistral numbers.
   - **Mitigation:** log backend/precision/template and treat local results as local baseline.
3. **Risk:** Hardware memory limits.
   - **Mitigation:** allow GGUF backend and reduced context/batch.
