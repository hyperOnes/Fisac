# Fisac Chat Interface (Liquid Local v1)

## Purpose

Local-first chat loop for Fisac with:

- FastAPI backend + SQLite persistence
- React/Vite frontend
- SSE token streaming
- Fisac-native liquid runtime generation
- Optional external context probing per conversation (never final text generation)
- Integrated realtime Cursor Lab (`/ws/cursor`)

## Runtime Modes

- Liquid-native mode: default and always available.
- Context-probe mode: enabled per conversation when `gemini_enabled=true` and `FISCAL_EXTERNAL_CONTEXT_ENABLED=1`.
- Fisac runs `FISCAL_CONTEXT_QUERY_COUNT` focused context probes (default `5`) per turn, rotating providers.
- Providers: Gemini (`FISCAL_GEMINI_API_KEY` or `FISCAL_GEMINI_API_KEYS`) and optional OpenAI (`FISCAL_OPENAI_API_KEY` or `FISCAL_OPENAI_API_KEYS`) as failover/rotation.
- Context probe failures never break chat completion; run metadata records `error_code=external_context_error` only when all probes fail.
- Default Gemini model: `gemini-3.1-pro-preview`.

## Core Endpoints

- `POST /api/conversations` (defaults `gemini_enabled=true`)
- `GET /api/conversations`
- `PATCH /api/conversations/{conversation_id}` (`title?`, `gemini_enabled?`)
- `GET /api/conversations/{conversation_id}/messages`
- `DELETE /api/conversations/{conversation_id}`
- `POST /api/chat/respond` (SSE `ack`, `token`, `done`, `error`)
- `GET /api/model/status` (`gemini_configured`, `gemini_available`, `openai_configured`, `external_context_enabled`, `context_query_count`, `generation_backend`, `pure_liquid_active`, `gemini_context_only`, `mode`)
- `GET /api/tools`
- `POST /api/tools/call`
- `GET /api/conversations/{conversation_id}/tool-calls`
- `WS /ws/cursor`

## Chat Flow

1. User message is persisted.
2. Fisac bridge encodes text + context summary and executes `online_step`.
3. Liquid decoder builds concise text from liquid vector state + conversation-local memory + recent context.
4. If the active conversation has `gemini_enabled=true`, external context probes are run and merged into liquid decoding.
5. Final response streams over SSE and is persisted with run metrics.

## Per-Conversation Gemini Toggle

- Stored in DB column `conversations.gemini_enabled` (`INTEGER NOT NULL DEFAULT 1`).
- Shown in chat header as `Gemini ON/OFF`.
- Toggle is optimistic in UI and rolls back on API error.
- Toggle state follows the active conversation.

## Cursor Lab

- Open the `Cursor Lab` tab in the UI.
- Browser sends frames (`x`, `y`, `t_ms`, viewport size) to `/ws/cursor`.
- Backend keeps a rolling `30`-frame feature buffer `[x, y, vx, vy]`.
- Inference cadence: every `3` frames.
- Output includes:
  - point prediction (`gru`, `liquid`)
  - trajectories (`gru_path`, `liquid_path`, 5 points)
  - EMA metrics (`gru_mse_ema`, `liquid_mse_ema`, `rtt_ms`)
  - `buffer_ready`

## Offline/Local Policy

- Chat core operates fully local with no cloud dependency.
- External context providers are optional and key-gated.
- Cursor inference is local PyTorch.

## Run

```bash
./scripts/dev_chat.sh
```

`scripts/dev_chat.sh` auto-loads `chat_api/.env.local` when present.

- UI: `http://127.0.0.1:5173`
- API: `http://127.0.0.1:8000`

## Test

```bash
./scripts/run_chat_tests.sh
```

Also available:

```bash
pytest -q
npm --prefix chat_ui test -- --run
```
