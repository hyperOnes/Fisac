# Full Plan: Build a Chat Interface for Fiscal

## 1. Objective

Create a production-ready chat interface for Fiscal that supports:

1. Stateful multi-turn conversation
2. Streaming responses
3. Tool-augmented answers (optional, controlled)
4. Reliable local operation (offline-first where feasible)
5. Observability, testing, and reproducible deployment

## 2. Scope

### In scope

1. Chat backend API
2. Chat UI (web desktop/mobile responsive)
3. Conversation/session persistence
4. Model adapter abstraction (Fiscal core + external fallback runtime)
5. Streaming token UX
6. Error handling, retries, and basic rate limiting
7. Security and privacy guardrails
8. CI checks and test coverage for critical paths

### Out of scope (initial release)

1. Voice I/O
2. Enterprise SSO/tenancy
3. Advanced multimodal authoring (images/audio upload)
4. Human-in-the-loop moderation dashboard

## 3. Product requirements

1. User can start a chat, send prompts, and receive streamed responses.
2. Chat history survives app refresh/restart.
3. Latency target:
   - first token under 1.5s on local default path (best effort)
4. Resilience:
   - no app crash on model/tool failure
   - user gets actionable error messages
5. Privacy:
   - default local storage
   - no outbound network calls unless explicitly enabled

## 4. Architecture

## 4.1 Components

1. **Frontend** (React + TypeScript):
   - chat composer
   - message timeline
   - streaming token renderer
   - conversation list/search
2. **Backend API** (FastAPI):
   - conversation CRUD
   - `/chat/respond` streaming endpoint (SSE)
   - model routing and tool orchestration
3. **Inference Adapter Layer**:
   - `FiscalAdapter` (wraps current core)
   - optional `TransformersAdapter` or `LlamaCppAdapter`
4. **Storage**:
   - SQLite for conversations, messages, metadata
   - optional vector index for retrieval memories
5. **Telemetry/Logging**:
   - structured logs
   - request IDs
   - latency + token counters

## 4.2 Data flow

1. User submits prompt in UI.
2. Backend loads conversation context and policy settings.
3. Adapter builds model-ready prompt/state.
4. Backend streams output chunks to UI via SSE.
5. Final message and usage metadata saved transactionally.

## 5. API and schema design

## 5.1 API endpoints

1. `POST /api/chat/respond` (SSE stream)
2. `POST /api/conversations`
3. `GET /api/conversations`
4. `GET /api/conversations/{id}/messages`
5. `PATCH /api/conversations/{id}`
6. `DELETE /api/conversations/{id}`

## 5.2 Core DB tables

1. `conversations`
   - `id`, `title`, `created_at`, `updated_at`, `settings_json`
2. `messages`
   - `id`, `conversation_id`, `role`, `content`, `status`, `created_at`
3. `runs`
   - `id`, `conversation_id`, `model_name`, `latency_ms`, `token_count`, `error`

## 6. Prompting and context policy

1. System prompt template with strict behavior boundaries.
2. Context window strategy:
   - recency-first history truncation
   - optional rolling summary message
3. Deterministic mode toggle for reproducibility.
4. Tool use policy:
   - allowlist by tool
   - explicit call traces in metadata

## 7. Model integration strategy

1. Define a common adapter interface:
   - `generate_stream(messages, config) -> token/events`
2. Implement Fiscal adapter first.
3. Add optional fallback adapter (Transformers or llama.cpp).
4. Add health checks:
   - warmup test
   - adapter capability flags

## 8. UX plan

1. Chat page sections:
   - left: conversation list
   - center: message thread
   - bottom: composer with send/stop/retry
2. Streaming states:
   - typing indicator
   - partial token rendering
   - cancel in-progress generation
3. Error UX:
   - inline retry card
   - preserved user prompt on failure
4. Mobile behavior:
   - sticky composer
   - virtual keyboard safe spacing

## 9. Security and privacy

1. No remote telemetry by default.
2. Secrets from env only; never stored in chat DB.
3. Input/output sanitization for rendered markdown.
4. Optional local encryption for chat DB at rest.

## 10. Observability and SLOs

1. Metrics:
   - `first_token_ms`
   - `total_latency_ms`
   - `tokens_out`
   - `error_rate`
2. Logs:
   - request ID
   - adapter used
   - tool-call count
3. Alerts (later phase):
   - sustained error spikes
   - latency regression

## 11. Testing strategy

## 11.1 Unit tests

1. Prompt assembly and truncation logic
2. Adapter output normalization
3. DB CRUD and transaction integrity

## 11.2 Integration tests

1. End-to-end send prompt -> stream -> persist
2. Retry and cancel behavior
3. Adapter failure fallback behavior

## 11.3 UI tests

1. Composer interactions
2. Streaming rendering and stop action
3. Conversation switching/history load

## 12. Delivery phases and milestones

## Phase A: Foundation (week 1)

1. Project scaffolding (API + UI + DB)
2. Conversation/message schema + CRUD
3. Non-streaming adapter stub

Exit criteria:

1. Can create conversation and store static responses.

## Phase B: Streaming chat MVP (week 2)

1. SSE streaming endpoint
2. UI timeline with streaming tokens
3. Basic retry/cancel

Exit criteria:

1. Stable multi-turn chat with persisted history.

## Phase C: Fiscal adapter + quality hardening (week 3)

1. Integrate real Fiscal model adapter
2. Context policy + summary truncation
3. Error handling and resilience pass

Exit criteria:

1. Chat works with Fiscal backend in normal user flow.

## Phase D: Performance and offline polish (week 4)

1. Latency tuning and cache layers
2. Offline mode validation
3. Packaging and release docs

Exit criteria:

1. Meets baseline latency and reliability targets.

## 13. Deliverables

1. `chat_api/` backend service
2. `chat_ui/` frontend application
3. `adapters/fiscal_adapter.py`
4. `db/migrations/` schema migrations
5. `tests/` unit + integration + UI tests
6. `docs/CHAT_INTERFACE.md` architecture and operations guide

## 14. Risks and mitigations

1. **Risk:** Fiscal adapter not naturally conversational.
   - **Mitigation:** add conversation wrapper prompt + constrained response formatter.
2. **Risk:** Latency spikes on local hardware.
   - **Mitigation:** streaming first token path, context compaction, configurable generation limits.
3. **Risk:** Context drift over long chats.
   - **Mitigation:** rolling summary and strict truncation strategy.

## 15. Definition of done

1. End-to-end chat UX works with persistent history.
2. Streaming and cancellation are reliable.
3. Tests pass in CI for critical paths.
4. Benchmark-level telemetry exists for latency/errors.
5. Docs include setup, run, and troubleshooting.
