from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import WebSocket, WebSocketDisconnect

from chat_api.config import Settings, get_settings
from chat_api.db import init_db
from chat_api.repository import ChatRepository
from chat_api.schemas import (
    ChatRespondRequest,
    ConversationCreateRequest,
    ConversationDTO,
    ConversationListResponse,
    ConversationPatchRequest,
    DeleteResponse,
    MessageDTO,
    MessageListResponse,
    ModelStatusResponse,
)
from chat_api.schemas_v2 import (
    ToolCallDTO,
    ToolCallListResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolListResponse,
    ToolSpecDTO,
)
from chat_api.services.chat_service import ChatService
from chat_api.services.conversational_composer import ConversationalComposer
from chat_api.services.context_window import ContextWindowPolicy
from chat_api.services.fiscal_text_bridge import FiscalTextBridge
from chat_api.services.gemini_client import GeminiClient
from chat_api.services.provider_pool import ProviderPool
from chat_api.services.openai_client import OpenAIClient
from chat_api.services.lifecycle_worker import LifecycleWorker
from chat_api.services.sse import format_sse
from chat_api.services.summary_service import SummaryService
from chat_api.services.tool_service import ToolService
from chat_api.services.truth_reasoner import TruthReasoner
from organic_cursor.app import CursorRuntime


settings: Settings = get_settings()
repo = ChatRepository(settings)
bridge = FiscalTextBridge(repo=repo, settings=settings)
context_policy = ContextWindowPolicy(
    keep_last_messages=settings.context_window_messages,
    summary_every_user_turns=settings.summary_every_user_turns,
)
summary_service = SummaryService(max_sentences=settings.summary_max_sentences)
tool_service = ToolService(repo=repo)
provider_pool = ProviderPool()
gemini_client = GeminiClient(settings=settings, provider_pool=provider_pool)
openai_client = OpenAIClient(settings=settings, provider_pool=provider_pool)
composer = ConversationalComposer()
truth_reasoner = TruthReasoner()
chat_service = ChatService(
    repo=repo,
    bridge=bridge,
    context_policy=context_policy,
    summary_service=summary_service,
    tool_service=tool_service,
    composer=composer,
    gemini_client=gemini_client,
    openai_client=openai_client,
    truth_reasoner=truth_reasoner,
)
lifecycle_worker = LifecycleWorker(chat_service=chat_service)
cursor_runtime = CursorRuntime(settings=settings)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db(settings)
    bridge.load_or_init()
    cursor_runtime.load_models()
    lifecycle_worker.start()
    try:
        yield
    finally:
        await lifecycle_worker.stop()
        chat_service.shutdown()


app = FastAPI(title="Fisac Chat API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_chat_service() -> ChatService:
    return chat_service


@app.get("/")
def root() -> dict[str, str | bool]:
    return {
        "ok": True,
        "service": "Fisac Chat API",
        "ui_url": "http://127.0.0.1:5173",
        "health_url": "http://127.0.0.1:8000/api/health",
        "docs_url": "http://127.0.0.1:8000/docs",
    }


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/api/conversations", response_model=ConversationDTO)
def create_conversation(
    payload: ConversationCreateRequest,
    svc: ChatService = Depends(get_chat_service),
) -> ConversationDTO:
    deterministic_forced = bool(
        settings.force_deterministic_global
        or settings.runtime_profile == "golden_lock"
        or settings.golden_disable_gemini
    )
    conv = svc.create_conversation(
        payload.title,
        gemini_enabled=(False if deterministic_forced else settings.gemini_default_enabled),
    )
    return ConversationDTO(**conv.__dict__)


@app.get("/api/conversations", response_model=ConversationListResponse)
def list_conversations(
    limit: int = Query(default=100, ge=1, le=500),
    svc: ChatService = Depends(get_chat_service),
) -> ConversationListResponse:
    items = [ConversationDTO(**c.__dict__) for c in svc.list_conversations(limit=limit)]
    return ConversationListResponse(items=items)


@app.get("/api/conversations/{conversation_id}/messages", response_model=MessageListResponse)
def list_messages(
    conversation_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    before: str | None = Query(default=None),
    svc: ChatService = Depends(get_chat_service),
) -> MessageListResponse:
    conv = repo.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    items = [MessageDTO(**m.__dict__) for m in svc.list_messages(conversation_id=conversation_id, limit=limit, before=before)]
    return MessageListResponse(items=items)


@app.patch("/api/conversations/{conversation_id}", response_model=ConversationDTO)
def patch_conversation(
    conversation_id: str,
    payload: ConversationPatchRequest,
    svc: ChatService = Depends(get_chat_service),
) -> ConversationDTO:
    if payload.title is None and payload.gemini_enabled is None:
        raise HTTPException(status_code=400, detail="At least one field (title or gemini_enabled) is required.")
    updated = svc.update_conversation(
        conversation_id=conversation_id,
        title=payload.title,
        gemini_enabled=payload.gemini_enabled,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDTO(**updated.__dict__)


@app.delete("/api/conversations/{conversation_id}", response_model=DeleteResponse)
def delete_conversation(
    conversation_id: str,
    svc: ChatService = Depends(get_chat_service),
) -> DeleteResponse:
    deleted = svc.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return DeleteResponse(ok=True)


@app.post("/api/chat/respond")
async def chat_respond(
    payload: ChatRespondRequest,
    svc: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    if not payload.stream:
        raise HTTPException(status_code=400, detail="Only stream=true is supported in v1.")

    run_id = payload.client_request_id or str(uuid4())

    async def _event_iter():
        async for evt in svc.stream_reply(
            conversation_id=payload.conversation_id,
            user_text=payload.message,
            run_id=run_id,
        ):
            yield format_sse(evt)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(_event_iter(), media_type="text/event-stream", headers=headers)


@app.get("/api/model/status", response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    status = bridge.model_status()
    status["gemini_configured"] = gemini_client.configured
    status["gemini_available"] = gemini_client.available
    status["gemini_key_count"] = gemini_client.key_count
    status["gemini_available_key_count"] = gemini_client.available_key_count
    status["openai_configured"] = openai_client.configured
    status["openai_key_count"] = openai_client.key_count
    status["openai_available_key_count"] = openai_client.available_key_count
    status["external_context_enabled"] = settings.external_context_enabled
    status["context_query_count"] = settings.context_query_count
    status["mode"] = status.get("generation_backend", settings.generation_backend)
    status.update(chat_service.get_runtime_status())
    return ModelStatusResponse(**status)


@app.get("/api/runs/{run_id}/decision-trace")
def run_decision_trace(run_id: str) -> dict[str, object]:
    trace = chat_service.get_decision_trace(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Decision trace not found")
    return trace


@app.get("/api/tools", response_model=ToolListResponse)
def list_tools() -> ToolListResponse:
    items = [ToolSpecDTO(name=t.name, description=t.description, input_hint=t.input_hint) for t in tool_service.list_tools()]
    return ToolListResponse(items=items)


@app.post("/api/tools/call", response_model=ToolCallResponse)
def call_tool(payload: ToolCallRequest) -> ToolCallResponse:
    result = tool_service.call_tool(payload.tool, payload.args)
    return ToolCallResponse(ok=result.ok, tool=result.tool, output=result.output, error=result.error)


@app.get("/api/conversations/{conversation_id}/tool-calls", response_model=ToolCallListResponse)
def list_tool_calls(
    conversation_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
) -> ToolCallListResponse:
    conv = repo.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    rows = repo.list_tool_calls(conversation_id=conversation_id, limit=limit)
    return ToolCallListResponse(items=[ToolCallDTO(**r.__dict__) for r in rows])


@app.websocket("/ws/cursor")
async def ws_cursor(ws: WebSocket) -> None:
    await ws.accept()
    try:
        await cursor_runtime.handle_connection(ws)
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("Unhandled cursor websocket error")
        return
