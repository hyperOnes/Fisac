from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = Field(default=None)


class ConversationPatchRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=120)
    gemini_enabled: Optional[bool] = None


class ConversationDTO(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    gemini_enabled: bool
    last_message_preview: Optional[str] = None


class ConversationListResponse(BaseModel):
    items: list[ConversationDTO]


class MessageDTO(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    status: str
    run_id: Optional[str] = None
    latency_ms: Optional[float] = None
    confidence: Optional[float] = None
    mse: Optional[float] = None
    generation_source: Optional[str] = None
    generation_attempts: Optional[int] = None
    quality_flags: Optional[str] = None


class MessageListResponse(BaseModel):
    items: list[MessageDTO]


class DeleteResponse(BaseModel):
    ok: bool


class ChatRespondRequest(BaseModel):
    conversation_id: str
    message: str = Field(min_length=1)
    stream: bool = True
    client_request_id: Optional[str] = None


class ModelStatusResponse(BaseModel):
    ready: bool
    device: str
    feature_dim: int
    num_experts: int
    gemini_configured: bool
    gemini_available: bool
    gemini_key_count: int
    gemini_available_key_count: int
    openai_configured: bool
    openai_key_count: int
    openai_available_key_count: int
    external_context_enabled: bool
    context_query_count: int
    generation_backend: str
    pure_liquid_active: bool
    gemini_context_only: bool
    mode: str
    jury_mode: bool = False
    context_probe_success_rate_1h: float = 0.0
    provider_health: dict[str, object] = Field(default_factory=dict)
    jury_head_version: Optional[str] = None
    synthesis_head_version: Optional[str] = None
    learned_heads_enabled: bool = False
