from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Optional


@dataclass
class ConversationRecord:
    id: str
    title: str
    created_at: str
    updated_at: str
    gemini_enabled: bool = True
    last_message_preview: Optional[str] = None


@dataclass
class MessageRecord:
    id: str
    conversation_id: str
    role: str
    content: str
    status: str
    created_at: str
    run_id: Optional[str] = None
    latency_ms: Optional[float] = None
    confidence: Optional[float] = None
    mse: Optional[float] = None
    generation_source: Optional[str] = None
    generation_attempts: Optional[int] = None
    quality_flags: Optional[str] = None


@dataclass
class RunRecord:
    id: str
    conversation_id: str
    started_at: str
    ended_at: str
    latency_ms: float
    mse: float
    confidence: float
    pruned_now: int
    myelinated_now: int
    generation_source: Literal["gemini", "gemini_regen", "deterministic", "liquid_native", "gemini_raw"] = "deterministic"
    generation_attempts: int = 1
    quality_flags: Optional[str] = None
    output_chars: int = 0
    runtime_profile: str = "default"
    baseline_id: Optional[str] = None
    context_probes_total: int = 0
    context_probes_success: int = 0
    candidate_count: int = 0
    winner_index: Optional[int] = None
    winner_score: Optional[float] = None
    answer_mode: Optional[str] = None
    echo_score: Optional[float] = None
    coverage_score: Optional[float] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ResponseMemoryRecord:
    id: str
    conversation_id: str
    user_text: str
    assistant_text: str
    user_vec: bytes
    assistant_vec: bytes
    confidence: float
    created_at: str


@dataclass
class ToolCallRecord:
    id: str
    run_id: Optional[str]
    conversation_id: str
    tool_name: str
    tool_args: str
    ok: int
    output_json: str
    error: Optional[str]
    created_at: str
