from __future__ import annotations

from pydantic import BaseModel, Field


class ToolSpecDTO(BaseModel):
    name: str
    description: str
    input_hint: str


class ToolListResponse(BaseModel):
    items: list[ToolSpecDTO]


class ToolCallRequest(BaseModel):
    tool: str = Field(min_length=1)
    args: dict = Field(default_factory=dict)


class ToolCallResponse(BaseModel):
    ok: bool
    tool: str
    output: dict
    error: str | None = None


class ToolCallDTO(BaseModel):
    id: str
    run_id: str | None
    conversation_id: str
    tool_name: str
    tool_args: str
    ok: int
    output_json: str
    error: str | None
    created_at: str


class ToolCallListResponse(BaseModel):
    items: list[ToolCallDTO]
