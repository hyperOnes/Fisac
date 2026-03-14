from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class SSEEvent:
    event: str
    data: dict[str, Any]


def format_sse(event: SSEEvent) -> bytes:
    payload = json.dumps(event.data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event.event}\ndata: {payload}\n\n".encode("utf-8")
