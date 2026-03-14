from __future__ import annotations

import json
from typing import Any


def render_tool_result(tool: str, payload: dict[str, Any], ok: bool, error: str | None = None) -> str:
    if not ok:
        return f"Tool `{tool}` failed: {error or 'unknown error'}."
    pretty = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    return f"Tool `{tool}` result:\n{pretty}"
