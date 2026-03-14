from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Callable, Optional

from chat_api.repository import ChatRepository


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_hint: str


@dataclass
class ToolResult:
    ok: bool
    tool: str
    output: dict[str, Any]
    error: Optional[str] = None


class _SafeMathEvaluator(ast.NodeVisitor):
    _ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
    _ALLOWED_UNARY = (ast.UAdd, ast.USub)

    def visit(self, node: ast.AST) -> Any:  # noqa: ANN401
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants are allowed")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self._ALLOWED_UNARY):
            val = self.visit(node.operand)
            return +val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp) and isinstance(node.op, self._ALLOWED_BINOPS):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
        raise ValueError("Unsupported expression")


class ToolService:
    def __init__(self, repo: ChatRepository) -> None:
        self.repo = repo
        self._tools: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "time.now": self._time_now,
            "math.eval": self._math_eval,
            "context.search": self._context_search,
        }

    def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(name="time.now", description="Return current local UTC timestamp", input_hint="{}"),
            ToolSpec(name="math.eval", description="Evaluate safe arithmetic expression", input_hint='{"expression":"(2+3)*4"}'),
            ToolSpec(
                name="context.search",
                description="Search recent conversation messages",
                input_hint='{"conversation_id":"<id>","query":"keyword"}',
            ),
        ]

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        fn = self._tools.get(name)
        if fn is None:
            return ToolResult(ok=False, tool=name, output={}, error=f"Unknown tool: {name}")
        try:
            return fn(args)
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(ok=False, tool=name, output={}, error=str(exc))

    def parse_tool_command(self, text: str) -> tuple[str, dict[str, Any]] | None:
        stripped = text.strip()
        if not stripped.startswith("/tool "):
            return None
        body = stripped[len("/tool ") :].strip()
        if not body:
            return None
        parts = body.split(maxsplit=1)
        tool_name = parts[0].strip()
        if len(parts) == 1:
            return tool_name, {}
        arg_text = parts[1].strip()
        if not arg_text:
            return tool_name, {}
        if arg_text.startswith("{"):
            payload = json.loads(arg_text)
            if not isinstance(payload, dict):
                raise ValueError("Tool args JSON must be an object")
            return tool_name, payload
        # Shortcut: free-text maps to query/expression depending on tool.
        if tool_name == "math.eval":
            return tool_name, {"expression": arg_text}
        if tool_name == "context.search":
            return tool_name, {"query": arg_text}
        return tool_name, {"text": arg_text}

    def _time_now(self, args: dict[str, Any]) -> ToolResult:
        del args
        return ToolResult(
            ok=True,
            tool="time.now",
            output={"utc": datetime.now(timezone.utc).isoformat()},
        )

    def _math_eval(self, args: dict[str, Any]) -> ToolResult:
        expression = str(args.get("expression", "")).strip()
        if not expression:
            return ToolResult(ok=False, tool="math.eval", output={}, error="Missing 'expression'")
        if len(expression) > 200:
            return ToolResult(ok=False, tool="math.eval", output={}, error="Expression too long")
        tree = ast.parse(expression, mode="eval")
        value = _SafeMathEvaluator().visit(tree)
        return ToolResult(ok=True, tool="math.eval", output={"expression": expression, "result": float(value)})

    def _context_search(self, args: dict[str, Any]) -> ToolResult:
        conversation_id = str(args.get("conversation_id", "")).strip()
        query = str(args.get("query", "")).strip().lower()
        if not conversation_id:
            return ToolResult(ok=False, tool="context.search", output={}, error="Missing 'conversation_id'")
        if not query:
            return ToolResult(ok=False, tool="context.search", output={}, error="Missing 'query'")

        messages = self.repo.list_messages(conversation_id=conversation_id, limit=200, before=None)
        hits = []
        for m in messages:
            content_l = m.content.lower()
            if query in content_l:
                hits.append(
                    {
                        "message_id": m.id,
                        "role": m.role,
                        "created_at": m.created_at,
                        "snippet": m.content[:220],
                    }
                )
        return ToolResult(ok=True, tool="context.search", output={"query": query, "hits": hits[:10], "count": len(hits)})
