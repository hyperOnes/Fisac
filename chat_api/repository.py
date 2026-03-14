from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from typing import Iterable, Optional
from uuid import uuid4

from .config import Settings
from .db import connect
from .models import ConversationRecord, MessageRecord, ResponseMemoryRecord, RunRecord, ToolCallRecord


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _conn(self):
        return connect(self.settings.db_path)

    def create_conversation(self, title: Optional[str] = None, gemini_enabled: bool = True) -> ConversationRecord:
        now = _utc_now()
        conversation_id = str(uuid4())
        conv_title = title.strip() if title and title.strip() else "New Chat"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at, gemini_enabled) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, conv_title, now, now, 1 if gemini_enabled else 0),
            )
            conn.commit()
        return ConversationRecord(
            id=conversation_id,
            title=conv_title,
            created_at=now,
            updated_at=now,
            gemini_enabled=bool(gemini_enabled),
        )

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at, gemini_enabled FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return ConversationRecord(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            gemini_enabled=bool(row["gemini_enabled"]),
        )

    def list_conversations(self, limit: int = 100) -> list[ConversationRecord]:
        safe_limit = max(1, min(limit, 500))
        query = """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                c.gemini_enabled,
                (
                    SELECT substr(m.content, 1, 140)
                    FROM messages m
                    WHERE m.conversation_id = c.id
                    ORDER BY m.created_at DESC
                    LIMIT 1
                ) AS last_message_preview
            FROM conversations c
            ORDER BY c.updated_at DESC
            LIMIT ?
        """
        with self._conn() as conn:
            rows = conn.execute(query, (safe_limit,)).fetchall()
        return [
            ConversationRecord(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                gemini_enabled=bool(row["gemini_enabled"]),
                last_message_preview=row["last_message_preview"],
            )
            for row in rows
        ]

    def update_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        gemini_enabled: Optional[bool] = None,
    ) -> Optional[ConversationRecord]:
        if title is None and gemini_enabled is None:
            return self.get_conversation(conversation_id)
        now = _utc_now()
        assignments: list[str] = ["updated_at = ?"]
        params: list[object] = [now]
        if title is not None:
            assignments.insert(0, "title = ?")
            params.insert(0, title)
        if gemini_enabled is not None:
            assignments.insert(0, "gemini_enabled = ?")
            params.insert(0, 1 if gemini_enabled else 0)
        params.append(conversation_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE conversations SET {', '.join(assignments)} WHERE id = ?",
                tuple(params),
            )
            conn.commit()
        return self.get_conversation(conversation_id)

    def touch_conversation(self, conversation_id: str) -> None:
        now = _utc_now()
        with self._conn() as conn:
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            conn.commit()

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            return cur.rowcount > 0

    def create_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        status: str,
        run_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> MessageRecord:
        now = _utc_now()
        mid = message_id or str(uuid4())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO messages(id, conversation_id, role, content, status, created_at, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (mid, conversation_id, role, content, status, now, run_id),
            )
            conn.commit()
        return MessageRecord(
            id=mid,
            conversation_id=conversation_id,
            role=role,
            content=content,
            status=status,
            created_at=now,
            run_id=run_id,
        )

    def update_message(self, message_id: str, content: str, status: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE messages SET content = ?, status = ? WHERE id = ?",
                (content, status, message_id),
            )
            conn.commit()

    def list_messages(self, conversation_id: str, limit: int = 200, before: Optional[str] = None) -> list[MessageRecord]:
        safe_limit = max(1, min(limit, 1000))
        params: list[object] = [conversation_id]
        where = "m.conversation_id = ?"
        if before:
            where += " AND m.created_at < ?"
            params.append(before)
        params.append(safe_limit)
        query = f"""
            SELECT
                m.id,
                m.conversation_id,
                m.role,
                m.content,
                m.status,
                m.created_at,
                m.run_id,
                r.latency_ms,
                r.confidence,
                r.mse,
                r.generation_source,
                r.generation_attempts,
                r.quality_flags
            FROM messages m
            LEFT JOIN runs r ON r.id = m.run_id
            WHERE {where}
            ORDER BY m.created_at DESC
            LIMIT ?
        """
        with self._conn() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        out = [
            MessageRecord(
                id=row["id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                content=row["content"],
                status=row["status"],
                created_at=row["created_at"],
                run_id=row["run_id"],
                latency_ms=row["latency_ms"],
                confidence=row["confidence"],
                mse=row["mse"],
                generation_source=row["generation_source"],
                generation_attempts=row["generation_attempts"],
                quality_flags=row["quality_flags"],
            )
            for row in rows
        ]
        out.reverse()
        return out

    def count_user_messages(self, conversation_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE conversation_id = ? AND role = 'user'",
                (conversation_id,),
            ).fetchone()
        return int(row["c"]) if row else 0

    def list_recent_messages_for_summary(self, conversation_id: str, limit: int = 80) -> list[MessageRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, status, created_at, run_id
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, max(1, min(limit, 500))),
            ).fetchall()
        out = [
            MessageRecord(
                id=row["id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                content=row["content"],
                status=row["status"],
                created_at=row["created_at"],
                run_id=row["run_id"],
            )
            for row in rows
        ]
        out.reverse()
        return out

    def create_run(self, run: RunRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO runs(
                    id, conversation_id, started_at, ended_at, latency_ms,
                    mse, confidence, pruned_now, myelinated_now,
                    generation_source, generation_attempts, quality_flags, output_chars,
                    runtime_profile, baseline_id,
                    context_probes_total, context_probes_success,
                    candidate_count, winner_index, winner_score,
                    answer_mode, echo_score, coverage_score,
                    error_code, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.conversation_id,
                    run.started_at,
                    run.ended_at,
                    run.latency_ms,
                    run.mse,
                    run.confidence,
                    run.pruned_now,
                    run.myelinated_now,
                    run.generation_source,
                    run.generation_attempts,
                    run.quality_flags,
                    run.output_chars,
                    run.runtime_profile,
                    run.baseline_id,
                    run.context_probes_total,
                    run.context_probes_success,
                    run.candidate_count,
                    run.winner_index,
                    run.winner_score,
                    run.answer_mode,
                    run.echo_score,
                    run.coverage_score,
                    run.error_code,
                    run.error_message,
                ),
            )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[sqlite3.Row]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return row

    def list_recent_runs(self, limit: int = 50) -> list[sqlite3.Row]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                (max(1, min(limit, 1000)),),
            ).fetchall()
        return list(rows)

    def upsert_state_snapshot(self, conversation_id: str, step: int, state_blob: bytes) -> None:
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO state_snapshots(conversation_id, step, state_blob, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id)
                DO UPDATE SET step = excluded.step, state_blob = excluded.state_blob, updated_at = excluded.updated_at
                """,
                (conversation_id, int(step), state_blob, now),
            )
            conn.commit()

    def get_state_snapshot(self, conversation_id: str) -> Optional[tuple[int, bytes, str]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT step, state_blob, updated_at FROM state_snapshots WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return int(row["step"]), bytes(row["state_blob"]), str(row["updated_at"])

    def list_snapshot_conversation_ids(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT conversation_id FROM state_snapshots").fetchall()
        return [str(r["conversation_id"]) for r in rows]

    def add_response_memory(
        self,
        conversation_id: str,
        user_text: str,
        assistant_text: str,
        user_vec: bytes,
        assistant_vec: bytes,
        confidence: float,
    ) -> ResponseMemoryRecord:
        rid = str(uuid4())
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO response_memory(
                    id, conversation_id, user_text, assistant_text,
                    user_vec, assistant_vec, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    conversation_id,
                    user_text,
                    assistant_text,
                    user_vec,
                    assistant_vec,
                    float(confidence),
                    now,
                ),
            )
            conn.commit()
        return ResponseMemoryRecord(
            id=rid,
            conversation_id=conversation_id,
            user_text=user_text,
            assistant_text=assistant_text,
            user_vec=user_vec,
            assistant_vec=assistant_vec,
            confidence=float(confidence),
            created_at=now,
        )

    def list_response_memory(self, conversation_id: str, limit: int = 256) -> list[ResponseMemoryRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, user_text, assistant_text, user_vec, assistant_vec, confidence, created_at
                FROM response_memory
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, max(1, min(limit, 5000))),
            ).fetchall()
        return [
            ResponseMemoryRecord(
                id=row["id"],
                conversation_id=row["conversation_id"],
                user_text=row["user_text"],
                assistant_text=row["assistant_text"],
                user_vec=bytes(row["user_vec"]),
                assistant_vec=bytes(row["assistant_vec"]),
                confidence=float(row["confidence"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def list_response_memory_global(self, limit: int = 256) -> list[ResponseMemoryRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, user_text, assistant_text, user_vec, assistant_vec, confidence, created_at
                FROM response_memory
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, min(limit, 10000)),),
            ).fetchall()
        return [
            ResponseMemoryRecord(
                id=row["id"],
                conversation_id=row["conversation_id"],
                user_text=row["user_text"],
                assistant_text=row["assistant_text"],
                user_vec=bytes(row["user_vec"]),
                assistant_vec=bytes(row["assistant_vec"]),
                confidence=float(row["confidence"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def create_tool_call(
        self,
        conversation_id: str,
        tool_name: str,
        tool_args: dict,
        ok: bool,
        output: dict,
        run_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> ToolCallRecord:
        cid = str(uuid4())
        now = _utc_now()
        tool_args_json = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
        output_json = json.dumps(output, ensure_ascii=False, sort_keys=True)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO tool_calls(
                    id, run_id, conversation_id, tool_name, tool_args,
                    ok, output_json, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cid,
                    run_id,
                    conversation_id,
                    tool_name,
                    tool_args_json,
                    1 if ok else 0,
                    output_json,
                    error,
                    now,
                ),
            )
            conn.commit()
        return ToolCallRecord(
            id=cid,
            run_id=run_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            tool_args=tool_args_json,
            ok=1 if ok else 0,
            output_json=output_json,
            error=error,
            created_at=now,
        )

    def list_tool_calls(self, conversation_id: str, limit: int = 100) -> list[ToolCallRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, run_id, conversation_id, tool_name, tool_args, ok, output_json, error, created_at
                FROM tool_calls
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, max(1, min(limit, 1000))),
            ).fetchall()
        out = [
            ToolCallRecord(
                id=row["id"],
                run_id=row["run_id"],
                conversation_id=row["conversation_id"],
                tool_name=row["tool_name"],
                tool_args=row["tool_args"],
                ok=int(row["ok"]),
                output_json=row["output_json"],
                error=row["error"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
        out.reverse()
        return out
