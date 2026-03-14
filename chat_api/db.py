from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterator

from .config import Settings


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        gemini_enabled INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        run_id TEXT NULL,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        mse REAL NOT NULL,
        confidence REAL NOT NULL,
        pruned_now INTEGER NOT NULL,
        myelinated_now INTEGER NOT NULL,
        generation_source TEXT NOT NULL DEFAULT 'deterministic',
        generation_attempts INTEGER NOT NULL DEFAULT 1,
        quality_flags TEXT NULL,
        output_chars INTEGER NOT NULL DEFAULT 0,
        runtime_profile TEXT NOT NULL DEFAULT 'default',
        baseline_id TEXT NULL,
        context_probes_total INTEGER NOT NULL DEFAULT 0,
        context_probes_success INTEGER NOT NULL DEFAULT 0,
        candidate_count INTEGER NOT NULL DEFAULT 0,
        winner_index INTEGER NULL,
        winner_score REAL NULL,
        answer_mode TEXT NULL,
        echo_score REAL NULL,
        coverage_score REAL NULL,
        error_code TEXT NULL,
        error_message TEXT NULL,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS state_snapshots (
        conversation_id TEXT PRIMARY KEY,
        step INTEGER NOT NULL,
        state_blob BLOB NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS response_memory (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        user_text TEXT NOT NULL,
        assistant_text TEXT NOT NULL,
        user_vec BLOB NOT NULL,
        assistant_vec BLOB NOT NULL,
        confidence REAL NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_calls (
        id TEXT PRIMARY KEY,
        run_id TEXT NULL,
        conversation_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        tool_args TEXT NOT NULL,
        ok INTEGER NOT NULL,
        output_json TEXT NOT NULL,
        error TEXT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_conv_created ON messages(conversation_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_runs_conv_started ON runs(conversation_id, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_response_memory_conv_created ON response_memory(conversation_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_tool_calls_conv_created ON tool_calls(conversation_id, created_at)",
]


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r["name"]) for r in rows}


def _apply_migrations(conn: sqlite3.Connection) -> None:
    conv_columns = _table_columns(conn, "conversations")
    if "gemini_enabled" not in conv_columns:
        conn.execute("ALTER TABLE conversations ADD COLUMN gemini_enabled INTEGER NOT NULL DEFAULT 1")
        conn.execute("UPDATE conversations SET gemini_enabled = 1 WHERE gemini_enabled IS NULL")

    runs_columns = _table_columns(conn, "runs")
    if "generation_source" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN generation_source TEXT NOT NULL DEFAULT 'deterministic'")
        conn.execute("UPDATE runs SET generation_source = 'deterministic' WHERE generation_source IS NULL")
    if "generation_attempts" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN generation_attempts INTEGER NOT NULL DEFAULT 1")
        conn.execute("UPDATE runs SET generation_attempts = 1 WHERE generation_attempts IS NULL")
    if "quality_flags" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN quality_flags TEXT NULL")
    if "output_chars" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN output_chars INTEGER NOT NULL DEFAULT 0")
        conn.execute("UPDATE runs SET output_chars = 0 WHERE output_chars IS NULL")
    if "runtime_profile" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN runtime_profile TEXT NOT NULL DEFAULT 'default'")
        conn.execute("UPDATE runs SET runtime_profile = 'default' WHERE runtime_profile IS NULL")
    if "baseline_id" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN baseline_id TEXT NULL")
    if "context_probes_total" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN context_probes_total INTEGER NOT NULL DEFAULT 0")
        conn.execute("UPDATE runs SET context_probes_total = 0 WHERE context_probes_total IS NULL")
    if "context_probes_success" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN context_probes_success INTEGER NOT NULL DEFAULT 0")
        conn.execute("UPDATE runs SET context_probes_success = 0 WHERE context_probes_success IS NULL")
    if "candidate_count" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN candidate_count INTEGER NOT NULL DEFAULT 0")
        conn.execute("UPDATE runs SET candidate_count = 0 WHERE candidate_count IS NULL")
    if "winner_index" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN winner_index INTEGER NULL")
    if "winner_score" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN winner_score REAL NULL")
    if "answer_mode" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN answer_mode TEXT NULL")
    if "echo_score" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN echo_score REAL NULL")
    if "coverage_score" not in runs_columns:
        conn.execute("ALTER TABLE runs ADD COLUMN coverage_score REAL NULL")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(settings: Settings) -> None:
    with connect(settings.db_path) as conn:
        for stmt in SCHEMA_STATEMENTS:
            conn.execute(stmt)
        _apply_migrations(conn)
        conn.commit()
