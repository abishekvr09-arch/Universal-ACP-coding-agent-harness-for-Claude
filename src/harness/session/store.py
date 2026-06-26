"""SQLite session store.

Persists the system prompt (restored byte-for-byte each turn — the cached prefix
lives here, not in process memory) and the message log (appended BEFORE tool
execution for crash-resilience).

WAL mode from day one: the CLI, the ACP server, and a future web UI may all touch
one DB. WAL allows concurrent readers with a single writer and avoids the default
journal's reader/writer lock contention. Each thread gets its own connection
(`check_same_thread=False` + short-lived connections) — SQLite connections are
not safe to share across threads.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    parent_id    TEXT,
    system       TEXT NOT NULL,
    model        TEXT,
    created_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    seq          INTEGER NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,          -- JSON-encoded content blocks
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);
CREATE TABLE IF NOT EXISTS compactions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    before_tokens INTEGER NOT NULL,
    after_tokens  INTEGER NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_compactions_session ON compactions(session_id, id);
"""


class SessionStore:
    def __init__(self, path: str | Path = "harness.db") -> None:
        self.path = str(path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # --------------------------------------------------------------- sessions --
    def create_session(
        self, session_id: str, system: str, model: str | None = None, parent_id: str | None = None
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO sessions(id, parent_id, system, model, created_at) "
                "VALUES (?,?,?,?,?)",
                (session_id, parent_id, system, model, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_system(self, session_id: str) -> str | None:
        """Restore the system prompt byte-for-byte. A miss is a silent cache-cost
        multiplier — callers should WARN, not swallow."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT system FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    # --------------------------------------------------------------- messages --
    def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        """Append one Anthropic-shaped message. Called BEFORE tool execution."""
        conn = self._connect()
        try:
            (n,) = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id=?",
                (session_id,),
            ).fetchone()
            conn.execute(
                "INSERT INTO messages(session_id, seq, role, content, created_at) "
                "VALUES (?,?,?,?,?)",
                (
                    session_id,
                    n,
                    message.get("role", "user"),
                    json.dumps(message.get("content", []), default=_json_default),
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY seq",
                (session_id,),
            ).fetchall()
            return [{"role": r, "content": json.loads(c)} for r, c in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------ compaction --
    def record_compaction(self, session_id: str, before_tokens: int, after_tokens: int) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO compactions(session_id, before_tokens, after_tokens, created_at) "
                "VALUES (?,?,?,?)",
                (session_id, before_tokens, after_tokens, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def compaction_stats(self, session_id: str) -> tuple[int, list[float]]:
        """Return (total_compactions, last_two_save_ratios) for anti-thrash and
        head-protection decay. Save ratio = 1 - after/before."""
        conn = self._connect()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM compactions WHERE session_id=?", (session_id,)
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT before_tokens, after_tokens FROM compactions "
                "WHERE session_id=? ORDER BY id DESC LIMIT 2",
                (session_id,),
            ).fetchall()
            saves = [1.0 - (a / b) if b else 0.0 for b, a in rows]
            return count, saves
        finally:
            conn.close()

    def persist_fn(self, session_id: str):
        """Return a `persist` callback bound to this session, for AgentConfig."""
        return lambda message: self.append_message(session_id, message)


def _json_default(obj: Any) -> Any:
    # Anthropic SDK content blocks aren't JSON-native; fall back to their dict
    # form or string repr so persistence never crashes the loop.
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                pass
    return str(obj)
