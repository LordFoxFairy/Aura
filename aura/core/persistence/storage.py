"""Persistent session storage backed by stdlib sqlite3.

Schema (§4.5): single ``messages`` table with UNIQUE(session_id, turn_index).
``turn_index`` is the message ordinal within the session (0-based).
``save`` performs a full replacement: DELETE then INSERT.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from langchain_core.messages import BaseMessage

from aura.core.history import deserialize_messages, serialize_messages
from aura.core.persistence import journal

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    turn_index   INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(session_id, turn_index)
);
CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id, turn_index);
"""


class SessionStorage:
    """SQLite-backed storage for per-session message lists."""

    _conn: sqlite3.Connection

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SessionStorage:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def save(self, session_id: str, messages: list[BaseMessage]) -> None:
        """Full replacement: wipe session's rows and insert the current list."""
        journal.write(
            "storage_save", session=session_id, count=len(messages),
        )
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            payloads = serialize_messages(messages)
            rows: list[tuple[str, int, str]] = [
                (session_id, i, json.dumps(payload)) for i, payload in enumerate(payloads)
            ]
            if rows:
                cur.executemany(
                    "INSERT INTO messages (session_id, turn_index, payload_json) VALUES (?, ?, ?)",
                    rows,
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def load(self, session_id: str) -> list[BaseMessage]:
        """Return all messages for *session_id* ordered by turn_index."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT payload_json FROM messages WHERE session_id = ? ORDER BY turn_index",
            (session_id,),
        )
        dicts = [json.loads(row[0]) for row in cur.fetchall()]
        messages = deserialize_messages(dicts)
        journal.write(
            "storage_load", session=session_id, count=len(messages),
        )
        return messages

    def clear(self, session_id: str) -> None:
        """Delete all rows for *session_id*."""
        journal.write("storage_clear", session=session_id)
        cur = self._conn.cursor()
        cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._conn.commit()

    def list_sessions(self) -> list[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT DISTINCT session_id FROM messages ORDER BY session_id")
        return [row[0] for row in cur.fetchall()]

    def exists(self, session_id: str) -> bool:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1", (session_id,),
        )
        return cur.fetchone() is not None
