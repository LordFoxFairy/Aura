"""Secondary index (index.sqlite): refresh from JSONL + recent-session queries."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_INDEX_SCHEMA_SQL = (
    "CREATE TABLE IF NOT EXISTS sessions("
    "session_id TEXT PRIMARY KEY, "
    "created_at TEXT NOT NULL DEFAULT (datetime('now')), "
    "last_used_at TEXT NOT NULL DEFAULT (datetime('now')), "
    "message_count INTEGER NOT NULL DEFAULT 0, "
    "first_user_prompt TEXT NOT NULL DEFAULT '');"
)


@dataclass(frozen=True)
class IndexRow:
    session_id: str
    message_count: int
    first_user_prompt: str
    last_used_at: str


class SessionIndex:
    """Owns ``index.sqlite``; refreshed per-session from the JSONL transcript."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def refresh(self, session_id: str, jsonl_path: Path) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        message_count, first_prompt = _scan_jsonl(jsonl_path)
        idx = sqlite3.connect(str(self._path))
        try:
            idx.executescript(_INDEX_SCHEMA_SQL)
            idx.execute(
                "INSERT INTO sessions("
                "session_id, message_count, first_user_prompt, "
                "created_at, last_used_at"
                ") VALUES (?, ?, ?, datetime('now'), datetime('now')) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "  message_count = excluded.message_count, "
                "  first_user_prompt = excluded.first_user_prompt, "
                "  last_used_at = datetime('now')",
                (session_id, message_count, first_prompt),
            )
            idx.commit()
        finally:
            idx.close()

    def delete(self, session_id: str) -> None:
        if not self._path.exists():
            return
        idx = sqlite3.connect(str(self._path))
        try:
            idx.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            idx.commit()
        finally:
            idx.close()

    def recent(self, limit: int) -> list[IndexRow]:
        if not self._path.exists():
            return []
        idx = sqlite3.connect(str(self._path))
        try:
            rows = idx.execute(
                "SELECT session_id, message_count, first_user_prompt, "
                "last_used_at FROM sessions ORDER BY last_used_at DESC, "
                "session_id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            idx.close()
        return [
            IndexRow(
                session_id=sid,
                message_count=int(count),
                first_user_prompt=prompt or "",
                last_used_at=last,
            )
            for sid, count, prompt, last in rows
        ]


def _scan_jsonl(jsonl_path: Path) -> tuple[int, str]:
    message_count = 0
    first_prompt = ""
    if not jsonl_path.exists():
        return message_count, first_prompt
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                env = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = env.get("payload") if isinstance(env, dict) else None
            if not isinstance(payload, dict):
                payload = env if isinstance(env, dict) else None
            if payload is None:
                continue
            message_count += 1
            if not first_prompt and payload.get("type") == "human":
                data = payload.get("data") or {}
                content = data.get("content")
                if isinstance(content, str):
                    first_prompt = content
    return message_count, first_prompt
