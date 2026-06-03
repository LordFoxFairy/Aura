"""Persistent session storage — sqlite3 index + JSONL transcripts.

Layout (v3, per-project nested)::

    <storage_root>/
      projects/<encoded-cwd>/<session-id>.jsonl
      projects/<encoded-cwd>/<session-id>/subagents/agent-<task>.jsonl
      index.sqlite
      teams/<team_id>/...

``<encoded-cwd>`` rewrites each ``/`` as ``-``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from aura.infrastructure.persistence import journal, storage_paths
from aura.infrastructure.persistence.session_index import SessionIndex

_PREVIEW_MAX_CHARS: int = 79


@dataclass(frozen=True)
class SessionMeta:
    session_id: str
    created_at: datetime
    last_used_at: datetime
    message_count: int
    first_user_prompt: str


@dataclass(frozen=True)
class TranscriptMeta:
    task_id: str
    path: Path
    message_count: int
    last_modified: datetime


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
    """SQLite + JSONL storage for per-session message lists."""

    _conn: sqlite3.Connection

    def __init__(self, path: Path, *, cwd: Path | None = None) -> None:
        self._path = path
        self._in_memory: bool = str(path) == ":memory:"
        # Pin cwd at construction so a later os.chdir() can't silently rebucket sessions.
        self._default_cwd: Path = (
            cwd if cwd is not None else Path.cwd()
        ).resolve()
        if not self._in_memory:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        self._append_lock = threading.Lock()
        self._index = SessionIndex(storage_paths.index_path(self._path.parent))

    @property
    def path(self) -> Path:
        return self._path

    def _projects_dir(self) -> Path:
        return storage_paths.projects_dir(self._path.parent)

    def _encode_cwd(self, cwd: Path | None = None) -> str:
        return storage_paths.encode_cwd(cwd if cwd is not None else self._default_cwd)

    def _project_dir(self, cwd: Path | None = None) -> Path:
        return storage_paths.project_dir(
            self._path.parent, cwd if cwd is not None else self._default_cwd,
        )

    def session_jsonl_path(
        self, session_id: str, *, cwd: Path | None = None,
    ) -> Path:
        return storage_paths.session_jsonl_path(
            self._path.parent, cwd if cwd is not None else self._default_cwd, session_id,
        )

    def memory_dir(self, *, cwd: Path | None = None) -> Path:
        # Lazy: not created until first write.
        return storage_paths.memory_dir(
            self._path.parent, cwd if cwd is not None else self._default_cwd,
        )

    def session_dir(
        self, session_id: str, *, cwd: Path | None = None,
    ) -> Path:
        return storage_paths.session_dir(
            self._path.parent, cwd if cwd is not None else self._default_cwd, session_id,
        )

    def subagent_transcript_path(
        self,
        task_id: str,
        *,
        parent_session_id: str | None = None,
        cwd: Path | None = None,
    ) -> Path:
        """``parent_session_id=None`` falls back to the flat ad-hoc bucket."""
        return storage_paths.subagent_transcript_path(
            self._path.parent,
            cwd if cwd is not None else self._default_cwd,
            task_id,
            parent_session_id,
        )

    def subagent_metadata_path(
        self,
        task_id: str,
        *,
        parent_session_id: str | None = None,
        cwd: Path | None = None,
    ) -> Path:
        transcript = self.subagent_transcript_path(
            task_id,
            parent_session_id=parent_session_id,
            cwd=cwd,
        )
        return transcript.with_suffix(".meta.json")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SessionStorage:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _validate_session_id(self, session_id: str) -> None:
        storage_paths.validate_session_id(session_id)

    def _validate_task_id(self, task_id: str) -> None:
        storage_paths.validate_task_id(task_id)

    def append(self, session_id: str, message: BaseMessage) -> None:
        """Append one envelope line + refresh the index; ``:memory:`` skips disk."""
        self._validate_session_id(session_id)
        if self._in_memory:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM messages "
                "WHERE session_id = ?",
                (session_id,),
            )
            next_idx = int(cur.fetchone()[0])
            payload = json.dumps(messages_to_dict([message])[0])
            cur.execute(
                "INSERT INTO messages "
                "(session_id, turn_index, payload_json) VALUES (?, ?, ?)",
                (session_id, next_idx, payload),
            )
            self._conn.commit()
            return
        jsonl_path = self.session_jsonl_path(session_id)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "ts": datetime.now(UTC).isoformat(),
            "payload": messages_to_dict([message])[0],
        }
        with self._append_lock, jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(envelope, ensure_ascii=False))
            fh.write("\n")
        self._index.refresh(session_id, jsonl_path)

    def write_subagent_transcript(
        self,
        task_id: str,
        messages: list[BaseMessage],
        *,
        parent_session_id: str | None = None,
        cwd: Path | None = None,
    ) -> Path:
        self._validate_task_id(task_id)
        path = self.subagent_transcript_path(
            task_id,
            parent_session_id=parent_session_id,
            cwd=cwd,
        )
        if self._in_memory:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        payloads = messages_to_dict(messages)
        with path.open("w", encoding="utf-8") as fh:
            for payload in payloads:
                fh.write(json.dumps(payload, ensure_ascii=False))
                fh.write("\n")
        return path

    def list_subagent_transcripts(self) -> list[TranscriptMeta]:
        """Enumerate persisted subagent transcripts, newest first, deduped by task_id."""
        by_task: dict[str, TranscriptMeta] = {}

        def _consume(p: Path) -> None:
            if not p.is_file():
                return
            name = p.name
            task_id: str | None = None
            if name.startswith("agent-") and name.endswith(".jsonl"):
                task_id = name[len("agent-"): -len(".jsonl")]
            elif name.startswith("subagent-") and name.endswith(".jsonl"):
                task_id = name[len("subagent-"): -len(".jsonl")]
            if task_id is None:
                return
            try:
                with p.open("r", encoding="utf-8") as fh:
                    count = sum(1 for line in fh if line.strip())
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
            except OSError:
                return
            current = by_task.get(task_id)
            if current is None or mtime > current.last_modified:
                by_task[task_id] = TranscriptMeta(
                    task_id=task_id,
                    path=p,
                    message_count=count,
                    last_modified=mtime,
                )

        flat = self._path.parent / "subagents"
        if flat.is_dir():
            for p in flat.iterdir():
                _consume(p)

        projects = self._projects_dir()
        if projects.is_dir():
            for proj in projects.iterdir():
                if not proj.is_dir():
                    continue
                for session_sub in proj.iterdir():
                    if not session_sub.is_dir():
                        continue
                    sub_dir = session_sub / "subagents"
                    if not sub_dir.is_dir():
                        continue
                    for p in sub_dir.iterdir():
                        _consume(p)

        out = list(by_task.values())
        out.sort(key=lambda m: m.last_modified, reverse=True)
        return out

    def load_subagent_transcript(self, task_id: str) -> list[BaseMessage]:
        self._validate_task_id(task_id)
        candidates: list[Path] = []
        projects = self._projects_dir()
        if projects.is_dir():
            for proj in projects.iterdir():
                if not proj.is_dir():
                    continue
                for session_sub in proj.iterdir():
                    if not session_sub.is_dir():
                        continue
                    sub_dir = session_sub / "subagents"
                    if not sub_dir.is_dir():
                        continue
                    candidates.append(sub_dir / f"agent-{task_id}.jsonl")
                    candidates.append(sub_dir / f"subagent-{task_id}.jsonl")
        flat = self._path.parent / "subagents"
        candidates.append(flat / f"agent-{task_id}.jsonl")
        candidates.append(flat / f"subagent-{task_id}.jsonl")

        for path in candidates:
            if not path.exists():
                continue
            dicts: list[dict[str, object]] = []
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        dicts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return list(messages_from_dict(dicts))
        return []

    def save(self, session_id: str, messages: list[BaseMessage]) -> None:
        """Save full history; prefix-extension appends in place, divergence rewrites atomically."""
        self._validate_session_id(session_id)
        journal.write(
            "storage_save", session=session_id, count=len(messages),
        )
        new_payloads = messages_to_dict(messages)
        if not self._in_memory:
            jsonl_path = self.session_jsonl_path(session_id)
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            existing = self._read_jsonl_payloads(jsonl_path)
            if (
                len(new_payloads) >= len(existing)
                and new_payloads[: len(existing)] == existing
            ):
                tail = new_payloads[len(existing):]
                with self._append_lock, jsonl_path.open("a", encoding="utf-8") as fh:
                    for p in tail:
                        envelope = {
                            "ts": datetime.now(UTC).isoformat(),
                            "payload": p,
                        }
                        fh.write(json.dumps(envelope, ensure_ascii=False))
                        fh.write("\n")
            else:
                tmp = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
                with tmp.open("w", encoding="utf-8") as fh:
                    for p in new_payloads:
                        envelope = {
                            "ts": datetime.now(UTC).isoformat(),
                            "payload": p,
                        }
                        fh.write(json.dumps(envelope, ensure_ascii=False))
                        fh.write("\n")
                tmp.replace(jsonl_path)
            self._index.refresh(session_id, jsonl_path)
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            rows: list[tuple[str, int, str]] = [
                (session_id, i, json.dumps(p)) for i, p in enumerate(new_payloads)
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

    def _read_jsonl_payloads(self, jsonl_path: Path) -> list[dict[str, object]]:
        if not jsonl_path.exists():
            return []
        out: list[dict[str, object]] = []
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    env = json.loads(line)
                except json.JSONDecodeError:
                    journal.write(
                        "storage_skip_corrupt_line",
                        path=str(jsonl_path),
                    )
                    continue
                if isinstance(env, dict) and "payload" in env:
                    payload = env["payload"]
                    if isinstance(payload, dict):
                        out.append(payload)
                elif isinstance(env, dict):
                    out.append(env)
        return out

    def load(self, session_id: str) -> list[BaseMessage]:
        """Load messages; falls back to the in-process table when JSONL is empty/absent."""
        self._validate_session_id(session_id)
        cur = self._conn.cursor()
        jsonl_path = self.session_jsonl_path(session_id)
        payloads = self._read_jsonl_payloads(jsonl_path)
        if not payloads:
            cur.execute(
                "SELECT payload_json FROM messages WHERE session_id = ? ORDER BY turn_index",
                (session_id,),
            )
            payloads = [json.loads(row[0]) for row in cur.fetchall()]
        messages = list(messages_from_dict(payloads))
        journal.write(
            "storage_load", session=session_id, count=len(messages),
        )
        return messages

    def clear(self, session_id: str) -> None:
        self._validate_session_id(session_id)
        journal.write("storage_clear", session=session_id)
        cur = self._conn.cursor()
        cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._conn.commit()
        if self._in_memory:
            return
        jsonl_path = self.session_jsonl_path(session_id)
        if jsonl_path.exists():
            jsonl_path.unlink()
        self._index.delete(session_id)

    def list_sessions(self, *, limit: int = 20) -> list[SessionMeta]:
        """Recent sessions newest-first; ``:memory:`` falls back to the in-process table."""
        out: list[SessionMeta] = []
        for row in self._index.recent(limit):
            try:
                last_dt = datetime.strptime(row.last_used_at, "%Y-%m-%d %H:%M:%S")
            except (TypeError, ValueError):
                last_dt = datetime.now()
            out.append(SessionMeta(
                session_id=row.session_id,
                created_at=last_dt,
                last_used_at=last_dt,
                message_count=row.message_count,
                first_user_prompt=_truncate_one_line(row.first_user_prompt),
            ))
        if out:
            return out
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT
                session_id,
                MIN(created_at) AS created_at,
                MAX(created_at) AS last_used_at,
                COUNT(*) AS message_count
            FROM messages
            GROUP BY session_id
            ORDER BY MAX(created_at) DESC, session_id ASC
            LIMIT ?
            """,
            (limit,),
        )
        rows_legacy = cur.fetchall()
        for session_id, created_at, last_used_at, msg_count in rows_legacy:
            preview = self._first_user_prompt(session_id)
            out.append(
                SessionMeta(
                    session_id=session_id,
                    created_at=_parse_naive(created_at),
                    last_used_at=_parse_naive(last_used_at),
                    message_count=int(msg_count),
                    first_user_prompt=preview,
                ),
            )
        return out

    def session_count(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT session_id) FROM messages")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def list_team_ids(self) -> list[str]:
        teams_root = storage_paths.teams_root(self._path.parent)
        if not teams_root.is_dir():
            return []
        return sorted(p.name for p in teams_root.iterdir() if p.is_dir())

    def team_config_path(self, team_id: str) -> Path:
        path = self._team_dir(team_id) / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def team_inbox_path(self, team_id: str, member: str) -> Path:
        path = self._team_dir(team_id) / "inbox" / f"{member}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def team_transcript_path(self, team_id: str, member: str) -> Path:
        path = self._team_dir(team_id) / "transcripts" / f"{member}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def team_root(self, team_id: str) -> Path:
        path = self._team_dir(team_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _team_dir(self, team_id: str) -> Path:
        return storage_paths.team_dir(self._path.parent, team_id)

    def _first_user_prompt(self, session_id: str) -> str:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT payload_json FROM messages "
            "WHERE session_id = ? ORDER BY turn_index",
            (session_id,),
        )
        for (payload_json,) in cur.fetchall():
            try:
                payload = json.loads(payload_json)
            except (json.JSONDecodeError, TypeError):
                continue
            msg_type = payload.get("type")
            data = payload.get("data") or {}
            content = data.get("content")
            if msg_type == "human" and isinstance(content, str) and content:
                return _truncate_one_line(content)
        return ""


def _truncate_one_line(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= _PREVIEW_MAX_CHARS:
        return flat
    return flat[:_PREVIEW_MAX_CHARS] + "…"


def _parse_naive(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
