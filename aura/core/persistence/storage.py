"""Persistent session storage backed by stdlib sqlite3.

Schema: one ``messages`` table with UNIQUE(session_id, turn_index), where
``turn_index`` is the 0-based message ordinal within a session. ``save``
performs a full replacement (DELETE + INSERT) so a torn write can't
leave a half-persisted turn on disk.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from aura.core.persistence import journal

#: First-prompt preview length in characters (matches list_sessions ellipsis).
_PREVIEW_MAX_CHARS: int = 79


@dataclass(frozen=True)
class SessionMeta:
    """Lightweight metadata snapshot for a single persisted session.

    Returned by :meth:`SessionStorage.list_sessions` — frozen so the
    picker / ``/resume`` UI can pass instances around without aliasing
    bugs. ``first_user_prompt`` is truncated to roughly one line for
    use as a row label; the full message is recovered via
    :meth:`SessionStorage.load`.
    """

    session_id: str
    created_at: datetime
    last_used_at: datetime
    message_count: int
    first_user_prompt: str


@dataclass(frozen=True)
class TranscriptMeta:
    """Lightweight metadata for a persisted subagent transcript file."""

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
    """SQLite-backed storage for per-session message lists."""

    _conn: sqlite3.Connection

    def __init__(self, path: Path) -> None:
        self._path = path
        # ``check_same_thread=False`` lets background threads (tests using
        # ThreadPoolExecutor, or future producer threads) reach the same
        # connection without spinning up their own.
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        # F-05-002 — migration from the legacy ``messages`` table to the
        # JSONL-per-session layout. Runs once; idempotent because we
        # rename the source DB after draining it. Skip for in-memory dbs
        # (tests construct fresh ``:memory:`` instances each time and
        # there's no on-disk legacy data to drain).
        if str(path) != ":memory:":
            self._maybe_migrate_legacy()
        # Append synchronisation — JSONL writes from concurrent threads
        # must not interleave bytes mid-line. ``threading.Lock`` is a
        # cheap mutex; the file open + write + flush stay inside it.
        import threading
        self._append_lock = threading.Lock()

    def _maybe_migrate_legacy(self) -> None:
        """Drain legacy ``messages`` rows into JSONL once + back up the file.

        After the drain the source SQLite file is renamed to
        ``<path>.legacy-<ts>`` so the source-of-truth shifts to JSONL
        and a future construction is a no-op (the renamed file no longer
        sits at ``self._path``).
        """
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT session_id, payload_json FROM messages ORDER BY session_id, turn_index"
            )
            rows = cur.fetchall()
        except sqlite3.DatabaseError:
            return
        if not rows:
            return
        # Group by session_id.
        from collections import defaultdict
        from datetime import UTC
        buckets: dict[str, list[str]] = defaultdict(list)
        for sid, payload in rows:
            buckets[sid].append(payload)
        any_drained = False
        sessions_dir = self._path.parent / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        for sid, payloads in buckets.items():
            jsonl = sessions_dir / f"{sid}.jsonl"
            if jsonl.exists():
                continue  # already migrated this session
            with jsonl.open("w", encoding="utf-8") as fh:
                for p in payloads:
                    envelope = {
                        "ts": datetime.now(UTC).isoformat(),
                        "payload": json.loads(p),
                    }
                    fh.write(json.dumps(envelope, ensure_ascii=False))
                    fh.write("\n")
            any_drained = True
            # Update index.
            self._refresh_index_for_session(sid, jsonl)
        if any_drained:
            # Rename the source DB so this branch does not fire again.
            backup = self._path.with_suffix(
                self._path.suffix + f".legacy-{int(datetime.now().timestamp())}"
            )
            self._conn.commit()
            self._conn.close()
            import contextlib as _contextlib
            with _contextlib.suppress(OSError):
                self._path.rename(backup)
            # Re-open a fresh DB at the original path so downstream
            # callers keep working.
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()

    def _refresh_index_for_session(self, session_id: str, jsonl_path: Path) -> None:
        """Recompute the index row for a session from its JSONL file."""
        sessions_dir = self._path.parent / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        index_path = sessions_dir / "index.sqlite"
        # Count + extract first prompt.
        message_count = 0
        first_prompt = ""
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        env = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    payload = env.get("payload") if isinstance(env, dict) else None
                    if not isinstance(payload, dict):
                        # Legacy / direct payload form.
                        payload = env if isinstance(env, dict) else None
                    if payload is None:
                        continue
                    message_count += 1
                    if (
                        not first_prompt
                        and payload.get("type") == "human"
                    ):
                        data = payload.get("data") or {}
                        content = data.get("content")
                        if isinstance(content, str):
                            first_prompt = content
        idx = sqlite3.connect(str(index_path))
        try:
            idx.executescript(
                "CREATE TABLE IF NOT EXISTS sessions("
                "session_id TEXT PRIMARY KEY, "
                "message_count INTEGER NOT NULL DEFAULT 0, "
                "first_user_prompt TEXT NOT NULL DEFAULT '', "
                "last_used_at TEXT NOT NULL DEFAULT (datetime('now')));"
            )
            idx.execute(
                "INSERT OR REPLACE INTO sessions(session_id, message_count, "
                "first_user_prompt, last_used_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (session_id, message_count, first_prompt),
            )
            idx.commit()
        finally:
            idx.close()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SessionStorage:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _validate_session_id(self, session_id: str) -> None:
        if not session_id or "/" in session_id or ".." in session_id:
            raise ValueError(f"invalid session_id: {session_id!r}")

    def _sessions_dir(self) -> Path:
        return self._path.parent / "sessions"

    def append(self, session_id: str, message: BaseMessage) -> None:
        """Append a single message to *session_id*'s persisted history.

        Round 6M JSONL persistence — appends one wrapped envelope line
        to ``<storage_root>/sessions/<session_id>.jsonl`` and refreshes
        the SQLite index row's count + first-user-prompt. Validates
        ``session_id`` strictly so a malicious id can't escape the
        sessions/ directory via path traversal.
        """
        self._validate_session_id(session_id)
        sessions_dir = self._sessions_dir()
        sessions_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = sessions_dir / f"{session_id}.jsonl"
        # Write the message envelope under the lock so concurrent
        # threads can't interleave mid-line.
        from datetime import UTC
        envelope = {
            "ts": datetime.now(UTC).isoformat(),
            "payload": messages_to_dict([message])[0],
        }
        with self._append_lock, jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(envelope, ensure_ascii=False))
            fh.write("\n")
        # Refresh the index from the JSONL.
        self._refresh_index_for_session(session_id, jsonl_path)

    def write_subagent_transcript(
        self, task_id: str, messages: list[BaseMessage],
    ) -> Path:
        """Persist a subagent's full transcript JSONL.

        Lives under ``<root>/subagents/subagent-<task_id>.jsonl`` so the
        :meth:`list_subagent_transcripts` enumerator can find it. Returns
        the path written.
        """
        sub_dir = self._path.parent / "subagents"
        sub_dir.mkdir(parents=True, exist_ok=True)
        path = sub_dir / f"subagent-{task_id}.jsonl"
        payloads = messages_to_dict(messages)
        with path.open("w", encoding="utf-8") as fh:
            for payload in payloads:
                fh.write(json.dumps(payload, ensure_ascii=False))
                fh.write("\n")
        return path

    def list_subagent_transcripts(self) -> list[TranscriptMeta]:
        """Enumerate persisted subagent transcripts, newest first."""
        sub_dir = self._path.parent / "subagents"
        if not sub_dir.is_dir():
            return []
        out: list[TranscriptMeta] = []
        for p in sub_dir.iterdir():
            if not p.is_file() or not p.name.startswith("subagent-"):
                continue
            if not p.name.endswith(".jsonl"):
                continue
            task_id = p.name[len("subagent-"): -len(".jsonl")]
            try:
                with p.open("r", encoding="utf-8") as fh:
                    count = sum(1 for line in fh if line.strip())
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
            except OSError:
                continue
            out.append(
                TranscriptMeta(
                    task_id=task_id,
                    path=p,
                    message_count=count,
                    last_modified=mtime,
                )
            )
        out.sort(key=lambda m: m.last_modified, reverse=True)
        return out

    def load_subagent_transcript(self, task_id: str) -> list[BaseMessage]:
        """Reload a persisted subagent transcript by task_id."""
        path = self._path.parent / "subagents" / f"subagent-{task_id}.jsonl"
        if not path.exists():
            return []
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

    def save(self, session_id: str, messages: list[BaseMessage]) -> None:
        """Save *session_id*'s history.

        If the new history is a strict prefix-extension of what's on
        disk (the common case during normal turn append), the new
        messages are appended via the same JSONL append path so the
        file's first N lines remain byte-identical (callers depend on
        this for deduplicated cache writes / offline diff). Otherwise
        the JSONL file is rewritten atomically (compaction shrinks
        history, so existing lines must be replaced wholesale).
        """
        self._validate_session_id(session_id)
        journal.write(
            "storage_save", session=session_id, count=len(messages),
        )
        sessions_dir = self._sessions_dir()
        sessions_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = sessions_dir / f"{session_id}.jsonl"
        # Compare existing JSONL prefix against the new payload list.
        existing = self._read_jsonl_payloads(jsonl_path)
        new_payloads = messages_to_dict(messages)
        if (
            len(new_payloads) >= len(existing)
            and new_payloads[: len(existing)] == existing
        ):
            # Pure append — write only the tail.
            tail = new_payloads[len(existing):]
            from datetime import UTC
            with self._append_lock, jsonl_path.open("a", encoding="utf-8") as fh:
                for p in tail:
                    envelope = {
                        "ts": datetime.now(UTC).isoformat(),
                        "payload": p,
                    }
                    fh.write(json.dumps(envelope, ensure_ascii=False))
                    fh.write("\n")
        else:
            # Atomic rewrite via temp + rename.
            from datetime import UTC
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
        self._refresh_index_for_session(session_id, jsonl_path)
        # Mirror into the legacy messages table so the in-memory mode +
        # other consumers that index it keep working.
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
        """Read raw payload dicts from a session JSONL (skip torn lines)."""
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
                    # Bare-payload form — still accept.
                    out.append(env)
        return out

    def load(self, session_id: str) -> list[BaseMessage]:
        """Return all messages for *session_id* in append order.

        Always touches the SQLite connection first so callers that
        ``close()`` the storage and then attempt a load see the
        underlying ``sqlite3.ProgrammingError`` they expect (matches the
        pre-JSONL contract).
        """
        self._validate_session_id(session_id)
        # Touch the connection first — raises ProgrammingError if closed.
        cur = self._conn.cursor()
        # Prefer JSONL (the source of truth post-migration).
        jsonl_path = self._sessions_dir() / f"{session_id}.jsonl"
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
        """Delete all rows for *session_id*."""
        self._validate_session_id(session_id)
        journal.write("storage_clear", session=session_id)
        cur = self._conn.cursor()
        cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._conn.commit()
        # Drop the JSONL + index row.
        jsonl_path = self._sessions_dir() / f"{session_id}.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()
        index_path = self._sessions_dir() / "index.sqlite"
        if index_path.exists():
            idx = sqlite3.connect(str(index_path))
            try:
                idx.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                idx.commit()
            finally:
                idx.close()

    # ------------------------------------------------------------------
    # Resume support: enumeration of persisted sessions
    # ------------------------------------------------------------------

    def list_sessions(self, *, limit: int = 20) -> list[SessionMeta]:
        """Return up to ``limit`` recent sessions, most-recent-first.

        Prefers the JSONL index when present; falls back to the legacy
        messages table for in-memory and pre-migration callers.
        """
        index_path = self._sessions_dir() / "index.sqlite"
        out: list[SessionMeta] = []
        if index_path.exists():
            idx = sqlite3.connect(str(index_path))
            try:
                rows = idx.execute(
                    "SELECT session_id, message_count, first_user_prompt, "
                    "last_used_at FROM sessions ORDER BY last_used_at DESC, "
                    "session_id ASC LIMIT ?",
                    (limit,),
                ).fetchall()
            finally:
                idx.close()
            for sid, count, prompt, last in rows:
                try:
                    last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M:%S")
                except (TypeError, ValueError):
                    last_dt = datetime.now()
                out.append(SessionMeta(
                    session_id=sid,
                    created_at=last_dt,
                    last_used_at=last_dt,
                    message_count=int(count),
                    first_user_prompt=_truncate_one_line(prompt or ""),
                ))
            if out:
                return out
        # Fallback: legacy messages table.
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
        """Return the distinct session count on disk."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT session_id) FROM messages")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def list_team_ids(self) -> list[str]:
        """Return distinct team IDs persisted on disk.

        Walks ``<storage_dir>/teams/`` and returns each subdirectory
        name. The team manager writes ``teams/<team_id>/config.json``
        atomically (see :meth:`team_config_path`), so directory
        existence ⇔ team existence.
        """
        teams_root = self._teams_root()
        if not teams_root.is_dir():
            return []
        return sorted(p.name for p in teams_root.iterdir() if p.is_dir())

    def team_config_path(self, team_id: str) -> Path:
        """Return the on-disk config path for ``team_id``.

        Creates the parent directory eagerly so the team manager's
        atomic-write (write-tmp + ``replace``) doesn't fail with a
        ``FileNotFoundError`` on the very first persist. Lifecycle:
        the directory is created on first read/write of any team's
        config and never removed by storage — team deletion lives in
        the team manager and removes the per-team subtree directly.
        """
        path = self._team_dir(team_id) / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def team_inbox_path(self, team_id: str, member: str) -> Path:
        """Return the JSONL inbox path for ``member`` in ``team_id``.

        The mailbox writes one JSON object per line; the parent
        directory is created on demand. ``member`` may be a member
        name or a reserved sentinel (``"leader"``, ``"broadcast"``).
        """
        path = self._team_dir(team_id) / "inbox" / f"{member}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def team_transcript_path(self, team_id: str, member: str) -> Path:
        """Return the JSONL transcript path for ``member`` in ``team_id``."""
        path = self._team_dir(team_id) / "transcripts" / f"{member}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def team_root(self, team_id: str) -> Path:
        """Return the per-team root directory, created on first use.

        Public counterpart to :meth:`_team_dir` — exposed so test
        scaffolding (and team-deletion code) can compute the on-disk
        location without re-deriving the layout. Creating the directory
        eagerly mirrors the other ``team_*_path`` helpers.
        """
        path = self._team_dir(team_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _team_dir(self, team_id: str) -> Path:
        """Per-team subdirectory under :meth:`_teams_root`."""
        return self._teams_root() / team_id

    def _teams_root(self) -> Path:
        """Return the root directory holding per-team subdirectories."""
        return self._path.parent / "teams"

    def _first_user_prompt(self, session_id: str) -> str:
        """Find the earliest HumanMessage payload for ``session_id``."""
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
    """Collapse to one line + ellipsis if too long.

    Total visual width = ``_PREVIEW_MAX_CHARS + 1`` (the ellipsis col)
    so the picker row stays bounded. Existing ellipses in the source
    text are preserved verbatim — only oversize input gets a forced one.
    """
    flat = " ".join(text.split())
    if len(flat) <= _PREVIEW_MAX_CHARS:
        return flat
    return flat[:_PREVIEW_MAX_CHARS] + "…"


def _parse_naive(s: str) -> datetime:
    """Parse SQLite ``datetime('now')`` output as a naive datetime.

    SQLite emits UTC in the form ``YYYY-MM-DD HH:MM:SS``; we want a
    naive datetime (no tzinfo) so the picker can format it directly
    against ``datetime.now()`` without timezone gymnastics.
    """
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

