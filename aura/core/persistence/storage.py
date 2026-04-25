"""Persistent session storage backed by stdlib sqlite3 + JSONL.

Layout (v3, claude-code-compat per-project nested):
    <storage_root>/
      projects/
        <encoded-cwd>/
          <session-id>.jsonl                <- session transcript
          <session-id>/
            subagents/
              agent-<task_id>.jsonl         <- subagent transcript
              agent-<task_id>.meta.json     <- subagent metadata (Track B)
      index.sqlite                          <- index of ALL sessions
      teams/<team_id>/...                   <- team buckets (unchanged)
      sessions.legacy-<ts>/                 <- migrated v2 backup
      subagents.legacy-<ts>/                <- migrated v2 backup

``<encoded-cwd>`` matches claude-code's encoding: an absolute path with
each ``/`` rewritten as ``-``. Example: ``/Users/foo/proj`` →
``-Users-foo-proj``.

Schema: a tiny stdlib ``messages`` table mirrors recent saves so
``:memory:`` callers and legacy in-process consumers keep working
without disk side-effects. ``save`` performs a full replacement
(DELETE + INSERT) so a torn write can't leave a half-persisted turn.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from aura.core.persistence import journal

#: First-prompt preview length in characters (matches list_sessions ellipsis).
_PREVIEW_MAX_CHARS: int = 79


@dataclass(frozen=True)
class SessionMeta:
    """Lightweight metadata snapshot for a single persisted session."""

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


def _encode_cwd_str(cwd: Path) -> str:
    """Encode an absolute cwd into claude-code's projects/ bucket name.

    Each path separator becomes ``-``. The leading ``/`` becomes a leading
    ``-`` so ``/Users/foo`` → ``-Users-foo``. Non-absolute paths are
    resolved against the current working directory first so two callers
    that both pass the same logical project agree on the bucket name.
    """
    abs_cwd = cwd if cwd.is_absolute() else (Path.cwd() / cwd).resolve()
    s = str(abs_cwd)
    return s.replace(os.sep, "-")


class SessionStorage:
    """SQLite-backed storage for per-session message lists.

    The on-disk layout is the v3 per-project nested form documented at
    the module docstring. ``_path`` is the canonical SQLite file
    (``<storage_root>/aura.db`` in production); ``_path.parent`` is the
    storage root and is the anchor for all derived paths.
    """

    _conn: sqlite3.Connection

    def __init__(self, path: Path, *, cwd: Path | None = None) -> None:
        self._path = path
        self._in_memory: bool = str(path) == ":memory:"
        # Capture cwd at construction so the default for new writes is
        # stable for the life of the process — a later os.chdir() must
        # NOT silently rebucket sessions mid-flight.
        self._default_cwd: Path = (
            cwd if cwd is not None else Path.cwd()
        ).resolve()
        if not self._in_memory:
            path.parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` lets background threads (tests using
        # ThreadPoolExecutor, or future producer threads) reach the same
        # connection without spinning up their own.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        # Append synchronisation — JSONL writes from concurrent threads
        # must not interleave bytes mid-line. ``threading.Lock`` is a
        # cheap mutex; the file open + write + flush stay inside it.
        self._append_lock = threading.Lock()
        # F-05-002 — drain the legacy ``messages`` SQLite into JSONL once.
        # Followed by the v2→v3 layout migration (flat sessions/ + flat
        # subagents/ → projects/<encoded-cwd>/...).  Both are skipped
        # for in-memory dbs (no on-disk legacy data possible).
        if not self._in_memory:
            self._maybe_migrate_legacy()
            self._maybe_migrate_v2_to_v3()

    # ------------------------------------------------------------------
    # New v3 path API — claude-code per-project nested layout.
    # ------------------------------------------------------------------

    def _projects_dir(self) -> Path:
        """Return ``<storage_root>/projects/`` (the per-project parent)."""
        return self._path.parent / "projects"

    def _encode_cwd(self, cwd: Path | None = None) -> str:
        """Encode *cwd* (or the storage's default cwd) → ``-Users-foo-bar``."""
        return _encode_cwd_str(cwd if cwd is not None else self._default_cwd)

    def _project_dir(self, cwd: Path | None = None) -> Path:
        """Return ``<projects_dir>/<encoded-cwd>/``."""
        return self._projects_dir() / self._encode_cwd(cwd)

    def session_jsonl_path(
        self, session_id: str, *, cwd: Path | None = None,
    ) -> Path:
        """Return the JSONL transcript path for *session_id*.

        Layout: ``<projects>/<encoded-cwd>/<session-id>.jsonl``.
        """
        self._validate_session_id(session_id)
        return self._project_dir(cwd) / f"{session_id}.jsonl"

    def session_dir(
        self, session_id: str, *, cwd: Path | None = None,
    ) -> Path:
        """Return the per-session sub-resources directory.

        Layout: ``<projects>/<encoded-cwd>/<session-id>/``. The directory
        is the parent of any subagent / future per-session resource.
        """
        self._validate_session_id(session_id)
        return self._project_dir(cwd) / session_id

    def subagent_transcript_path(
        self,
        task_id: str,
        *,
        parent_session_id: str | None = None,
        cwd: Path | None = None,
    ) -> Path:
        """Path for a subagent transcript JSONL.

        Layout: ``<session_dir>/subagents/agent-<task_id>.jsonl``.

        ``parent_session_id=None`` falls back to the ad-hoc bucket
        ``<storage_root>/subagents/agent-<task_id>.jsonl`` so existing
        callers that haven't yet learned to pass ``parent_session_id``
        still get a stable, derivable path. Once Track B threads the
        parent session id through, every transcript naturally nests
        under the parent's session dir.
        """
        self._validate_task_id(task_id)
        if parent_session_id is None:
            return (
                self._path.parent / "subagents" / f"agent-{task_id}.jsonl"
            )
        return (
            self.session_dir(parent_session_id, cwd=cwd)
            / "subagents"
            / f"agent-{task_id}.jsonl"
        )

    def subagent_metadata_path(
        self,
        task_id: str,
        *,
        parent_session_id: str | None = None,
        cwd: Path | None = None,
    ) -> Path:
        """Path for a subagent's sidecar ``.meta.json`` (Track B writes it)."""
        transcript = self.subagent_transcript_path(
            task_id,
            parent_session_id=parent_session_id,
            cwd=cwd,
        )
        return transcript.with_suffix(".meta.json")

    # ------------------------------------------------------------------
    # Migration: legacy SQLite ``messages`` table → JSONL.
    # ------------------------------------------------------------------

    def _maybe_migrate_legacy(self) -> None:
        """Drain the legacy ``messages`` table into JSONL once.

        After draining, the SQLite file is renamed to
        ``<path>.legacy-<ts>`` so the source-of-truth shifts to JSONL
        and a future construction is a no-op (the renamed file no
        longer sits at ``self._path``). Drained sessions land in the
        v3 layout directly, keyed by the storage's default cwd — there
        is no per-row cwd attribution in the legacy schema, so all
        rows go into the current project's bucket. The user can move
        them by hand if a session belongs to a different project.
        """
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT session_id, payload_json FROM messages "
                "ORDER BY session_id, turn_index"
            )
            rows = cur.fetchall()
        except sqlite3.DatabaseError:
            return
        if not rows:
            return
        from collections import defaultdict
        buckets: dict[str, list[str]] = defaultdict(list)
        for sid, payload in rows:
            buckets[sid].append(payload)
        any_drained = False
        for sid, payloads in buckets.items():
            jsonl = self.session_jsonl_path(sid)
            jsonl.parent.mkdir(parents=True, exist_ok=True)
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
            self._refresh_index_for_session(sid, jsonl)
        if any_drained:
            backup = self._path.with_suffix(
                self._path.suffix + f".legacy-{int(datetime.now().timestamp())}"
            )
            self._conn.commit()
            self._conn.close()
            with contextlib.suppress(OSError):
                self._path.rename(backup)
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Migration: v2 flat layout → v3 per-project nested layout.
    # ------------------------------------------------------------------

    def _maybe_migrate_v2_to_v3(self) -> None:
        """One-shot, idempotent v2→v3 migration.

        Detects the v2 markers on disk (``<root>/sessions/`` containing
        ``*.jsonl`` or an ``index.sqlite``, and/or ``<root>/subagents/``
        containing ``*.jsonl``). Renames each to a timestamped backup
        directory so an operator can inspect the originals manually and
        merges any v2 ``index.sqlite`` rows into the new top-level
        ``index.sqlite``. We deliberately do NOT attempt to rebucket
        orphan subagents — the v2 format had no parent_session_id, so
        there's no honest way to map them. The legacy directory is the
        user's responsibility to inspect.
        """
        root = self._path.parent
        legacy_sessions = root / "sessions"
        legacy_subagents = root / "subagents"
        ts = int(datetime.now().timestamp())

        renamed_sessions: Path | None = None
        renamed_subagents: Path | None = None

        # The v2 sessions/ dir is "legacy" only if it contains the v2
        # markers (jsonl files, OR an index.sqlite at its root).  An
        # empty directory or one used solely for v3 transient state is
        # left alone so we don't churn fresh installs.
        if legacy_sessions.is_dir():
            has_jsonl = any(legacy_sessions.glob("*.jsonl"))
            has_index = (legacy_sessions / "index.sqlite").exists()
            if has_jsonl or has_index:
                renamed_sessions = (
                    root / f"sessions.legacy-{ts}"
                )
                with contextlib.suppress(OSError):
                    legacy_sessions.rename(renamed_sessions)

        if legacy_subagents.is_dir():
            has_jsonl = any(legacy_subagents.glob("*.jsonl"))
            if has_jsonl:
                renamed_subagents = (
                    root / f"subagents.legacy-{ts}"
                )
                with contextlib.suppress(OSError):
                    legacy_subagents.rename(renamed_subagents)

        # If we moved a v2 sessions/ dir, fold its index.sqlite rows
        # into the new top-level index.sqlite. Each row ends up
        # bucket-attributed to the storage's default cwd — same caveat
        # as the legacy SQLite migration.
        if renamed_sessions is not None:
            old_index = renamed_sessions / "index.sqlite"
            if old_index.exists():
                self._merge_legacy_index(old_index)

        if renamed_sessions is not None or renamed_subagents is not None:
            journal.write(
                "storage_layout_v3_migration",
                renamed_sessions=(
                    str(renamed_sessions) if renamed_sessions else None
                ),
                renamed_subagents=(
                    str(renamed_subagents) if renamed_subagents else None
                ),
            )

    def _merge_legacy_index(self, old_index_path: Path) -> None:
        """Copy rows from a v2 ``sessions/index.sqlite`` into the new top-level."""
        try:
            old_conn = sqlite3.connect(str(old_index_path))
        except sqlite3.DatabaseError:
            return
        try:
            try:
                rows = old_conn.execute(
                    "SELECT session_id, message_count, first_user_prompt, "
                    "created_at, last_used_at FROM sessions"
                ).fetchall()
            except sqlite3.DatabaseError:
                return
        finally:
            old_conn.close()
        if not rows:
            return
        new_index = self._index_path()
        new_index.parent.mkdir(parents=True, exist_ok=True)
        idx = sqlite3.connect(str(new_index))
        try:
            idx.executescript(_INDEX_SCHEMA_SQL)
            for sid, count, prompt, created, last in rows:
                idx.execute(
                    "INSERT INTO sessions("
                    "session_id, message_count, first_user_prompt, "
                    "created_at, last_used_at"
                    ") VALUES (?, ?, ?, COALESCE(?, datetime('now')), "
                    "COALESCE(?, datetime('now'))) "
                    "ON CONFLICT(session_id) DO UPDATE SET "
                    "  message_count = excluded.message_count, "
                    "  first_user_prompt = excluded.first_user_prompt, "
                    "  last_used_at = excluded.last_used_at",
                    (sid, int(count or 0), prompt or "", created, last),
                )
            idx.commit()
        finally:
            idx.close()

    # ------------------------------------------------------------------
    # Index (top-level v3) helpers.
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        """Return the v3 top-level ``index.sqlite`` path."""
        return self._path.parent / "index.sqlite"

    def _refresh_index_for_session(
        self, session_id: str, jsonl_path: Path,
    ) -> None:
        """Recompute the index row for *session_id* from its JSONL file."""
        index_path = self._index_path()
        index_path.parent.mkdir(parents=True, exist_ok=True)
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
            idx.executescript(_INDEX_SCHEMA_SQL)
            # UPSERT preserves the original ``created_at`` on update.
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

    # ------------------------------------------------------------------
    # Connection lifecycle.
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SessionStorage:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _validate_session_id(self, session_id: str) -> None:
        if not session_id or "/" in session_id or ".." in session_id:
            raise ValueError(f"invalid session_id: {session_id!r}")

    def _validate_task_id(self, task_id: str) -> None:
        if not task_id or "/" in task_id or ".." in task_id:
            raise ValueError(f"invalid task_id: {task_id!r}")

    # ------------------------------------------------------------------
    # Append / load / save.
    # ------------------------------------------------------------------

    def append(self, session_id: str, message: BaseMessage) -> None:
        """Append a single message to *session_id*'s persisted history.

        Writes one wrapped envelope line to the v3 nested JSONL path
        (``<projects>/<encoded-cwd>/<session-id>.jsonl``) and refreshes
        the top-level index row's count + first-user-prompt. Validates
        ``session_id`` strictly so a malicious id can't escape the
        bucket via path traversal.

        For ``:memory:`` storage, the message is mirrored into the
        in-process ``messages`` table only — no on-disk artifacts are
        produced (matches the contract a transient test expects).
        """
        self._validate_session_id(session_id)
        if self._in_memory:
            # Mirror into the in-process table at the next ordinal so
            # ``load`` returns it without round-tripping through disk.
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
        self._refresh_index_for_session(session_id, jsonl_path)

    def write_subagent_transcript(
        self,
        task_id: str,
        messages: list[BaseMessage],
        *,
        parent_session_id: str | None = None,
        cwd: Path | None = None,
    ) -> Path:
        """Persist a subagent's full transcript JSONL.

        Returns the path written. When ``parent_session_id`` is supplied
        the file lives nested under the parent's session dir
        (``<session_dir>/subagents/agent-<task_id>.jsonl``). When not,
        the v2-shaped flat ``<root>/subagents/agent-<task_id>.jsonl``
        is used so existing callers without parent attribution still
        get a stable path.
        """
        self._validate_task_id(task_id)
        path = self.subagent_transcript_path(
            task_id,
            parent_session_id=parent_session_id,
            cwd=cwd,
        )
        if self._in_memory:
            # No on-disk side effects for transient :memory: storage —
            # the path is still returned so callers that pin it on a
            # TaskRecord see a stable, non-leaky value.
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        payloads = messages_to_dict(messages)
        with path.open("w", encoding="utf-8") as fh:
            for payload in payloads:
                fh.write(json.dumps(payload, ensure_ascii=False))
                fh.write("\n")
        return path

    def list_subagent_transcripts(self) -> list[TranscriptMeta]:
        """Enumerate persisted subagent transcripts, newest first.

        Walks every ``subagents/`` directory under the storage root —
        both the legacy flat ``<root>/subagents/`` (for backward read
        compatibility) and every ``<projects>/<encoded-cwd>/<session>/
        subagents/`` (the v3 nested layout). Files starting with
        ``agent-`` (v3) or ``subagent-`` (legacy) are recognised.
        """
        # Dedupe by task_id — Track A and the legacy run.py writer can
        # both produce a file for the same task (one with the
        # ``agent-`` prefix, one with ``subagent-``). Newest wins so a
        # rerun's data isn't masked by a stale older file.
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

        # Legacy flat directory (also where parent_session_id=None
        # writes land).
        flat = self._path.parent / "subagents"
        if flat.is_dir():
            for p in flat.iterdir():
                _consume(p)

        # v3 nested directories: walk every session sub-dir under
        # every project bucket.
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
        """Reload a persisted subagent transcript by task_id.

        Searches every potential location (v3 nested + legacy flat) and
        returns the first match. ``task_id`` is path-validated so a
        malicious value can't escape the search root.
        """
        self._validate_task_id(task_id)
        candidates: list[Path] = []
        # v3 nested first.
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
        # Legacy flat (and parent_session_id=None writes).
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
            self._refresh_index_for_session(session_id, jsonl_path)
        # Mirror into the in-process messages table so :memory: callers
        # and other consumers that index it keep working.
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
                    out.append(env)
        return out

    def load(self, session_id: str) -> list[BaseMessage]:
        """Return all messages for *session_id* in append order.

        Read order:
          1. v3 nested path (the source of truth post-migration).
          2. v2 backup (``<root>/sessions.legacy-*/<session>.jsonl``)
             so a session that didn't migrate cleanly can still be
             resumed by id.
          3. The in-process ``messages`` table (the path used by
             ``:memory:`` callers and tests that pre-fill via ``save``
             without on-disk persistence).
        """
        self._validate_session_id(session_id)
        # Touch the connection first — raises ProgrammingError if closed.
        cur = self._conn.cursor()
        # 1. v3 nested.
        jsonl_path = self.session_jsonl_path(session_id)
        payloads = self._read_jsonl_payloads(jsonl_path)
        # 2. v2 legacy backup buckets.
        if not payloads:
            for legacy_dir in sorted(
                self._path.parent.glob("sessions.legacy-*"),
                reverse=True,
            ):
                legacy_path = legacy_dir / f"{session_id}.jsonl"
                if legacy_path.exists():
                    payloads = self._read_jsonl_payloads(legacy_path)
                    if payloads:
                        break
        # 3. In-process messages table.
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
        """Delete all rows for *session_id* (v3 path + index + table)."""
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
        index_path = self._index_path()
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
    # Resume support: enumeration of persisted sessions.
    # ------------------------------------------------------------------

    def list_sessions(self, *, limit: int = 20) -> list[SessionMeta]:
        """Return up to *limit* recent sessions, most-recent-first.

        Reads from the top-level ``index.sqlite`` (v3 layout, where every
        project's sessions are co-indexed). Falls back to the in-process
        messages table for ``:memory:`` callers.
        """
        index_path = self._index_path()
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
        # Fallback: in-process messages table.
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
        """Return the on-disk config path for ``team_id``."""
        path = self._team_dir(team_id) / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def team_inbox_path(self, team_id: str, member: str) -> Path:
        """Return the JSONL inbox path for ``member`` in ``team_id``."""
        path = self._team_dir(team_id) / "inbox" / f"{member}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def team_transcript_path(self, team_id: str, member: str) -> Path:
        """Return the JSONL transcript path for ``member`` in ``team_id``."""
        path = self._team_dir(team_id) / "transcripts" / f"{member}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def team_root(self, team_id: str) -> Path:
        """Return the per-team root directory, created on first use."""
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


_INDEX_SCHEMA_SQL = (
    "CREATE TABLE IF NOT EXISTS sessions("
    "session_id TEXT PRIMARY KEY, "
    "created_at TEXT NOT NULL DEFAULT (datetime('now')), "
    "last_used_at TEXT NOT NULL DEFAULT (datetime('now')), "
    "message_count INTEGER NOT NULL DEFAULT 0, "
    "first_user_prompt TEXT NOT NULL DEFAULT '');"
)


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
    """Parse SQLite ``datetime('now')`` output as a naive datetime."""
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
