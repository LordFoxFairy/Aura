"""F-05-002 / V14-Persistence-V3: per-project nested JSONL session storage.

Pins the wire-level invariants ``aura.core.persistence.storage`` exposes:

  - ``append`` writes one JSONL line per call to the v3 path.
  - ``load`` returns messages in the order they landed.
  - Corrupt lines are skipped + journaled, not raised.
  - The top-level ``index.sqlite`` is updated on every ``append``.
  - ``list_sessions`` reads from the index across ALL project buckets.
  - Concurrent appends do not interleave bytes mid-line.
  - Migration from the legacy SQLite ``messages`` table is one-shot
    and idempotent and lands directly in the v3 layout.
  - Migration from the v2 flat layout (``<root>/sessions/`` +
    ``<root>/subagents/``) renames the originals to timestamped
    backups and merges the v2 index into the new top-level index.
  - A session resume after JSONL writes loads cleanly.
  - A session in the v2 backup remains readable via :meth:`load`.
  - Two project cwds isolate cleanly into separate buckets.
"""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from aura.core.persistence.storage import SessionStorage


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _project_root(storage: SessionStorage) -> Path:
    """Return the project bucket dir for the storage's default cwd."""
    return storage._project_dir()


def test_append_writes_one_line(tmp_path: Path) -> None:
    project_cwd = tmp_path / "proj-A"
    project_cwd.mkdir()
    storage = SessionStorage(tmp_path / "aura.db", cwd=project_cwd)
    storage.append("session-1", HumanMessage(content="hello"))
    storage.append("session-1", AIMessage(content="hi back"))

    jsonl = _project_root(storage) / "session-1.jsonl"
    assert jsonl.exists()
    # Encoded cwd matches CC convention: /-replaced.
    assert jsonl.parent.name.startswith("-")
    lines = _read_jsonl(jsonl)
    assert len(lines) == 2
    for line in lines:
        assert "ts" in line
        assert "payload" in line
        payload = line["payload"]
        assert isinstance(payload, dict)
        assert "type" in payload


def test_load_returns_messages_in_order(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    for i in range(5):
        storage.append("s", HumanMessage(content=f"msg-{i}"))
    msgs = storage.load("s")
    assert [m.content for m in msgs] == [f"msg-{i}" for i in range(5)]


def test_load_skips_corrupt_line_and_logs(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    storage.append("s", HumanMessage(content="good"))

    jsonl = storage.session_jsonl_path("s")
    with jsonl.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
        f.write('{"payload": {"type": "human", "data": {"content": "after-tear"}}}\n')

    msgs = storage.load("s")
    contents = [m.content for m in msgs]
    assert "good" in contents
    assert "after-tear" in contents
    assert len(msgs) == 2


def test_index_sqlite_updated_on_append(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    storage.append("s", HumanMessage(content="first prompt"))

    # The index is at the storage root, NOT nested under sessions/.
    index = tmp_path / "index.sqlite"
    assert index.exists()
    conn = sqlite3.connect(str(index))
    row = conn.execute(
        "SELECT session_id, message_count, first_user_prompt FROM sessions",
    ).fetchone()
    conn.close()
    assert row[0] == "s"
    assert row[1] == 1
    assert row[2] == "first prompt"

    storage.append("s", AIMessage(content="reply"))
    conn = sqlite3.connect(str(index))
    count = conn.execute(
        "SELECT message_count FROM sessions WHERE session_id = ?", ("s",),
    ).fetchone()[0]
    conn.close()
    assert count == 2


def test_list_sessions_reads_index(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    storage.append("alpha", HumanMessage(content="a"))
    storage.append("beta", HumanMessage(content="b"))
    storage.append("gamma", HumanMessage(content="c"))

    metas = storage.list_sessions()
    ids = {m.session_id for m in metas}
    assert ids == {"alpha", "beta", "gamma"}

    storage.append("alpha", AIMessage(content="follow-up"))
    metas = storage.list_sessions()
    assert metas[0].session_id == "alpha"


def test_concurrent_appends_dont_interleave_lines(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    payloads = [f"thread-{i}-msg" for i in range(40)]

    def _do(content: str) -> None:
        storage.append("s", HumanMessage(content=content))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_do, payloads))

    jsonl = storage.session_jsonl_path("s")
    lines = _read_jsonl(jsonl)
    assert len(lines) == 40
    seen = {line["payload"]["data"]["content"] for line in lines}  # type: ignore[index]
    assert seen == set(payloads)


def test_migration_from_legacy_sqlite_format(tmp_path: Path) -> None:
    """Pre-v0.16 SQLite db is drained into the v3 JSONL layout."""
    legacy = tmp_path / "aura.db"
    conn = sqlite3.connect(str(legacy))
    conn.executescript(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(session_id, turn_index)
        );
        """
    )
    payloads = [
        {"type": "human", "data": {"content": "legacy hello"}},
        {"type": "ai", "data": {"content": "legacy reply"}},
    ]
    for i, p in enumerate(payloads):
        conn.execute(
            "INSERT INTO messages (session_id, turn_index, payload_json) "
            "VALUES (?, ?, ?)",
            ("legacy-session", i, json.dumps(p)),
        )
    conn.commit()
    conn.close()

    storage = SessionStorage(legacy, cwd=tmp_path)
    msgs = storage.load("legacy-session")
    assert [m.content for m in msgs] == ["legacy hello", "legacy reply"]

    backups = list(tmp_path.glob("aura.db.legacy-*"))
    assert backups, "legacy db should be renamed, not deleted"
    # Drained data must land in the v3 nested path.
    drained = storage.session_jsonl_path("legacy-session")
    assert drained.exists()


def test_migration_idempotent(tmp_path: Path) -> None:
    """Running the migration twice does not double-insert messages."""
    legacy = tmp_path / "aura.db"
    conn = sqlite3.connect(str(legacy))
    conn.executescript(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(session_id, turn_index)
        );
        """
    )
    conn.execute(
        "INSERT INTO messages (session_id, turn_index, payload_json) "
        "VALUES (?, ?, ?)",
        ("s", 0, json.dumps({"type": "human", "data": {"content": "once"}})),
    )
    conn.commit()
    conn.close()

    storage1 = SessionStorage(legacy, cwd=tmp_path)
    msgs1 = storage1.load("s")
    jsonl = storage1.session_jsonl_path("s")
    storage1.close()

    before_size = jsonl.stat().st_size

    storage2 = SessionStorage(legacy, cwd=tmp_path)
    msgs2 = storage2.load("s")
    storage2.close()

    after_size = jsonl.stat().st_size
    assert before_size == after_size
    assert [m.content for m in msgs1] == [m.content for m in msgs2] == ["once"]


def test_session_resume_after_jsonl_write_loads_correctly(
    tmp_path: Path,
) -> None:
    """Persistence-shape change must keep ``Agent``-level resume semantics."""
    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    storage.append("resume-target", HumanMessage(content="first turn"))
    storage.append("resume-target", AIMessage(content="reply"))
    storage.append("resume-target", HumanMessage(content="second turn"))

    storage.close()
    reopened = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    msgs = reopened.load("resume-target")
    assert [m.content for m in msgs] == [
        "first turn", "reply", "second turn",
    ]
    metas = reopened.list_sessions()
    assert any(m.session_id == "resume-target" for m in metas)


def test_save_with_appended_tail_uses_append_path(tmp_path: Path) -> None:
    """``save`` with same prefix + new tail goes through append, not rewrite."""
    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    h1 = HumanMessage(content="one")
    h2 = AIMessage(content="two")
    h3 = HumanMessage(content="three")
    storage.save("s", [h1, h2])
    jsonl = storage.session_jsonl_path("s")
    lines_before = _read_jsonl(jsonl)
    assert len(lines_before) == 2

    storage.save("s", [h1, h2, h3])
    lines_after = _read_jsonl(jsonl)
    assert lines_after[0] == lines_before[0]
    assert lines_after[1] == lines_before[1]
    assert len(lines_after) == 3


def test_save_with_shrunk_history_rewrites_atomically(tmp_path: Path) -> None:
    """``save`` after compact (shrunk history) rewrites the JSONL file."""
    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    storage.save("s", [
        HumanMessage(content="a"),
        AIMessage(content="b"),
        HumanMessage(content="c"),
    ])
    storage.save("s", [HumanMessage(content="summary-only")])
    msgs = storage.load("s")
    assert [m.content for m in msgs] == ["summary-only"]


def test_clear_removes_jsonl_and_index_row(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    storage.append("s", HumanMessage(content="hello"))
    jsonl = storage.session_jsonl_path("s")
    assert jsonl.exists()

    storage.clear("s")
    assert not jsonl.exists()
    assert storage.list_sessions() == []
    assert storage.session_count() == 0


def test_session_id_validation_rejects_path_traversal(tmp_path: Path) -> None:
    """Constructed JSONL path can't escape the project bucket."""
    import pytest

    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    with pytest.raises(ValueError):
        storage.append("../../etc/passwd", HumanMessage(content="x"))
    with pytest.raises(ValueError):
        storage.append("a/b", HumanMessage(content="x"))
    with pytest.raises(ValueError):
        storage.append("", HumanMessage(content="x"))


def test_in_memory_mode_keeps_messages_in_process(tmp_path: Path) -> None:
    storage = SessionStorage(Path(":memory:"), cwd=tmp_path)
    storage.save("s", [HumanMessage(content="x"), ToolMessage(
        content="r", tool_call_id="t1",
    )])
    msgs = storage.load("s")
    assert len(msgs) == 2
    metas = storage.list_sessions()
    assert any(m.session_id == "s" for m in metas)
    # Nothing on disk under tmp_path — in-memory mode never touches it.
    assert not (tmp_path / "projects").exists()
    assert not (tmp_path / "sessions").exists()


def test_index_handles_legacy_created_at_schema(tmp_path: Path) -> None:
    """Regression: NOT NULL constraint failed: sessions.created_at.

    A pre-existing top-level index.sqlite from an older Aura version
    defined ``sessions(session_id, created_at NOT NULL, last_used_at,
    message_count, first_user_prompt)``. The new code's
    ``CREATE TABLE IF NOT EXISTS`` was a no-op on the existing schema,
    then INSERT without ``created_at`` raised ``IntegrityError`` on
    the very first append.

    Fix: include ``created_at`` in the new schema with a sane DEFAULT,
    and INSERT explicit ``datetime('now')`` so the column is populated
    even when the table was created by a future code version that did
    not list a default.
    """
    legacy_idx = tmp_path / "index.sqlite"
    legacy_conn = sqlite3.connect(str(legacy_idx))
    try:
        legacy_conn.executescript(
            """
            CREATE TABLE sessions (
                session_id        TEXT PRIMARY KEY,
                created_at        TEXT NOT NULL,
                last_used_at      TEXT NOT NULL,
                message_count     INTEGER NOT NULL DEFAULT 0,
                first_user_prompt TEXT NOT NULL DEFAULT ''
            );
            """
        )
        legacy_conn.commit()
    finally:
        legacy_conn.close()

    storage = SessionStorage(tmp_path / "sessions.db", cwd=tmp_path)
    try:
        storage.append("regression-session", HumanMessage(content="hello"))
        sessions = storage.list_sessions(limit=10)
        assert any(s.session_id == "regression-session" for s in sessions)
    finally:
        storage.close()


# ---- v3 layout migration + new-API tests ----------------------------------


def test_migration_from_v2_legacy_to_v3_renames_old_dirs(
    tmp_path: Path,
) -> None:
    """The flat v2 ``sessions/`` and ``subagents/`` dirs are renamed to backups."""
    # Simulate a v2 install: flat sessions/, flat subagents/, nested
    # sessions/index.sqlite.
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "old-session.jsonl").write_text(
        json.dumps({
            "ts": "2025-01-01T00:00:00",
            "payload": {"type": "human", "data": {"content": "v2 hi"}},
        }) + "\n",
        encoding="utf-8",
    )
    legacy_idx = sessions_dir / "index.sqlite"
    conn = sqlite3.connect(str(legacy_idx))
    conn.executescript(
        "CREATE TABLE sessions("
        "session_id TEXT PRIMARY KEY, "
        "created_at TEXT NOT NULL DEFAULT (datetime('now')), "
        "last_used_at TEXT NOT NULL DEFAULT (datetime('now')), "
        "message_count INTEGER NOT NULL DEFAULT 0, "
        "first_user_prompt TEXT NOT NULL DEFAULT '');"
    )
    conn.execute(
        "INSERT INTO sessions(session_id, message_count, first_user_prompt) "
        "VALUES (?, ?, ?)",
        ("old-session", 1, "v2 hi"),
    )
    conn.commit()
    conn.close()

    sub_dir = tmp_path / "subagents"
    sub_dir.mkdir()
    (sub_dir / "subagent-old-task.jsonl").write_text(
        json.dumps({"type": "human", "data": {"content": "child"}}) + "\n",
        encoding="utf-8",
    )

    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    try:
        # Old flat dirs were renamed.
        assert not sessions_dir.exists()
        assert not sub_dir.exists()
        backup_sessions = list(tmp_path.glob("sessions.legacy-*"))
        backup_subagents = list(tmp_path.glob("subagents.legacy-*"))
        assert backup_sessions, "expected sessions.legacy-<ts>"
        assert backup_subagents, "expected subagents.legacy-<ts>"
        # Index rows were folded into the new top-level index.
        new_index = tmp_path / "index.sqlite"
        assert new_index.exists()
        idx = sqlite3.connect(str(new_index))
        try:
            row = idx.execute(
                "SELECT message_count, first_user_prompt FROM sessions "
                "WHERE session_id = ?",
                ("old-session",),
            ).fetchone()
        finally:
            idx.close()
        assert row is not None
        assert row[0] == 1
        assert row[1] == "v2 hi"
        # list_sessions surfaces the migrated row.
        ids = {s.session_id for s in storage.list_sessions(limit=10)}
        assert "old-session" in ids
    finally:
        storage.close()


def test_legacy_session_still_loadable_after_migration(tmp_path: Path) -> None:
    """A v2 session that lives only in the backup dir is still loadable."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "rescue.jsonl").write_text(
        json.dumps({
            "ts": "2025-01-01T00:00:00",
            "payload": {
                "type": "human",
                "data": {"content": "rescue me"},
            },
        }) + "\n"
        + json.dumps({
            "ts": "2025-01-01T00:00:01",
            "payload": {
                "type": "ai",
                "data": {"content": "ok"},
            },
        }) + "\n",
        encoding="utf-8",
    )

    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    try:
        msgs = storage.load("rescue")
        assert [m.content for m in msgs] == ["rescue me", "ok"]
    finally:
        storage.close()


def test_encode_cwd_handles_special_chars(tmp_path: Path) -> None:
    """The cwd encoder mirrors claude-code: each ``/`` → ``-``."""
    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    encoded = storage._encode_cwd(Path("/Users/foo/proj"))
    assert encoded == "-Users-foo-proj"
    # Trailing slash + multi-level depths.
    encoded = storage._encode_cwd(Path("/a/b/c/d"))
    assert encoded == "-a-b-c-d"


def test_two_projects_isolated_in_separate_buckets(tmp_path: Path) -> None:
    """Same session_id under two different cwds writes to two buckets."""
    proj_a = tmp_path / "proj-A"
    proj_b = tmp_path / "proj-B"
    proj_a.mkdir()
    proj_b.mkdir()

    storage_a = SessionStorage(tmp_path / "aura.db", cwd=proj_a)
    storage_a.append("session-x", HumanMessage(content="A msg"))
    storage_a.close()

    storage_b = SessionStorage(tmp_path / "aura.db", cwd=proj_b)
    storage_b.append("session-x", HumanMessage(content="B msg"))

    bucket_a = tmp_path / "projects" / storage_b._encode_cwd(proj_a)
    bucket_b = tmp_path / "projects" / storage_b._encode_cwd(proj_b)
    assert bucket_a != bucket_b
    assert (bucket_a / "session-x.jsonl").exists()
    assert (bucket_b / "session-x.jsonl").exists()
    # Each bucket holds only its own append.
    a_msgs = _read_jsonl(bucket_a / "session-x.jsonl")
    b_msgs = _read_jsonl(bucket_b / "session-x.jsonl")
    assert a_msgs[0]["payload"]["data"]["content"] == "A msg"  # type: ignore[index]
    assert b_msgs[0]["payload"]["data"]["content"] == "B msg"  # type: ignore[index]
    # The shared top-level index lists BOTH (last-write-wins on
    # session_id collision is acceptable — the bucket attribution
    # lives on disk where it's recoverable).
    metas = storage_b.list_sessions(limit=10)
    assert any(m.session_id == "session-x" for m in metas)
    storage_b.close()


def test_subagent_path_nested_under_session(tmp_path: Path) -> None:
    """``subagent_transcript_path(parent_session_id=...)`` nests correctly."""
    storage = SessionStorage(tmp_path / "aura.db", cwd=tmp_path)
    p = storage.subagent_transcript_path(
        "task-XYZ", parent_session_id="parent-session",
    )
    assert p.name == "agent-task-XYZ.jsonl"
    assert p.parent.name == "subagents"
    assert p.parent.parent.name == "parent-session"
    assert p.parent.parent.parent == storage._project_dir()

    meta = storage.subagent_metadata_path(
        "task-XYZ", parent_session_id="parent-session",
    )
    assert meta.parent == p.parent
    assert meta.name == "agent-task-XYZ.meta.json"

    # Without parent_session_id, the legacy flat path is used.
    flat = storage.subagent_transcript_path("orphan-task")
    assert flat.parent == tmp_path / "subagents"
    assert flat.name == "agent-orphan-task.jsonl"

    # write_subagent_transcript writes to the resolved path.
    written = storage.write_subagent_transcript(
        "task-XYZ", [HumanMessage(content="hi child")],
        parent_session_id="parent-session",
    )
    assert written == p
    assert written.exists()
    loaded = storage.load_subagent_transcript("task-XYZ")
    assert [m.content for m in loaded] == ["hi child"]
