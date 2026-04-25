"""F-05-002: append-only JSONL session storage tests.

Pins the wire-level invariants ``aura.core.persistence.storage`` exposes:

  - ``append`` writes one JSONL line per call.
  - ``load`` returns messages in the order they landed.
  - Corrupt lines are skipped + journaled, not raised.
  - The SQLite index is updated on every ``append``.
  - ``list_sessions`` reads from the index.
  - Concurrent appends (threadpool) do not interleave bytes mid-line.
  - Migration from the legacy ``messages`` SQLite is one-shot and
    idempotent.
  - A session resume after JSONL writes loads cleanly.
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


def test_append_writes_one_line(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "aura.db")
    storage.append("session-1", HumanMessage(content="hello"))
    storage.append("session-1", AIMessage(content="hi back"))

    jsonl = tmp_path / "sessions" / "session-1.jsonl"
    assert jsonl.exists()
    lines = _read_jsonl(jsonl)
    assert len(lines) == 2
    # Each line carries the wrapped envelope.
    for line in lines:
        assert "ts" in line
        assert "payload" in line
        payload = line["payload"]
        assert isinstance(payload, dict)
        assert "type" in payload


def test_load_returns_messages_in_order(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "aura.db")
    for i in range(5):
        storage.append("s", HumanMessage(content=f"msg-{i}"))
    msgs = storage.load("s")
    assert [m.content for m in msgs] == [f"msg-{i}" for i in range(5)]


def test_load_skips_corrupt_line_and_logs(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "aura.db")
    storage.append("s", HumanMessage(content="good"))

    # Inject a torn-tail line directly into the JSONL.
    jsonl = tmp_path / "sessions" / "s.jsonl"
    with jsonl.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
        f.write('{"payload": {"type": "human", "data": {"content": "after-tear"}}}\n')

    msgs = storage.load("s")
    contents = [m.content for m in msgs]
    assert "good" in contents
    assert "after-tear" in contents
    assert len(msgs) == 2  # the torn line was skipped, not raised


def test_index_sqlite_updated_on_append(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "aura.db")
    storage.append("s", HumanMessage(content="first prompt"))

    index = tmp_path / "sessions" / "index.sqlite"
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
    storage = SessionStorage(tmp_path / "aura.db")
    storage.append("alpha", HumanMessage(content="a"))
    storage.append("beta", HumanMessage(content="b"))
    storage.append("gamma", HumanMessage(content="c"))

    metas = storage.list_sessions()
    ids = {m.session_id for m in metas}
    assert ids == {"alpha", "beta", "gamma"}

    # Most recently touched session sorts first.
    storage.append("alpha", AIMessage(content="follow-up"))
    metas = storage.list_sessions()
    assert metas[0].session_id == "alpha"


def test_concurrent_appends_dont_interleave_lines(tmp_path: Path) -> None:
    """Two threads hammering ``append`` keep one-message-per-line invariant."""
    storage = SessionStorage(tmp_path / "aura.db")
    payloads = [f"thread-{i}-msg" for i in range(40)]

    def _do(content: str) -> None:
        storage.append("s", HumanMessage(content=content))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_do, payloads))

    jsonl = tmp_path / "sessions" / "s.jsonl"
    lines = _read_jsonl(jsonl)
    assert len(lines) == 40
    # Every line decoded cleanly — no interleaved garbage.
    seen = {line["payload"]["data"]["content"] for line in lines}  # type: ignore[index]
    assert seen == set(payloads)


def test_migration_from_legacy_sqlite_format(tmp_path: Path) -> None:
    """A pre-v0.16 SQLite db is drained into JSONL on first construction."""
    legacy = tmp_path / "aura.db"
    # Build the legacy schema by hand.
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

    storage = SessionStorage(legacy)
    msgs = storage.load("legacy-session")
    assert [m.content for m in msgs] == ["legacy hello", "legacy reply"]

    # Original SQLite is preserved as a backup.
    backups = list(tmp_path.glob("aura.db.legacy-*"))
    assert backups, "legacy db should be renamed, not deleted"


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

    storage1 = SessionStorage(legacy)
    msgs1 = storage1.load("s")
    storage1.close()

    # Re-construct on the same root. The legacy db has been moved to a
    # backup — but even if a fresh legacy db had been re-created here,
    # the marker in index.sqlite would short-circuit migration. Confirm
    # that the on-disk JSONL is unchanged after a second construction.
    jsonl = tmp_path / "sessions" / "s.jsonl"
    before_size = jsonl.stat().st_size

    storage2 = SessionStorage(legacy)
    msgs2 = storage2.load("s")
    storage2.close()

    after_size = jsonl.stat().st_size
    assert before_size == after_size
    assert [m.content for m in msgs1] == [m.content for m in msgs2] == ["once"]


def test_session_resume_after_jsonl_write_loads_correctly(
    tmp_path: Path,
) -> None:
    """Persistence-shape change must keep ``Agent``-level resume semantics."""
    storage = SessionStorage(tmp_path / "aura.db")
    storage.append("resume-target", HumanMessage(content="first turn"))
    storage.append("resume-target", AIMessage(content="reply"))
    storage.append("resume-target", HumanMessage(content="second turn"))

    # Simulate process restart: drop reference, re-open.
    storage.close()
    reopened = SessionStorage(tmp_path / "aura.db")
    msgs = reopened.load("resume-target")
    assert [m.content for m in msgs] == [
        "first turn", "reply", "second turn",
    ]
    metas = reopened.list_sessions()
    assert any(m.session_id == "resume-target" for m in metas)


def test_save_with_appended_tail_uses_append_path(tmp_path: Path) -> None:
    """``save`` with same prefix + new tail goes through append, not rewrite."""
    storage = SessionStorage(tmp_path / "aura.db")
    h1 = HumanMessage(content="one")
    h2 = AIMessage(content="two")
    h3 = HumanMessage(content="three")
    storage.save("s", [h1, h2])
    jsonl = tmp_path / "sessions" / "s.jsonl"
    lines_before = _read_jsonl(jsonl)
    assert len(lines_before) == 2

    storage.save("s", [h1, h2, h3])
    lines_after = _read_jsonl(jsonl)
    # Append path: the FIRST two lines must be byte-identical to before.
    assert lines_after[0] == lines_before[0]
    assert lines_after[1] == lines_before[1]
    assert len(lines_after) == 3


def test_save_with_shrunk_history_rewrites_atomically(tmp_path: Path) -> None:
    """``save`` after compact (shrunk history) rewrites the JSONL file."""
    storage = SessionStorage(tmp_path / "aura.db")
    storage.save("s", [
        HumanMessage(content="a"),
        AIMessage(content="b"),
        HumanMessage(content="c"),
    ])
    storage.save("s", [HumanMessage(content="summary-only")])
    msgs = storage.load("s")
    assert [m.content for m in msgs] == ["summary-only"]


def test_clear_removes_jsonl_and_index_row(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "aura.db")
    storage.append("s", HumanMessage(content="hello"))
    jsonl = tmp_path / "sessions" / "s.jsonl"
    assert jsonl.exists()

    storage.clear("s")
    assert not jsonl.exists()
    assert storage.list_sessions() == []
    assert storage.session_count() == 0


def test_session_id_validation_rejects_path_traversal(tmp_path: Path) -> None:
    """Constructed JSONL path can't escape the sessions/ dir."""
    import pytest

    storage = SessionStorage(tmp_path / "aura.db")
    with pytest.raises(ValueError):
        storage.append("../../etc/passwd", HumanMessage(content="x"))
    with pytest.raises(ValueError):
        storage.append("a/b", HumanMessage(content="x"))
    with pytest.raises(ValueError):
        storage.append("", HumanMessage(content="x"))


def test_in_memory_mode_keeps_messages_in_process(tmp_path: Path) -> None:
    storage = SessionStorage(Path(":memory:"))
    storage.save("s", [HumanMessage(content="x"), ToolMessage(
        content="r", tool_call_id="t1",
    )])
    msgs = storage.load("s")
    assert len(msgs) == 2
    metas = storage.list_sessions()
    assert any(m.session_id == "s" for m in metas)
    # Nothing on disk under tmp_path — in-memory mode never touches it.
    assert not (tmp_path / "sessions").exists()
