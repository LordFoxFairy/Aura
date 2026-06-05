"""Tests for aura.core.storage — SessionStorage over stdlib sqlite3."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_to_dict,
)

from aura.infrastructure.persistence import storage_paths
from aura.infrastructure.persistence.storage import (
    SessionStorage,
    _parse_naive,
    _truncate_one_line,
)


def _four_messages() -> list[BaseMessage]:
    return [
        HumanMessage(content="hello"),
        AIMessage(content="hi", tool_calls=[{"name": "t", "args": {}, "id": "tc_1"}]),
        ToolMessage(content="result", tool_call_id="tc_1"),
        SystemMessage(content="you are helpful"),
    ]


def test_save_and_load_roundtrips(tmp_path: Path) -> None:
    with SessionStorage(tmp_path / "aura.db") as store:
        msgs = _four_messages()
        store.save("default", msgs)
        restored = store.load("default")

    assert len(restored) == 4
    assert isinstance(restored[0], HumanMessage)
    assert restored[0].content == "hello"

    assert isinstance(restored[1], AIMessage)
    assert restored[1].tool_calls[0]["id"] == "tc_1"

    assert isinstance(restored[2], ToolMessage)
    assert restored[2].tool_call_id == "tc_1"
    assert restored[2].content == "result"

    assert isinstance(restored[3], SystemMessage)
    assert restored[3].content == "you are helpful"


def test_save_is_full_replace_not_append(tmp_path: Path) -> None:
    with SessionStorage(tmp_path / "aura.db") as store:
        store.save(
            "default",
            [
                HumanMessage(content="a"),
                HumanMessage(content="b"),
                HumanMessage(content="c"),
            ],
        )
        store.save("default", [HumanMessage(content="only")])
        restored = store.load("default")

    assert len(restored) == 1
    assert restored[0].content == "only"


def test_sessions_are_isolated(tmp_path: Path) -> None:
    with SessionStorage(tmp_path / "aura.db") as store:
        store.save("default", [HumanMessage(content="default-msg")])
        store.save("other", [HumanMessage(content="other-msg")])

        default_msgs = store.load("default")
        other_msgs = store.load("other")

    assert len(default_msgs) == 1
    assert default_msgs[0].content == "default-msg"
    assert len(other_msgs) == 1
    assert other_msgs[0].content == "other-msg"


def test_clear_deletes_only_target_session(tmp_path: Path) -> None:
    with SessionStorage(tmp_path / "aura.db") as store:
        store.save("a", [HumanMessage(content="from-a")])
        store.save("b", [HumanMessage(content="from-b")])
        store.clear("a")

        a_msgs = store.load("a")
        b_msgs = store.load("b")

    assert a_msgs == []
    assert len(b_msgs) == 1
    assert b_msgs[0].content == "from-b"


def test_load_empty_session_returns_empty_list(tmp_path: Path) -> None:
    with SessionStorage(tmp_path / "aura.db") as store:
        result = store.load("nonexistent-session")
    assert result == []


def test_parent_dir_auto_created(tmp_path: Path) -> None:
    db_path = tmp_path / "nonexistent_subdir" / "aura.db"
    assert not db_path.parent.exists()
    with SessionStorage(db_path) as store:
        store.save("s", [HumanMessage(content="x")])
    assert db_path.parent.exists()
    assert db_path.exists()


def test_turn_index_is_message_ordinal(tmp_path: Path) -> None:
    db_path = tmp_path / "aura.db"
    with SessionStorage(db_path) as store:
        store.save(
            "s",
            [
                HumanMessage(content="first"),
                AIMessage(content="second"),
                HumanMessage(content="third"),
            ],
        )

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT turn_index FROM messages WHERE session_id = ? ORDER BY turn_index",
        ("s",),
    ).fetchall()
    conn.close()

    assert [r[0] for r in rows] == [0, 1, 2]


def test_save_empty_list(tmp_path: Path) -> None:
    with SessionStorage(tmp_path / "aura.db") as store:
        store.save("empty-session", [])
        result = store.load("empty-session")
    assert result == []


# --- append (disk + in-memory) ------------------------------------------


def test_append_disk_accumulates_then_loads(tmp_path: Path) -> None:
    """Live turn-by-turn append must persist every envelope, not just the last."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        store.append("live", HumanMessage(content="one"))
        store.append("live", AIMessage(content="two"))
        restored = store.load("live")
    assert [m.content for m in restored] == ["one", "two"]


def test_append_in_memory_uses_turn_index_not_disk(tmp_path: Path) -> None:
    """``:memory:`` append must round-trip through sqlite without writing any JSONL."""
    store = SessionStorage(Path(":memory:"), cwd=tmp_path)
    store.append("m", HumanMessage(content="a"))
    store.append("m", HumanMessage(content="b"))
    restored = store.load("m")
    store.close()
    assert [m.content for m in restored] == ["a", "b"]
    assert not (tmp_path / "projects").exists()


def test_append_rejects_traversal_session_id(tmp_path: Path) -> None:
    """A poisoned session_id must be rejected before any path is opened."""
    with (
        SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store,
        pytest.raises(ValueError, match="invalid session_id"),
    ):
        store.append("../escape", HumanMessage(content="x"))


# --- save: prefix-extension fast path vs divergent rewrite --------------


def test_save_prefix_extension_appends_in_place(tmp_path: Path) -> None:
    """Extending history must append the tail, preserving the shared prefix intact."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        store.save("s", [HumanMessage(content="a"), AIMessage(content="b")])
        store.save(
            "s",
            [
                HumanMessage(content="a"),
                AIMessage(content="b"),
                HumanMessage(content="c"),
            ],
        )
        restored = store.load("s")
    assert [m.content for m in restored] == ["a", "b", "c"]


def test_save_divergence_rewrites_atomically(tmp_path: Path) -> None:
    """A divergent history must fully rewrite the JSONL, dropping stale tail lines."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        store.save("s", [HumanMessage(content="a"), AIMessage(content="old")])
        store.save("s", [HumanMessage(content="a"), AIMessage(content="new")])
        restored = store.load("s")
        jsonl = store.session_jsonl_path("s")
    assert [m.content for m in restored] == ["a", "new"]
    assert not jsonl.with_suffix(jsonl.suffix + ".tmp").exists()


def test_save_rollback_preserves_prior_rows_on_serialize_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-transaction failure must roll back, leaving the prior committed rows intact."""
    store = SessionStorage(Path(":memory:"), cwd=tmp_path)
    store.save("s", [HumanMessage(content="seed")])

    def _boom(*_a: object, **_k: object) -> str:
        raise ValueError("serialize boom")

    monkeypatch.setattr(json, "dumps", _boom)
    with pytest.raises(ValueError, match="serialize boom"):
        store.save("s", [HumanMessage(content="replacement")])
    monkeypatch.undo()

    restored = store.load("s")
    store.close()
    assert [m.content for m in restored] == ["seed"]


# --- _read_jsonl_payloads: corrupt / bare / non-dict lines --------------


def test_read_jsonl_tolerates_corrupt_bare_and_nondict_lines(tmp_path: Path) -> None:
    """A partially-corrupt transcript must still yield every recoverable message."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        jsonl = store.session_jsonl_path("mix")
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        wrapped = {
            "ts": "2020",
            "payload": messages_to_dict([HumanMessage(content="wrapped")])[0],
        }
        bare = messages_to_dict([HumanMessage(content="bare-legacy")])[0]
        lines = [
            json.dumps(wrapped),
            "CORRUPT {{{",
            json.dumps(bare),
            "[1, 2, 3]",
            json.dumps({"ts": "x", "payload": "not-a-dict"}),
            "",
        ]
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        restored = store.load("mix")
    assert [m.content for m in restored] == ["wrapped", "bare-legacy"]


# --- clear: jsonl unlink + index delete ---------------------------------


def test_clear_unlinks_jsonl_and_index_entry(tmp_path: Path) -> None:
    """Clearing a disk session must delete its transcript file, not just the rows."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        store.save("s", [HumanMessage(content="x")])
        jsonl = store.session_jsonl_path("s")
        assert jsonl.exists()
        store.clear("s")
        assert not jsonl.exists()
        assert store.list_sessions() == []


def test_clear_in_memory_is_noop_on_disk(tmp_path: Path) -> None:
    """Clearing a ``:memory:`` session must drop rows without touching the filesystem."""
    store = SessionStorage(Path(":memory:"), cwd=tmp_path)
    store.save("s", [HumanMessage(content="x")])
    store.clear("s")
    restored = store.load("s")
    store.close()
    assert restored == []


# --- list_sessions: index path, legacy fallback, limit, ordering --------


def test_list_sessions_disk_index_orders_newest_first_and_truncates(
    tmp_path: Path,
) -> None:
    """The recent-sessions panel must surface the newest session first, prompt previewed."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        store.save("older", [HumanMessage(content="older prompt")])
        store.save("newer", [HumanMessage(content="  newer    prompt  ")])
        metas = store.list_sessions(limit=20)
    by_id = {m.session_id: m for m in metas}
    assert metas[0].session_id == "newer"
    assert by_id["newer"].first_user_prompt == "newer prompt"
    assert by_id["older"].message_count == 1


def test_list_sessions_limit_caps_rows(tmp_path: Path) -> None:
    """A small limit must cap how many sessions the history view loads."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        store.save("a", [HumanMessage(content="first")])
        store.save("b", [HumanMessage(content="second")])
        store.save("c", [HumanMessage(content="third")])
        metas = store.list_sessions(limit=1)
    assert len(metas) == 1


def test_list_sessions_legacy_sqlite_fallback_when_index_empty(tmp_path: Path) -> None:
    """When no index exists (``:memory:``), history must fall back to the messages table."""
    store = SessionStorage(Path(":memory:"), cwd=tmp_path)
    store.save("s1", [HumanMessage(content="hello world")])
    store.save("s2", [HumanMessage(content="second"), AIMessage(content="reply")])
    metas = store.list_sessions(limit=10)
    store.close()
    by_id = {m.session_id: m for m in metas}
    assert by_id["s1"].message_count == 1
    assert by_id["s1"].first_user_prompt == "hello world"
    assert by_id["s2"].message_count == 2


def test_list_sessions_empty_store_returns_empty(tmp_path: Path) -> None:
    """A pristine store must report no sessions, not raise."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        assert store.list_sessions() == []


# --- _first_user_prompt (legacy fallback) edge cases --------------------


def test_first_user_prompt_skips_non_human_and_corrupt_rows(tmp_path: Path) -> None:
    """Preview must skip leading AI turns and unparsable rows to find the human prompt."""
    store = SessionStorage(Path(":memory:"), cwd=tmp_path)
    store.append("s", AIMessage(content="ai-first"))
    store.append("s", HumanMessage(content="the real prompt"))
    store._conn.execute(
        "INSERT INTO messages (session_id, turn_index, payload_json) VALUES (?, ?, ?)",
        ("s", 99, "NOT JSON {{{"),
    )
    store._conn.commit()
    metas = store.list_sessions(limit=5)
    store.close()
    assert metas[0].first_user_prompt == "the real prompt"


def test_first_user_prompt_empty_when_no_human_turn(tmp_path: Path) -> None:
    """An assistant-only session must yield a blank preview, never crash."""
    store = SessionStorage(Path(":memory:"), cwd=tmp_path)
    store.append("ai", AIMessage(content="just ai"))
    metas = store.list_sessions(limit=5)
    store.close()
    assert metas[0].first_user_prompt == ""


def test_first_user_prompt_skips_empty_and_nonstring_content(tmp_path: Path) -> None:
    """Preview must ignore blank or non-string human content and use the first real one."""
    store = SessionStorage(Path(":memory:"), cwd=tmp_path)
    rows = [
        (0, json.dumps({"type": "human", "data": {"content": ""}})),
        (1, json.dumps({"type": "human", "data": {"content": 123}})),
        (2, json.dumps({"type": "human", "data": {"content": "real one"}})),
    ]
    for turn, payload in rows:
        store._conn.execute(
            "INSERT INTO messages (session_id, turn_index, payload_json) "
            "VALUES (?, ?, ?)",
            ("s", turn, payload),
        )
    store._conn.commit()
    metas = store.list_sessions(limit=5)
    store.close()
    assert metas[0].first_user_prompt == "real one"


def test_list_sessions_falls_back_to_now_on_malformed_index_timestamp(
    tmp_path: Path,
) -> None:
    """A corrupt index timestamp must degrade to a usable row, not crash the history view."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        store.save("s", [HumanMessage(content="hi")])
        index_db = storage_paths.index_path(tmp_path)
        conn = sqlite3.connect(str(index_db))
        conn.execute(
            "UPDATE sessions SET last_used_at = 'garbage-ts' WHERE session_id = 's'",
        )
        conn.commit()
        conn.close()
        metas = store.list_sessions(limit=5)
    assert len(metas) == 1
    assert metas[0].session_id == "s"


# --- session_count -------------------------------------------------------


def test_session_count_is_distinct_sessions(tmp_path: Path) -> None:
    """Idempotent re-saves of the same id must not inflate the distinct session count."""
    store = SessionStorage(Path(":memory:"), cwd=tmp_path)
    assert store.session_count() == 0
    store.save("a", [HumanMessage(content="x")])
    store.save("a", [HumanMessage(content="x"), AIMessage(content="y")])
    store.save("b", [HumanMessage(content="z")])
    count = store.session_count()
    store.close()
    assert count == 2


# --- subagent transcripts: write / list / load / dedup ------------------


def test_subagent_write_load_nested_and_flat(tmp_path: Path) -> None:
    """Subagent transcripts must round-trip in both the nested and flat ad-hoc buckets."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        nested = store.write_subagent_transcript(
            "task1",
            [HumanMessage(content="q"), AIMessage(content="a")],
            parent_session_id="sess1",
        )
        flat = store.write_subagent_transcript("task2", [HumanMessage(content="flat")])
        assert nested.name == "agent-task1.jsonl"
        assert flat.parent.name == "subagents"
        assert [m.content for m in store.load_subagent_transcript("task1")] == ["q", "a"]
        assert [m.content for m in store.load_subagent_transcript("task2")] == ["flat"]


def test_subagent_load_missing_returns_empty(tmp_path: Path) -> None:
    """Loading an unknown task must yield an empty list, not raise."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        assert store.load_subagent_transcript("ghost") == []


def test_subagent_in_memory_write_skips_disk(tmp_path: Path) -> None:
    """``:memory:`` subagent writes must return the path without materializing a file."""
    store = SessionStorage(Path(":memory:"), cwd=tmp_path)
    path = store.write_subagent_transcript("t", [HumanMessage(content="x")])
    store.close()
    assert not path.exists()


def test_subagent_list_handles_legacy_prefix_and_corrupt_lines(tmp_path: Path) -> None:
    """Listing must recognize the legacy ``subagent-`` prefix and survive corrupt lines."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        flat = tmp_path / "subagents"
        flat.mkdir(parents=True)
        payload = messages_to_dict([HumanMessage(content="legacy")])[0]
        legacy = flat / "subagent-legacyTask.jsonl"
        legacy.write_text(
            json.dumps(payload) + "\n" + "NOT JSON {{{\n" + "\n",
            encoding="utf-8",
        )
        (flat / "random.txt").write_text("noise", encoding="utf-8")
        metas = store.list_subagent_transcripts()
        loaded = store.load_subagent_transcript("legacyTask")
    task_ids = {m.task_id for m in metas}
    assert task_ids == {"legacyTask"}
    assert [m.content for m in loaded] == ["legacy"]


def test_subagent_list_dedups_by_task_keeping_newest(tmp_path: Path) -> None:
    """A task written in two buckets must appear once, keyed to the newest copy."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        nested = store.write_subagent_transcript(
            "dup",
            [HumanMessage(content="old"), AIMessage(content="older")],
            parent_session_id="sess1",
        )
        flat = store.write_subagent_transcript("dup", [HumanMessage(content="newest")])
        os.utime(nested, (1000.0, 1000.0))
        os.utime(flat, (9000.0, 9000.0))
        metas = [m for m in store.list_subagent_transcripts() if m.task_id == "dup"]
    assert len(metas) == 1
    assert metas[0].path == flat


def test_subagent_walk_skips_stray_files_and_subagentless_sessions(
    tmp_path: Path,
) -> None:
    """The projects walk must ignore stray files and sessions lacking a subagents dir."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        store.write_subagent_transcript(
            "t1",
            [HumanMessage(content="ok")],
            parent_session_id="sess1",
        )
        projects = tmp_path / "projects"
        (projects / "stray.txt").write_text("x", encoding="utf-8")
        project_dir = next(p for p in projects.iterdir() if p.is_dir())
        (project_dir / "loose.txt").write_text("x", encoding="utf-8")
        (project_dir / "sess-no-subagents").mkdir()
        metas = store.list_subagent_transcripts()
        loaded = store.load_subagent_transcript("t1")
    assert {m.task_id for m in metas} == {"t1"}
    assert [m.content for m in loaded] == ["ok"]


def test_subagent_rejects_traversal_task_id(tmp_path: Path) -> None:
    """A poisoned task_id must be rejected before any subagent path is built."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        with pytest.raises(ValueError, match="invalid task_id"):
            store.write_subagent_transcript("../escape", [HumanMessage(content="x")])
        with pytest.raises(ValueError, match="invalid task_id"):
            store.load_subagent_transcript("a/b")


# --- path helpers & team layout -----------------------------------------


def test_path_and_layout_helpers(tmp_path: Path) -> None:
    """Storage layout accessors must compose stable, traversal-free paths under the root."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        assert store.path == tmp_path / "aura.db"
        assert store.memory_dir().name == "memory"
        assert store.session_dir("sess").name == "sess"
        meta = store.subagent_metadata_path("t", parent_session_id="sess1")
        assert meta.name == "agent-t.meta.json"


def test_team_path_helpers_create_parents_and_list(tmp_path: Path) -> None:
    """Team file accessors must eagerly create parents so first writes never fail."""
    with SessionStorage(tmp_path / "aura.db", cwd=tmp_path) as store:
        assert store.list_team_ids() == []
        cfg = store.team_config_path("team-a")
        inbox = store.team_inbox_path("team-a", "alice")
        transcript = store.team_transcript_path("team-a", "bob")
        root = store.team_root("team-b")
        assert cfg.parent.is_dir()
        assert inbox.parent.is_dir()
        assert transcript.parent.is_dir()
        assert root.is_dir()
        assert store.list_team_ids() == ["team-a", "team-b"]


# --- module-level helpers ------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  a   b  ", "a b"),
        ("", ""),
        ("single", "single"),
        ("x" * 79, "x" * 79),
    ],
)
def test_truncate_one_line_collapses_short_text(raw: str, expected: str) -> None:
    """Short previews must collapse whitespace without appending an ellipsis."""
    assert _truncate_one_line(raw) == expected


def test_truncate_one_line_appends_ellipsis_over_limit() -> None:
    """An over-long prompt must be clipped to the cap plus a single ellipsis."""
    out = _truncate_one_line("word " * 40)
    assert len(out) == 80
    assert out.endswith("…")


def test_parse_naive_reads_sqlite_timestamp_format() -> None:
    """Legacy timestamps stored by sqlite must parse back into a naive datetime."""
    dt = _parse_naive("2024-01-02 03:04:05")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (
        2024,
        1,
        2,
        3,
        4,
        5,
    )


def test_parse_naive_rejects_malformed_timestamp() -> None:
    """A malformed timestamp must surface as ValueError, not a silent wrong date."""
    with pytest.raises(ValueError):
        _parse_naive("not-a-timestamp")


