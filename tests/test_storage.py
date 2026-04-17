"""Tests for aura.core.storage — SessionStorage over stdlib sqlite3."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from aura.core.storage import SessionStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _four_messages() -> list[BaseMessage]:
    return [
        HumanMessage(content="hello"),
        AIMessage(content="hi", tool_calls=[{"name": "t", "args": {}, "id": "tc_1"}]),
        ToolMessage(content="result", tool_call_id="tc_1"),
        SystemMessage(content="you are helpful"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrips(tmp_path: Path) -> None:
    """Save 4 messages and load them back; verify order, types, and key fields."""
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
    """Second save replaces, not appends: 3 msgs → 1 msg → load returns exactly 1."""
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
    """Saving to one session_id does not affect another."""
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
    """clear(a) wipes a's rows; b's rows remain intact."""
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
    """Loading a never-saved session returns an empty list, not an error."""
    with SessionStorage(tmp_path / "aura.db") as store:
        result = store.load("nonexistent-session")
    assert result == []


def test_parent_dir_auto_created(tmp_path: Path) -> None:
    """SessionStorage auto-creates the parent directory when it doesn't exist."""
    db_path = tmp_path / "nonexistent_subdir" / "aura.db"
    assert not db_path.parent.exists()
    with SessionStorage(db_path) as store:
        store.save("s", [HumanMessage(content="x")])
    assert db_path.parent.exists()
    assert db_path.exists()


def test_turn_index_is_message_ordinal(tmp_path: Path) -> None:
    """turn_index is per-message ordinal (0, 1, 2), one row per BaseMessage."""
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
    """Saving an empty list does not crash and loads back as empty list."""
    with SessionStorage(tmp_path / "aura.db") as store:
        store.save("empty-session", [])
        result = store.load("empty-session")
    assert result == []
