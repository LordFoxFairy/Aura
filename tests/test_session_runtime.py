"""Phase 1 §5 — :class:`SessionRuntime` lifecycle in isolation.

The whole point of extracting :class:`SessionRuntime` from the Agent
god object is that lifecycle behaviour can be exercised WITHOUT
constructing a full Agent (no LangChain model, no HookChain, no
Context). These tests assert that contract directly: every case here
constructs only :class:`SessionStorage` + :class:`SessionRuntime`.

Tests cover the six lifecycle entry points called out in the Phase 1
plan: init, save, load, resume, clear, close — plus the streaming
buffer + notification queue helpers that ride alongside.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aura.application.runtime.session import SessionRuntime
from aura.domain.permission.rule import Rule
from aura.domain.permission.session import SessionRuleSet
from aura.domain.task import TaskNotification
from aura.infrastructure.persistence.storage import SessionStorage
from aura.schemas.state import ReadCarryover, ReadRecord

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "aura.db", cwd=tmp_path)


# ----------------------------------------------------------------------
# Init
# ----------------------------------------------------------------------


def test_init_minimal_args(storage: SessionStorage) -> None:
    """Bare SessionRuntime — no log dir, no rules, no inherited reads.

    Defaults: empty buffer, empty notification queue, no log path,
    SessionStart not yet fired."""
    rt = SessionRuntime(storage=storage, session_id="s-init")
    assert rt.session_id == "s-init"
    assert rt.storage is storage
    assert rt.session_log_path is None
    assert rt.session_rules is None
    assert rt.session_start_fired is False
    assert rt.partial_assistant_text == ""
    assert rt.pending_notifications == ()
    assert rt.carryover is None


def test_init_with_session_log_dir_creates_path(
    storage: SessionStorage, tmp_path: Path,
) -> None:
    """``session_log_dir`` is mkdir'd and the per-session JSONL path
    is computed as ``<dir>/<session_id>.jsonl``."""
    log_dir = tmp_path / "logs" / "nested"
    rt = SessionRuntime(
        storage=storage, session_id="s-1", session_log_dir=log_dir,
    )
    assert log_dir.is_dir()
    assert rt.session_log_path == log_dir / "s-1.jsonl"


def test_init_holds_session_rules_reference(
    storage: SessionStorage,
) -> None:
    """The rules object is held by reference — :meth:`clear` calls
    ``.clear()`` on the same instance the caller passed in."""
    rules = SessionRuleSet()
    rules.add(Rule(tool="read_file", content=None))
    rt = SessionRuntime(
        storage=storage, session_id="s-r", session_rules=rules,
    )
    assert rt.session_rules is rules
    assert len(rules.rules()) == 1


# ----------------------------------------------------------------------
# Save / load history
# ----------------------------------------------------------------------


def test_save_then_load_roundtrips_history(storage: SessionStorage) -> None:
    """Round-trip a couple of messages through the runtime — proves
    the storage delegation actually persists + reloads correctly
    without going through the Agent layer."""
    rt = SessionRuntime(storage=storage, session_id="s-save")
    history = [
        HumanMessage(content="hello"),
        AIMessage(content="hi back"),
    ]
    rt.save_history(history)
    loaded = rt.load_history()
    assert len(loaded) == 2
    assert loaded[0].content == "hello"
    assert loaded[1].content == "hi back"


def test_load_empty_session_returns_empty_list(
    storage: SessionStorage,
) -> None:
    rt = SessionRuntime(storage=storage, session_id="s-empty")
    assert rt.load_history() == []


# ----------------------------------------------------------------------
# Resume
# ----------------------------------------------------------------------


def test_resume_swaps_session_id_and_returns_message_count(
    storage: SessionStorage,
) -> None:
    """:meth:`resume` flips the live session_id, returns the row count,
    and drops the SessionStart re-arm flag."""
    # Seed two distinct sessions.
    rt = SessionRuntime(storage=storage, session_id="s-a")
    rt.save_history([HumanMessage(content="a-only")])
    rt_b = SessionRuntime(storage=storage, session_id="s-b")
    rt_b.save_history([
        HumanMessage(content="b-1"),
        AIMessage(content="b-2"),
    ])

    # Mark start fired + buffer some text on rt — resume should clear
    # both so the resumed session feels fresh.
    rt.mark_session_start_fired()
    rt.buffer_partial_assistant_text("partial")
    rt.enqueue_task_notification(TaskNotification(
        task_id="t1", status="completed", summary=None, description="x",
    ))

    count = rt.resume("s-b")
    assert count == 2
    assert rt.session_id == "s-b"
    assert rt.session_start_fired is False
    assert rt.partial_assistant_text == ""
    assert rt.pending_notifications == ()
    # And the loaded rows are the b session, not a's leftovers.
    loaded = rt.load_history()
    assert [m.content for m in loaded] == ["b-1", "b-2"]


def test_resume_unknown_session_raises_keyerror(
    storage: SessionStorage,
) -> None:
    rt = SessionRuntime(storage=storage, session_id="s-x")
    with pytest.raises(KeyError):
        rt.resume("does-not-exist")


def test_resume_retargets_session_log_path(
    storage: SessionStorage, tmp_path: Path,
) -> None:
    """When a log dir was wired, resume re-points the JSONL path to
    the resumed session_id so journal scope routes correctly."""
    log_dir = tmp_path / "logs"
    rt = SessionRuntime(
        storage=storage, session_id="s-1", session_log_dir=log_dir,
    )
    other = SessionRuntime(storage=storage, session_id="s-2")
    other.save_history([HumanMessage(content="hi")])

    rt.resume("s-2")
    assert rt.session_log_path == log_dir / "s-2.jsonl"


# ----------------------------------------------------------------------
# Clear
# ----------------------------------------------------------------------


def test_clear_drops_history_buffers_notifications_and_rules(
    storage: SessionStorage,
) -> None:
    """:meth:`clear` is the runtime side of /clear — drops persisted
    history, clears partial buffer + notification queue, re-arms
    SessionStart, and wipes session rules."""
    rules = SessionRuleSet()
    rules.add(Rule(tool="read_file", content=None))
    rt = SessionRuntime(
        storage=storage, session_id="s-c", session_rules=rules,
    )
    rt.save_history([HumanMessage(content="will-be-wiped")])
    rt.mark_session_start_fired()
    rt.buffer_partial_assistant_text("buffered")
    rt.enqueue_task_notification(TaskNotification(
        task_id="t", status="completed", summary=None, description="d",
    ))
    # Sanity — preconditions hold.
    assert len(rt.load_history()) == 1
    assert rt.session_start_fired is True

    rt.clear()

    assert rt.load_history() == []
    assert rt.session_start_fired is False
    assert rt.partial_assistant_text == ""
    assert rt.pending_notifications == ()
    assert rules.rules() == ()  # session rules were cleared in place
    assert rt.carryover is None


def test_clear_without_session_rules_is_no_op_on_rules(
    storage: SessionStorage,
) -> None:
    """No rules wired → clear must still succeed, not crash."""
    rt = SessionRuntime(storage=storage, session_id="s-norules")
    rt.save_history([HumanMessage(content="x")])
    rt.clear()  # should not raise
    assert rt.load_history() == []


# ----------------------------------------------------------------------
# Close
# ----------------------------------------------------------------------


def test_close_storage_is_idempotent(storage: SessionStorage) -> None:
    """:meth:`close_storage` closes the SQLite handle; calling twice
    must not raise (lifecycle is best-effort and may run on every
    teardown path)."""
    rt = SessionRuntime(storage=storage, session_id="s-close")
    rt.close_storage()
    rt.close_storage()  # idempotent — must not raise


# ----------------------------------------------------------------------
# Streaming buffer + notification helpers
# ----------------------------------------------------------------------


def test_buffer_partial_assistant_text_accumulates(
    storage: SessionStorage,
) -> None:
    rt = SessionRuntime(storage=storage, session_id="s-buf")
    rt.buffer_partial_assistant_text("hello ")
    rt.buffer_partial_assistant_text("world")
    assert rt.partial_assistant_text == "hello world"


def test_take_partial_assistant_text_returns_and_clears(
    storage: SessionStorage,
) -> None:
    rt = SessionRuntime(storage=storage, session_id="s-take")
    rt.buffer_partial_assistant_text("flush-me")
    text = rt.take_partial_assistant_text()
    assert text == "flush-me"
    assert rt.partial_assistant_text == ""


def test_drain_task_notifications_returns_oldest_first(
    storage: SessionStorage,
) -> None:
    rt = SessionRuntime(storage=storage, session_id="s-drain")
    a = TaskNotification(
        task_id="a", status="completed", summary=None, description="A",
    )
    b = TaskNotification(
        task_id="b", status="failed", summary="boom", description="B",
    )
    rt.enqueue_task_notification(a)
    rt.enqueue_task_notification(b)
    drained = rt.drain_task_notifications()
    assert drained == [a, b]
    assert rt.pending_notifications == ()


# ----------------------------------------------------------------------
# Read carryover (Workstream G8 + Phase 3 Task 4)
# ----------------------------------------------------------------------


def _carryover_with_one_record(path: Path) -> ReadCarryover:
    return ReadCarryover(
        records={
            path: ReadRecord(
                path=path,
                mtime_at_read=0.0,
                size_at_read=0,
                read_at_turn=1,
            ),
        },
        source_session_id="parent-1",
        generated_at_turn=1,
    )


def test_carryover_held_for_first_context_build(
    storage: SessionStorage, tmp_path: Path,
) -> None:
    """Subagent path: parent's :class:`ReadCarryover` flows in via
    constructor so the FIRST Context build can pick it up."""
    carry = _carryover_with_one_record(tmp_path / "f.py")
    rt = SessionRuntime(
        storage=storage, session_id="s-inh", carryover=carry,
    )
    assert rt.carryover is carry


def test_clear_drops_carryover(
    storage: SessionStorage, tmp_path: Path,
) -> None:
    """:meth:`clear` drops the carryover — fresh session must NOT
    resurrect a long-gone parent's fingerprints."""
    carry = _carryover_with_one_record(tmp_path / "f.py")
    rt = SessionRuntime(
        storage=storage, session_id="s-cinh", carryover=carry,
    )
    rt.clear()
    assert rt.carryover is None
