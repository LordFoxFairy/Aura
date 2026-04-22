"""TasksStore — keyed store for subagent TaskRecords.

Phase E (0.5.0) MVP. Append-only semantics for a completed record's fields:
once a terminal state is set (completed/failed/cancelled) + ``finished_at``
is stamped, subsequent mutations must not silently corrupt the record.

Covers the pure store; integration with the dispatch / tool layer lives in
``test_task_tools.py``.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from aura.core.tasks.store import TasksStore
from aura.core.tasks.types import TaskRecord


def test_create_record_and_list() -> None:
    store = TasksStore()
    rec = store.create(description="scan repo", prompt="look for TODOs")
    assert isinstance(rec, TaskRecord)
    assert rec.status == "running"
    assert rec.description == "scan repo"
    assert rec.prompt == "look for TODOs"
    assert rec.parent_id is None
    assert rec.final_result is None
    assert rec.error is None
    assert store.list() == [rec]


def test_append_message_mutates_record() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    msg = HumanMessage(content="hi")
    store.append_message(rec.id, msg)
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.messages == [msg]


def test_mark_completed_transitions_status_and_sets_final_result() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.append_message(rec.id, AIMessage(content="partial"))
    store.mark_completed(rec.id, "final-answer")
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "completed"
    assert r.final_result == "final-answer"
    assert r.finished_at is not None


def test_mark_failed_sets_error_and_status() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.mark_failed(rec.id, "RuntimeError: boom")
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "failed"
    assert r.error == "RuntimeError: boom"
    assert r.finished_at is not None


def test_get_returns_none_for_unknown_id() -> None:
    store = TasksStore()
    assert store.get("no-such-id") is None


def test_list_filters_by_status() -> None:
    store = TasksStore()
    a = store.create(description="a", prompt="pa")
    b = store.create(description="b", prompt="pb")
    c = store.create(description="c", prompt="pc")
    store.mark_completed(a.id, "done")
    store.mark_failed(b.id, "bad")
    running = store.list(status="running")
    assert running == [c]
    completed = store.list(status="completed")
    assert [r.id for r in completed] == [a.id]
    failed = store.list(status="failed")
    assert [r.id for r in failed] == [b.id]


def test_mark_cancelled_sets_status() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.mark_cancelled(rec.id)
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "cancelled"
    assert r.finished_at is not None
