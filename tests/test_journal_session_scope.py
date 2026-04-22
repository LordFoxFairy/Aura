"""Tests for :func:`journal.session_scope` — per-task routing via contextvars.

The contract:

- Inside the ``with session_scope(p):`` block, every ``journal.write`` (on
  the current task + any child tasks spawned inside it) routes to ``p``.
- Nested scopes stack — innermost wins — and pop cleanly on exit.
- Outside any scope, ``write`` falls back to the globally-configured path
  set via ``configure`` (or silently drops if neither is set).
- Two concurrent asyncio.Tasks, each in their own scope, must NOT
  cross-contaminate. This is the critical invariant that makes
  multiple-Agents-per-process safe.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from aura.core.persistence import journal


@pytest.fixture(autouse=True)
def _reset() -> Any:
    journal.reset()
    yield
    journal.reset()


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().strip().split("\n")
        if line
    ]


def test_session_scope_routes_writes_to_path(tmp_path: Path) -> None:
    target = tmp_path / "a.jsonl"
    with journal.session_scope(target):
        journal.write("scoped_event", value=1)
    events = _events(target)
    assert len(events) == 1
    assert events[0]["event"] == "scoped_event"
    assert events[0]["value"] == 1


def test_session_scope_nested_overrides(tmp_path: Path) -> None:
    outer = tmp_path / "outer.jsonl"
    inner = tmp_path / "inner.jsonl"
    with journal.session_scope(outer):
        journal.write("o1")
        with journal.session_scope(inner):
            journal.write("i1")
            journal.write("i2")
        # Back to outer after inner exits.
        journal.write("o2")

    outer_events = [e["event"] for e in _events(outer)]
    inner_events = [e["event"] for e in _events(inner)]
    assert outer_events == ["o1", "o2"]
    assert inner_events == ["i1", "i2"]


def test_session_scope_falls_back_to_global_on_exit(tmp_path: Path) -> None:
    global_log = tmp_path / "global.jsonl"
    scoped_log = tmp_path / "scoped.jsonl"
    journal.configure(global_log)

    with journal.session_scope(scoped_log):
        journal.write("in_scope")
    journal.write("after_scope")

    assert [e["event"] for e in _events(scoped_log)] == ["in_scope"]
    assert [e["event"] for e in _events(global_log)] == ["after_scope"]


@pytest.mark.asyncio
async def test_session_scope_async_isolation(tmp_path: Path) -> None:
    # Two concurrent tasks, each entering its own session_scope, must not
    # cross-contaminate. contextvars give us the isolation: Task creation
    # copies the current Context, so a ``with session_scope(...)`` entered
    # before spawning a task is inherited; one entered AFTER spawn lives
    # only in that task.
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"

    ready = asyncio.Event()

    async def worker(log: Path, tag: str) -> None:
        with journal.session_scope(log):
            # Yield so the two tasks definitely interleave — if contextvars
            # leaked across tasks we'd pick up the other task's path here.
            ready.set()
            await asyncio.sleep(0)
            journal.write(f"{tag}_first")
            await asyncio.sleep(0)
            journal.write(f"{tag}_second")

    await asyncio.gather(worker(log_a, "A"), worker(log_b, "B"))

    a_events = [e["event"] for e in _events(log_a)]
    b_events = [e["event"] for e in _events(log_b)]
    assert a_events == ["A_first", "A_second"]
    assert b_events == ["B_first", "B_second"]
    # Disjoint — no event from B landed in A's file, and vice versa.
    assert not set(a_events) & set(b_events)


def test_session_scope_without_global_or_scope_is_noop(tmp_path: Path) -> None:
    # No configure(), no session_scope → write() silently drops.
    journal.write("should_be_dropped")
    # Nothing to assert negatively on disk — just guard: no exception.
    # After exiting a scope we're back to no-scope no-global — same rule.
    with journal.session_scope(tmp_path / "transient.jsonl"):
        journal.write("scoped")
    journal.write("still_dropped")
    assert not (tmp_path / "does-not-exist.jsonl").exists()
