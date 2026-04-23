"""task_list — fleet summary with per-status counts.

The tool is read-only and operates over an in-memory store so these
tests stay pure/synchronous at the store level. We exercise: default
'all' filter, per-status filter, counts are always the full fleet,
limit caps the returned rows, and results are newest-first.
"""

from __future__ import annotations

import pytest

from aura.core.tasks.store import TasksStore
from aura.tools.task_list import TaskList


def _seed() -> TasksStore:
    store = TasksStore()
    a = store.create(description="a", prompt="pa")
    b = store.create(description="b", prompt="pb")
    c = store.create(description="c", prompt="pc")
    d = store.create(description="d", prompt="pd")
    # Force deterministic ordering (clock granularity isn't reliable).
    a.started_at = 100.0
    b.started_at = 200.0
    c.started_at = 300.0
    d.started_at = 400.0
    store.mark_completed(a.id, "ok")
    store.mark_failed(b.id, "boom")
    store.mark_cancelled(c.id)
    # d stays running
    return store


@pytest.mark.asyncio
async def test_task_list_all_returns_every_record_newest_first() -> None:
    store = _seed()
    tool = TaskList(store=store)
    out = await tool.ainvoke({})
    ids = [t["id"] for t in out["tasks"]]
    descs = [t["description"] for t in out["tasks"]]
    # Newest-first: d (400) > c (300) > b (200) > a (100).
    assert descs == ["d", "c", "b", "a"]
    assert len(ids) == 4


@pytest.mark.asyncio
async def test_task_list_filters_by_status() -> None:
    store = _seed()
    tool = TaskList(store=store)
    out = await tool.ainvoke({"status": "failed"})
    assert len(out["tasks"]) == 1
    assert out["tasks"][0]["status"] == "failed"
    assert out["tasks"][0]["description"] == "b"


@pytest.mark.asyncio
async def test_task_list_counts_cover_full_fleet_regardless_of_filter() -> None:
    store = _seed()
    tool = TaskList(store=store)
    out = await tool.ainvoke({"status": "running"})
    # Counts are always the full fleet — you asked for running tasks but
    # the counts tell you the full picture.
    assert out["counts"] == {
        "running": 1, "completed": 1, "failed": 1, "cancelled": 1,
    }
    # The filtered slice is only the single running task though.
    assert [t["description"] for t in out["tasks"]] == ["d"]


@pytest.mark.asyncio
async def test_task_list_respects_limit() -> None:
    store = _seed()
    tool = TaskList(store=store)
    out = await tool.ainvoke({"limit": 2})
    descs = [t["description"] for t in out["tasks"]]
    # Two newest: d, c.
    assert descs == ["d", "c"]
    # Counts still show the full fleet — limit is a display cap only.
    assert sum(out["counts"].values()) == 4


@pytest.mark.asyncio
async def test_task_list_empty_store_returns_zero_counts() -> None:
    tool = TaskList(store=TasksStore())
    out = await tool.ainvoke({})
    assert out["tasks"] == []
    assert out["counts"] == {
        "running": 0, "completed": 0, "failed": 0, "cancelled": 0,
    }
