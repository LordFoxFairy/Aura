"""task_list — fleet summary with per-status counts.

The tool is read-only and operates over an in-memory store so these
tests stay pure/synchronous at the store level. We exercise: default
'all' filter, per-status filter, counts are always the full fleet,
limit caps the returned rows, and results are newest-first.
"""

from __future__ import annotations

from typing import Any

import pytest

from aura.application.tasks.store import TasksStore
from aura.tools.task_list import TaskList, TaskListParams, _preview


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


def test_task_list_filters_teammates() -> None:
    store = TasksStore()
    store.create("worker", "p", kind="subagent")
    teammate = store.create("teammate: scout", "idle", kind="teammate")
    tool = TaskList(store=store)

    result = tool.invoke({"kind": "teammate"})

    assert [t["id"] for t in result["tasks"]] == [teammate.id]
    assert result["tasks"][0]["kind"] == "teammate"


def test_task_list_rows_include_observed_at() -> None:
    store = TasksStore()
    rec = store.create("done", "p")
    store.mark_completed(rec.id, "ok")
    observed = store.mark_observed(rec.id)
    tool = TaskList(store=store)

    result = tool.invoke({})

    assert result["tasks"][0]["id"] == rec.id
    assert result["tasks"][0]["observed_at"] == observed


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


def test_preview_omits_kind_when_all() -> None:
    """Default preview must read 'task_list: all' — no noisy kind= when unfiltered."""
    assert _preview({}) == "task_list: all"
    assert _preview({"status": "running"}) == "task_list: running"
    assert _preview({"status": "all", "kind": "all"}) == "task_list: all"


def test_preview_appends_kind_when_filtered() -> None:
    """A kind filter must surface in the preview so the user sees the narrowed scope."""
    assert _preview({"kind": "subagent"}) == "task_list: all, kind=subagent"
    assert (
        _preview({"status": "failed", "kind": "shell"})
        == "task_list: failed, kind=shell"
    )


def test_preview_tolerates_extra_unknown_keys() -> None:
    """Preview reads only status/kind — extra dict pollution must not crash the renderer."""
    args: dict[str, Any] = {"status": "running", "kind": "teammate", "junk": 999}
    assert _preview(args) == "task_list: running, kind=teammate"


def test_metadata_args_preview_is_wired() -> None:
    """The tool advertises _preview via aura_metadata so the UI can render scope."""
    preview = TaskList(store=TasksStore()).aura_metadata.args_preview
    assert preview is not None
    assert preview({"kind": "shell"}) == "task_list: all, kind=shell"


def test_status_and_kind_filters_compose() -> None:
    """Combined status+kind narrows on BOTH axes — a shell-but-running row must drop out."""
    store = TasksStore()
    keep = store.create("shell-fail", "p", kind="shell")
    store.create("shell-run", "p", kind="shell")  # running, must be excluded
    other = store.create("sub-fail", "p", kind="subagent")
    store.mark_failed(keep.id, "boom")
    store.mark_failed(other.id, "boom")
    tool = TaskList(store=store)

    result = tool.invoke({"status": "failed", "kind": "shell"})

    assert [t["id"] for t in result["tasks"]] == [keep.id]
    # Counts still span the full fleet regardless of the dual filter.
    assert result["counts"]["failed"] == 2


@pytest.mark.parametrize("limit", [0, -1, 201, 1000])
def test_limit_out_of_range_rejected_by_schema(limit: int) -> None:
    """limit is bounded [1,200]; out-of-range must raise, never silently clamp."""
    with pytest.raises(ValueError):
        TaskListParams(limit=limit)


@pytest.mark.parametrize("limit", [1, 200])
def test_limit_accepts_inclusive_bounds(limit: int) -> None:
    """The ge/le bounds are inclusive — 1 and 200 are valid window sizes."""
    assert TaskListParams(limit=limit).limit == limit


def test_invalid_status_literal_rejected() -> None:
    """An unknown status must be refused at the schema, not leak a bad filter downstream."""
    with pytest.raises(ValueError):
        TaskListParams.model_validate({"status": "pending"})


def test_invalid_kind_literal_rejected() -> None:
    """An unknown kind must be refused at the schema so the store never sees garbage."""
    with pytest.raises(ValueError):
        TaskListParams.model_validate({"kind": "daemon"})


@pytest.mark.asyncio
async def test_limit_one_returns_only_newest() -> None:
    """The minimal window (limit=1) returns exactly the newest row, counts unchanged."""
    store = _seed()
    out = await TaskList(store=store).ainvoke({"limit": 1})
    assert [t["description"] for t in out["tasks"]] == ["d"]
    assert sum(out["counts"].values()) == 4


@pytest.mark.asyncio
async def test_limit_exceeding_fleet_returns_all() -> None:
    """A window wider than the fleet returns every row — no padding, no error."""
    store = _seed()
    out = await TaskList(store=store).ainvoke({"limit": 200})
    assert len(out["tasks"]) == 4


def test_filter_with_no_matches_returns_empty_rows_full_counts() -> None:
    """A filter matching nothing yields zero rows yet preserves the full-fleet counts."""
    store = TasksStore()
    rec = store.create("only-running", "p")
    tool = TaskList(store=store)

    result = tool.invoke({"status": "completed"})

    assert result["tasks"] == []
    assert result["counts"]["running"] == 1
    assert store.get(rec.id) is not None


def test_run_is_idempotent_across_repeated_reads() -> None:
    """Read-only listing must be stable: two consecutive calls return identical payloads."""
    store = _seed()
    tool = TaskList(store=store)

    first = tool.invoke({"status": "all"})
    second = tool.invoke({"status": "all"})

    assert first == second


def test_observed_at_none_for_running_rows() -> None:
    """A still-running task has no observed_at — the row must carry None, not a stamp."""
    store = TasksStore()
    store.create("live", "p")
    tool = TaskList(store=store)

    result = tool.invoke({})

    assert result["tasks"][0]["observed_at"] is None
