"""TaskProgress — rolling child-agent tool activity.

The run_task coroutine drains ``Agent.astream`` and forwards each
``ToolCallStarted`` event into ``TasksStore.record_activity``. These tests
exercise the recording path directly (no real subagent) so the invariants
(bounded ring, tool_count monotonic, last_activity_at stamped) are pinned
without dragging a full chat loop into the fixture.
"""

from __future__ import annotations

from aura.core.tasks.store import TasksStore


def test_progress_updates_on_record_activity() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    assert rec.progress.tool_count == 0
    assert rec.progress.last_activity_at is None
    assert rec.progress.recent_activities == []

    store.record_activity(rec.id, "bash")
    store.record_activity(rec.id, "read_file")

    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.progress.tool_count == 2
    assert refreshed.progress.last_activity_at is not None
    assert refreshed.progress.recent_activities == ["bash", "read_file"]


def test_progress_recent_activities_bounded_to_five() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    # Fire more than the cap — only the last 5 should survive.
    for name in ["a", "b", "c", "d", "e", "f", "g"]:
        store.record_activity(rec.id, name)
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.progress.tool_count == 7  # monotonic count, not bounded
    assert refreshed.progress.recent_activities == ["c", "d", "e", "f", "g"]


def test_progress_record_activity_is_noop_on_unknown_id() -> None:
    # Racing cancel scenario: record_activity is called with an id that
    # was just removed. Must not raise — tests pin the no-op contract.
    store = TasksStore()
    store.record_activity("no-such", "bash")  # should just return
    assert store.get("no-such") is None


def test_progress_surfaces_via_task_get() -> None:
    # End-to-end: activity recorded in the store appears in task_get
    # output so the LLM polling loop can reason about it.
    import asyncio

    from aura.tools.task_get import TaskGet

    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.record_activity(rec.id, "bash")
    store.record_activity(rec.id, "grep")
    tool = TaskGet(store=store)
    out = asyncio.run(tool.ainvoke({"task_id": rec.id}))
    assert out["progress"]["tool_count"] == 2
    assert out["progress"]["recent_activities"] == ["bash", "grep"]
    assert out["progress"]["last_activity_at"] is not None
