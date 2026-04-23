"""task_get — full TaskRecord snapshot tool.

Covers the lifecycle axes the LLM cares about when polling: running with
no final_result, completed with result, failed with error, unknown id
surfaces ToolError, and the include_messages toggle flips the chatty
transcript on. The progress dict is always present (empty until
``run_task`` records activity) — task_list / task_progress tests exercise
the activity path; here we just verify the shape.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aura.core.tasks.store import TasksStore
from aura.schemas.tool import ToolError
from aura.tools.task_get import TaskGet


@pytest.mark.asyncio
async def test_task_get_running_task_has_no_final_result() -> None:
    store = TasksStore()
    rec = store.create(description="scan", prompt="look")
    tool = TaskGet(store=store)
    out = await tool.ainvoke({"task_id": rec.id})
    assert out["task_id"] == rec.id
    assert out["status"] == "running"
    assert out["final_result"] is None
    assert out["error"] is None
    assert out["finished_at"] is None
    assert out["duration_seconds"] is None
    assert out["description"] == "scan"
    assert out["parent_id"] is None
    assert out["progress"]["tool_count"] == 0
    assert out["progress"]["recent_activities"] == []
    # include_messages defaults False — transcript key must be absent.
    assert "messages" not in out


@pytest.mark.asyncio
async def test_task_get_completed_task_has_result_and_duration() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    rec.started_at = 100.0
    store.mark_completed(rec.id, "answer-42")
    rec.finished_at = 105.5  # deterministic
    tool = TaskGet(store=store)
    out = await tool.ainvoke({"task_id": rec.id})
    assert out["status"] == "completed"
    assert out["final_result"] == "answer-42"
    assert out["duration_seconds"] == pytest.approx(5.5)


@pytest.mark.asyncio
async def test_task_get_failed_task_surfaces_error() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.mark_failed(rec.id, "RuntimeError: bad")
    tool = TaskGet(store=store)
    out = await tool.ainvoke({"task_id": rec.id})
    assert out["status"] == "failed"
    assert out["error"] == "RuntimeError: bad"
    assert out["final_result"] is None


@pytest.mark.asyncio
async def test_task_get_unknown_id_raises_tool_error() -> None:
    store = TasksStore()
    tool = TaskGet(store=store)
    with pytest.raises(ToolError, match="unknown task_id"):
        await tool.ainvoke({"task_id": "no-such"})


@pytest.mark.asyncio
async def test_task_get_include_messages_serializes_transcript() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.append_message(rec.id, HumanMessage(content="hi"))
    store.append_message(rec.id, AIMessage(content="bye"))
    tool = TaskGet(store=store)
    out = await tool.ainvoke(
        {"task_id": rec.id, "include_messages": True},
    )
    assert "messages" in out
    assert out["messages"] == [
        {"type": "human", "content": "hi"},
        {"type": "ai", "content": "bye"},
    ]
