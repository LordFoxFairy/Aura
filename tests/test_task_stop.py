"""task_stop — cancels a running subagent.

Covers the happy path (running task cancels, record flips to cancelled),
error paths (unknown id, already-completed task), and the
"we do await the cancellation" invariant — the tool must not return
until the child has unwound, so the next task_get reflects the
terminal state.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from aura.config.schema import AuraConfig
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.run import run_task
from aura.core.tasks.store import TasksStore
from aura.schemas.tool import ToolError
from aura.tools.task_stop import TaskStop
from tests.conftest import FakeChatModel


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


class _HangingFake(FakeChatModel):
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        await asyncio.sleep(10)
        raise RuntimeError("should not reach here")


def _hanging_factory() -> SubagentFactory:
    return SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: _HangingFake(),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )


@pytest.mark.asyncio
async def test_task_stop_cancels_running_task() -> None:
    store = TasksStore()
    factory = _hanging_factory()
    running: dict[str, asyncio.Task[None]] = {}
    rec = store.create(description="slow", prompt="go")
    bg = asyncio.create_task(run_task(store, factory, rec.id))
    running[rec.id] = bg
    # Let the child actually enter _agenerate so the cancel has something
    # to race against.
    await asyncio.sleep(0.05)
    tool = TaskStop(store=store, running=running)
    out = await tool.ainvoke({"task_id": rec.id})
    assert out["task_id"] == rec.id
    assert out["status"] == "cancelled"
    # run_task's CancelledError branch flipped the store.
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "cancelled"
    assert r.finished_at is not None


@pytest.mark.asyncio
async def test_task_stop_raises_on_unknown_id() -> None:
    store = TasksStore()
    tool = TaskStop(store=store, running={})
    with pytest.raises(ToolError, match="unknown task_id"):
        await tool.ainvoke({"task_id": "no-such"})


@pytest.mark.asyncio
async def test_task_stop_raises_on_already_completed_task() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.mark_completed(rec.id, "done")
    tool = TaskStop(store=store, running={})
    with pytest.raises(ToolError, match="already in terminal state"):
        await tool.ainvoke({"task_id": rec.id})


@pytest.mark.asyncio
async def test_task_stop_awaits_cancellation_so_record_is_terminal() -> None:
    # Key invariant: after task_stop returns, the record MUST be in a
    # terminal state. If the tool returned before the child unwound, a
    # follow-up task_get could still see "running" and the LLM would
    # loop forever. We prove it by requiring status != "running"
    # synchronously in the same event-loop tick as the tool return.
    store = TasksStore()
    factory = _hanging_factory()
    running: dict[str, asyncio.Task[None]] = {}
    rec = store.create(description="slow", prompt="go")
    bg = asyncio.create_task(run_task(store, factory, rec.id))
    running[rec.id] = bg
    await asyncio.sleep(0.05)
    tool = TaskStop(store=store, running=running)
    await tool.ainvoke({"task_id": rec.id})
    # No extra await: the next line runs in the same tick as the return.
    r = store.get(rec.id)
    assert r is not None
    assert r.status != "running"
    assert r.status == "cancelled"


@pytest.mark.asyncio
async def test_task_stop_handles_missing_handle_via_direct_mark() -> None:
    # Race: store says running but the handle map has nothing (e.g. the
    # done-callback popped it right before task_stop looked). The tool
    # must still flip the record to cancelled rather than leave it
    # stuck in a half-running state.
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    tool = TaskStop(store=store, running={})  # no handle, still "running"
    out = await tool.ainvoke({"task_id": rec.id})
    assert out["status"] == "cancelled"
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "cancelled"
