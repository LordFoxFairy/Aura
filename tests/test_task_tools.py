"""task_create / task_output — fire-and-forget subagent tools.

These tests exercise the whole dispatch path end-to-end: the tool returns
a task_id immediately, the subagent runs detached as an asyncio.Task in the
same event loop, and task_output reflects progress. FakeChatModel produces a
single Final turn so the subagent completes in O(event-loop-tick).

Cancellation test models the "user hits Ctrl+C mid-subagent" case — the
Agent tracks the asyncio.Task handle so close()/cancel_all can propagate
CancelledError into the child, which the run_task loop turns into
status=cancelled.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from aura.config.schema import AuraConfig
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.run import run_task
from aura.core.tasks.store import TasksStore
from aura.schemas.tool import ToolError
from aura.tools.task_create import TaskCreate
from aura.tools.task_output import TaskOutput
from tests.conftest import FakeChatModel, FakeTurn


def _cfg(enabled: list[str] | None = None) -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {
            "enabled": enabled if enabled is not None else ["task_create", "task_output"]
        },
    })


def _make_factory(tmp_path: Path) -> tuple[TasksStore, SubagentFactory]:
    store = TasksStore()
    # Subagent gets a FakeChatModel with one Final turn.
    sub_model = FakeChatModel(turns=[FakeTurn(AIMessage(content="subagent-final"))])
    factory = SubagentFactory(
        parent_config=_cfg(enabled=[]),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: sub_model,
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    return store, factory


@pytest.mark.asyncio
async def test_task_create_returns_running_id(tmp_path: Path) -> None:
    store, factory = _make_factory(tmp_path)
    tasks: dict[str, asyncio.Task[None]] = {}
    tool = TaskCreate(store=store, factory=factory, running=tasks)
    out = await tool.ainvoke({"description": "scan", "prompt": "find TODOs"})
    assert out["description"] == "scan"
    assert out["status"] == "running"
    assert "task_id" in out
    rec = store.get(out["task_id"])
    assert rec is not None
    assert rec.description == "scan"
    # Let the detached task finish so pytest doesn't warn about pending
    # tasks. The done-callback may have already popped the handle, so keep
    # a local snapshot around if still present.
    for _ in range(10):
        t = tasks.get(out["task_id"])
        if t is None:
            break
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_task_create_spawns_task_that_completes(tmp_path: Path) -> None:
    store, factory = _make_factory(tmp_path)
    tasks: dict[str, asyncio.Task[None]] = {}
    # Snapshot the handle synchronously before the done-callback can pop it.
    snapshot: dict[str, asyncio.Task[None]] = {}

    class _SnapshotDict(dict[str, asyncio.Task[None]]):
        def __setitem__(self, k: str, v: asyncio.Task[None]) -> None:
            super().__setitem__(k, v)
            snapshot[k] = v

    tracked: _SnapshotDict = _SnapshotDict()
    tool = TaskCreate(store=store, factory=factory, running=tracked)
    out = await tool.ainvoke({"description": "d", "prompt": "p"})
    task_id: str = out["task_id"]
    # Wait for the detached task to finish.
    await asyncio.wait_for(snapshot[task_id], timeout=2.0)
    rec = store.get(task_id)
    assert rec is not None
    assert rec.status == "completed"
    assert rec.final_result == "subagent-final"
    assert tasks == {}  # tracked is what the tool uses; quieten unused-var
    del tasks


@pytest.mark.asyncio
async def test_task_output_raises_on_unknown_id(tmp_path: Path) -> None:
    store, factory = _make_factory(tmp_path)
    tool = TaskOutput(store=store)
    with pytest.raises(ToolError, match="unknown task_id"):
        await tool.ainvoke({"task_id": "no-such"})


@pytest.mark.asyncio
async def test_task_output_returns_record_for_running_task(tmp_path: Path) -> None:
    # Manually stage a running record (no detached task) — probe snapshot only.
    store, _ = _make_factory(tmp_path)
    rec = store.create(description="d", prompt="p")
    tool = TaskOutput(store=store)
    out = await tool.ainvoke({"task_id": rec.id})
    assert out["task_id"] == rec.id
    assert out["status"] == "running"
    assert out["final_result"] is None
    assert out["error"] is None


@pytest.mark.asyncio
async def test_task_output_reflects_final_result_after_subagent_finishes(
    tmp_path: Path,
) -> None:
    store, factory = _make_factory(tmp_path)
    snapshot: dict[str, asyncio.Task[None]] = {}

    class _SnapshotDict(dict[str, asyncio.Task[None]]):
        def __setitem__(self, k: str, v: asyncio.Task[None]) -> None:
            super().__setitem__(k, v)
            snapshot[k] = v

    tracked: _SnapshotDict = _SnapshotDict()
    tc = TaskCreate(store=store, factory=factory, running=tracked)
    out = await tc.ainvoke({"description": "d", "prompt": "p"})
    task_id = out["task_id"]
    await asyncio.wait_for(snapshot[task_id], timeout=2.0)
    to = TaskOutput(store=store)
    info = await to.ainvoke({"task_id": task_id})
    assert info["status"] == "completed"
    assert info["final_result"] == "subagent-final"


def test_subagent_factory_strips_mcp_servers_from_child_config() -> None:
    # Even when parent has mcp_servers configured, the subagent spawned
    # from SubagentFactory must see mcp_servers=[] (0.5.0 isolation).
    # MCP state (connections, discovered tools) is per-agent and MUST NOT
    # leak from parent to child — verified at the config boundary since
    # Agent.aconnect() short-circuits on empty mcp_servers.
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "mcp_servers": [{
            "name": "github",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {},
            "transport": "stdio",
        }],
    })
    factory = SubagentFactory(
        parent_config=parent_config,
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(turns=[]),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    child_agent = factory.spawn("sub-prompt")
    assert child_agent._config.mcp_servers == []
    child_agent.close()


@pytest.mark.asyncio
async def test_parent_cancel_cascades_to_subagent(tmp_path: Path) -> None:
    store = TasksStore()

    # Subagent model that hangs forever — lets us race a cancel against it.
    class _HangingFake(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            await asyncio.sleep(10)
            raise RuntimeError("should not get here")

    factory = SubagentFactory(
        parent_config=_cfg(enabled=[]),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: _HangingFake(),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )

    rec = store.create(description="slow", prompt="go")
    bg = asyncio.create_task(run_task(store, factory, rec.id))
    await asyncio.sleep(0.05)
    bg.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bg
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "cancelled"
