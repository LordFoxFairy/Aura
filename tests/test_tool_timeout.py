"""Tests for per-tool ``timeout_sec`` deadline wired through AgentLoop.

Finding A of the tool-pipeline audit: non-bash tools (grep, glob, read_file,
web_fetch, web_search) had NO deadline, so a pathological grep on a 10 GB
repo could freeze the turn indefinitely. ``timeout_sec`` on
``tool_metadata(...)`` wraps ``tool.ainvoke`` with ``asyncio.wait_for``
inside the loop's hook-chain. None = no deadline (used by ``bash`` /
``bash_background`` which own their own internal SIGTERM→SIGKILL ladder).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks import HookChain
from aura.core.loop import AgentLoop
from aura.core.registry import ToolRegistry
from aura.schemas.events import ToolCallCompleted
from aura.tools.base import build_tool
from aura.tools.bash import bash
from aura.tools.bash_background import BashBackground
from aura.tools.glob import glob
from aura.tools.grep import grep
from aura.tools.read_file import read_file
from aura.tools.web_fetch import web_fetch
from aura.tools.web_search import WebSearch
from tests.conftest import FakeChatModel, FakeTurn, make_minimal_context


class _NoArgs(BaseModel):
    pass


def _make_loop_with_tool(tool: BaseTool) -> tuple[AgentLoop, list[BaseMessage]]:
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(
            content="",
            tool_calls=[{"name": tool.name, "args": {}, "id": "tc_1"}],
        )),
        FakeTurn(message=AIMessage(content="done")),
    ])
    registry = ToolRegistry([tool])
    loop = AgentLoop(
        model=model,
        registry=registry,
        context=make_minimal_context(),
        hooks=HookChain(),
    )
    return loop, []


@pytest.mark.asyncio
async def test_slow_tool_with_timeout_raises_tool_error() -> None:
    # A tool that sleeps 2s with timeout_sec=0.1 must fail with a
    # ToolError whose message mentions "timed out" — the loop converts
    # asyncio.TimeoutError into ToolError so the model sees a
    # recoverable failure rather than a raw exception.

    async def _slow() -> dict[str, Any]:
        await asyncio.sleep(2.0)
        return {"never": "reached"}

    slow = build_tool(
        name="slow_tool",
        description="slow",
        args_schema=_NoArgs,
        coroutine=_slow,
    )
    # build_tool doesn't forward timeout_sec, so patch the metadata post hoc.
    # Production tools set this declaratively in tool_metadata(...); the
    # factory stays minimal for tests.
    assert slow.metadata is not None
    slow.metadata["timeout_sec"] = 0.1

    loop, history = _make_loop_with_tool(slow)
    completed: list[ToolCallCompleted] = []
    history.append(HumanMessage(content="go"))
    async for ev in loop.run_turn(history=history):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)

    assert len(completed) == 1
    assert completed[0].error is not None
    assert "timed out" in completed[0].error


@pytest.mark.asyncio
async def test_fast_tool_with_timeout_returns_normally() -> None:
    # Fast tool: well under the 0.1s deadline. Must complete normally
    # with no error set.

    async def _fast() -> dict[str, Any]:
        return {"ok": True}

    fast = build_tool(
        name="fast_tool",
        description="fast",
        args_schema=_NoArgs,
        coroutine=_fast,
    )
    assert fast.metadata is not None
    fast.metadata["timeout_sec"] = 0.1

    loop, history = _make_loop_with_tool(fast)
    completed: list[ToolCallCompleted] = []
    history.append(HumanMessage(content="go"))
    async for ev in loop.run_turn(history=history):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)

    assert len(completed) == 1
    assert completed[0].error is None
    assert completed[0].output == {"ok": True}


@pytest.mark.asyncio
async def test_timeout_sec_none_enforces_no_deadline() -> None:
    # With timeout_sec=None, a tool that sleeps longer than any plausible
    # wait_for cap must still complete — proves the loop does NOT stack
    # a default deadline on top of "no timeout" tools.

    async def _slow_but_ok() -> dict[str, Any]:
        # 0.5s — comfortably longer than any accidental default.
        await asyncio.sleep(0.5)
        return {"ok": True}

    t = build_tool(
        name="no_deadline",
        description="no deadline",
        args_schema=_NoArgs,
        coroutine=_slow_but_ok,
    )
    assert t.metadata is not None
    assert t.metadata.get("timeout_sec") is None

    loop, history = _make_loop_with_tool(t)
    completed: list[ToolCallCompleted] = []
    history.append(HumanMessage(content="go"))
    async for ev in loop.run_turn(history=history):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)

    assert len(completed) == 1
    assert completed[0].error is None
    assert completed[0].output == {"ok": True}


def test_bash_metadata_timeout_sec_is_none() -> None:
    # Regression guard: bash owns its own timeout mechanism (the
    # ``BashParams.timeout`` field + SIGTERM→SIGKILL ladder inside
    # ``_arun``). Stacking an outer asyncio.wait_for deadline on top would
    # double-fire on real timeouts and interfere with the cleanup ladder
    # that kills the subprocess. Leave this explicit so future edits
    # don't accidentally set it.
    assert (bash.metadata or {}).get("timeout_sec") is None


def test_bash_background_metadata_timeout_sec_is_none() -> None:
    # Same rationale: bash_background manages its own lifetime via the
    # TasksStore + its own 3s TERM → KILL ladder inside the detached task.
    # An outer wait_for would cancel the task-spawn call and leak the
    # running subprocess.
    from aura.core.tasks.store import TasksStore

    tool = BashBackground(store=TasksStore(), running_shells={})
    assert (tool.metadata or {}).get("timeout_sec") is None


# --------------------------------------------------------------------------
# Per-tool default assertions — documents the exact numeric floor each
# shipped tool carries, so an accidental regression flips a red test
# rather than sneaks through review.
# --------------------------------------------------------------------------
def test_grep_timeout_default_is_30() -> None:
    assert (grep.metadata or {}).get("timeout_sec") == 30.0


def test_glob_timeout_default_is_10() -> None:
    assert (glob.metadata or {}).get("timeout_sec") == 10.0


def test_read_file_timeout_default_is_10() -> None:
    assert (read_file.metadata or {}).get("timeout_sec") == 10.0


def test_web_fetch_timeout_default_is_30() -> None:
    assert (web_fetch.metadata or {}).get("timeout_sec") == 30.0


def test_web_search_timeout_default_is_30() -> None:
    # WebSearch is a class — instantiate once for metadata check.
    ws = WebSearch()
    assert (ws.metadata or {}).get("timeout_sec") == 30.0
