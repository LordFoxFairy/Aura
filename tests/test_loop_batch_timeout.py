"""Tests for B4 — batch wall-clock timeout in AgentLoop._run_batch.

Per-tool ``timeout_sec`` deadlines exist (see test_tool_timeout.py) but the
outer *batch* has no deadline: the slowest concurrency-safe sibling
dominates end-to-end turn latency, and a runaway tool that somehow
escapes its per-tool deadline can stall the whole turn indefinitely.

This module pins the B4 contract:

- Applies only when ``len(batch) > 1`` AND ``batch_timeout_sec > 0``.
  Size-1 batches (bash / bash_background) always pass through untouched —
  those tools own their own SIGTERM/SIGKILL ladder and the outer wait
  would race the cleanup.
- ``AURA_BATCH_TIMEOUT_SEC`` env (float, default 60.0). ``<= 0`` disables.
  ``AgentLoop(batch_timeout_sec=...)`` kwarg overrides the env.
- Timed-out tasks are cancelled, then awaited for clean shutdown, and
  the step's ToolResult is synthesised as
  ``ToolResult(ok=False, error=f"batch timeout after {deadline}s")``.
- Batch ordering is preserved: results follow the exact batch order so
  the ``zip(batch, results, strict=True)`` invariant in ``_run_batch``
  still matches tool_call_ids.
- ``post_tool`` hook chain fires on the synthesised error result too —
  consumers (size-budget, logger) see the same ToolResult shape whether
  a tool completed, errored, or was cancelled.
- Journal emits ``batch_timeout`` ONLY when at least one task was
  cancelled. Fields: ``session``, ``turn``, ``size``, ``timeout_sec``,
  ``cancelled_count``, ``cancelled_tool_call_ids`` (batch-order),
  ``completed_tool_call_ids`` (batch-order).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks import HookChain
from aura.core.loop import AgentLoop
from aura.core.persistence import journal
from aura.core.registry import ToolRegistry
from aura.schemas.events import ToolCallCompleted
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel, FakeTurn, make_minimal_context


class _NoArgs(BaseModel):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slow_tool(name: str, sleep_s: float) -> BaseTool:
    async def _coro() -> dict[str, Any]:
        await asyncio.sleep(sleep_s)
        return {"done": name}

    return build_tool(
        name=name,
        description=f"slow tool {name}",
        args_schema=_NoArgs,
        coroutine=_coro,
        is_concurrency_safe=True,
    )


def _fast_tool(name: str) -> BaseTool:
    async def _coro() -> dict[str, Any]:
        return {"done": name}

    return build_tool(
        name=name,
        description=f"fast tool {name}",
        args_schema=_NoArgs,
        coroutine=_coro,
        is_concurrency_safe=True,
    )


def _make_loop(
    tools: list[BaseTool],
    tool_calls: list[dict[str, Any]],
    *,
    hooks: HookChain | None = None,
    batch_timeout_sec: float | None = None,
    session_id: str = "batch-timeout-session",
) -> tuple[AgentLoop, list[BaseMessage]]:
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=tool_calls)),
        FakeTurn(message=AIMessage(content="done")),
    ])
    registry = ToolRegistry(tools)
    kwargs: dict[str, Any] = {
        "model": model,
        "registry": registry,
        "context": make_minimal_context(),
        "hooks": hooks or HookChain(),
        "session_id": session_id,
    }
    if batch_timeout_sec is not None:
        kwargs["batch_timeout_sec"] = batch_timeout_sec
    loop = AgentLoop(**kwargs)
    return loop, []


def _events(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text().strip().split("\n")
        if line
    ]


@pytest.fixture(autouse=True)
def _reset_journal() -> Any:
    journal.reset()
    yield
    journal.reset()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_timeout_fires_when_slowest_exceeds_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2 concurrent slow safe-tools must BOTH get batch-timeout errors."""
    monkeypatch.setenv("AURA_BATCH_TIMEOUT_SEC", "0.1")
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)

    slow_a = _slow_tool("slow_a", sleep_s=2.0)
    slow_b = _slow_tool("slow_b", sleep_s=2.0)
    tcs = [
        {"name": "slow_a", "args": {}, "id": "tc_a"},
        {"name": "slow_b", "args": {}, "id": "tc_b"},
    ]

    loop, history = _make_loop([slow_a, slow_b], tcs)
    history.append(HumanMessage(content="go"))
    completed: list[ToolCallCompleted] = []
    async for ev in loop.run_turn(history=history):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)

    assert len(completed) == 2
    for ev in completed:
        assert ev.error is not None
        assert "batch timeout after 0.1s" in ev.error

    events = _events(log_path)
    timeouts = [e for e in events if e["event"] == "batch_timeout"]
    assert len(timeouts) == 1
    j = timeouts[0]
    assert j["cancelled_count"] == 2
    assert j["size"] == 2
    assert j["timeout_sec"] == 0.1
    assert j["cancelled_tool_call_ids"] == ["tc_a", "tc_b"]
    assert j["completed_tool_call_ids"] == []


@pytest.mark.asyncio
async def test_batch_timeout_preserves_completed_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast sibling keeps its real result; only the slow one times out."""
    monkeypatch.setenv("AURA_BATCH_TIMEOUT_SEC", "0.1")
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)

    fast = _fast_tool("fast_a")
    slow = _slow_tool("slow_b", sleep_s=2.0)
    tcs = [
        {"name": "fast_a", "args": {}, "id": "tc_fast"},
        {"name": "slow_b", "args": {}, "id": "tc_slow"},
    ]

    loop, history = _make_loop([fast, slow], tcs)
    history.append(HumanMessage(content="go"))
    completed: list[ToolCallCompleted] = []
    async for ev in loop.run_turn(history=history):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)

    assert len(completed) == 2
    # Order preserved: fast first, slow second.
    assert completed[0].error is None
    assert completed[0].output == {"done": "fast_a"}
    assert completed[1].error is not None
    assert "batch timeout" in completed[1].error

    events = _events(log_path)
    timeouts = [e for e in events if e["event"] == "batch_timeout"]
    assert len(timeouts) == 1
    j = timeouts[0]
    assert j["cancelled_count"] == 1
    assert j["cancelled_tool_call_ids"] == ["tc_slow"]
    assert j["completed_tool_call_ids"] == ["tc_fast"]


@pytest.mark.asyncio
async def test_batch_timeout_env_zero_disables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AURA_BATCH_TIMEOUT_SEC=0 → feature off, both tools run to completion."""
    monkeypatch.setenv("AURA_BATCH_TIMEOUT_SEC", "0")
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)

    slow_a = _slow_tool("slow_a", sleep_s=0.05)
    slow_b = _slow_tool("slow_b", sleep_s=0.05)
    tcs = [
        {"name": "slow_a", "args": {}, "id": "tc_a"},
        {"name": "slow_b", "args": {}, "id": "tc_b"},
    ]

    loop, history = _make_loop([slow_a, slow_b], tcs)
    history.append(HumanMessage(content="go"))
    completed: list[ToolCallCompleted] = []
    async for ev in loop.run_turn(history=history):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)

    assert len(completed) == 2
    assert completed[0].error is None
    assert completed[0].output == {"done": "slow_a"}
    assert completed[1].error is None
    assert completed[1].output == {"done": "slow_b"}

    events = _events(log_path)
    assert [e for e in events if e["event"] == "batch_timeout"] == []


@pytest.mark.asyncio
async def test_batch_timeout_kwarg_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kwarg beats env: env=60 lenient, kwarg=0.1 fires."""
    monkeypatch.setenv("AURA_BATCH_TIMEOUT_SEC", "60")
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)

    slow_a = _slow_tool("slow_a", sleep_s=2.0)
    slow_b = _slow_tool("slow_b", sleep_s=2.0)
    tcs = [
        {"name": "slow_a", "args": {}, "id": "tc_a"},
        {"name": "slow_b", "args": {}, "id": "tc_b"},
    ]

    loop, history = _make_loop(
        [slow_a, slow_b], tcs, batch_timeout_sec=0.1,
    )
    history.append(HumanMessage(content="go"))
    completed: list[ToolCallCompleted] = []
    async for ev in loop.run_turn(history=history):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)

    assert len(completed) == 2
    for ev in completed:
        assert ev.error is not None
        assert "batch timeout" in ev.error

    events = _events(log_path)
    timeouts = [e for e in events if e["event"] == "batch_timeout"]
    assert len(timeouts) == 1
    assert timeouts[0]["timeout_sec"] == 0.1


@pytest.mark.asyncio
async def test_batch_timeout_skipped_for_size_1_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Size-1 batch (e.g. bash) must run its own course — no batch timeout."""
    monkeypatch.setenv("AURA_BATCH_TIMEOUT_SEC", "0.1")
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)

    # is_concurrency_safe=False → partitioned into size-1 batch. A per-tool
    # timeout_sec would be the *tool*'s ladder, not the batch's. We set no
    # per-tool timeout_sec here so the tool runs to natural completion (0.3s)
    # despite AURA_BATCH_TIMEOUT_SEC=0.1.
    lone = _slow_tool("lone_unsafe", sleep_s=0.3)
    # Flip concurrency safety off via metadata so partition_batches makes
    # it a size-1 batch.
    assert lone.metadata is not None
    lone.metadata["is_concurrency_safe"] = False
    tcs = [{"name": "lone_unsafe", "args": {}, "id": "tc_lone"}]

    loop, history = _make_loop([lone], tcs)
    history.append(HumanMessage(content="go"))
    completed: list[ToolCallCompleted] = []
    async for ev in loop.run_turn(history=history):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)

    assert len(completed) == 1
    assert completed[0].error is None
    assert completed[0].output == {"done": "lone_unsafe"}

    events = _events(log_path)
    assert [e for e in events if e["event"] == "batch_timeout"] == []


@pytest.mark.asyncio
async def test_batch_order_preserved_after_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 tools [slow, fast, slow] → results [timeout, fast, timeout]."""
    monkeypatch.setenv("AURA_BATCH_TIMEOUT_SEC", "0.1")
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)

    slow_a = _slow_tool("slow_a", sleep_s=2.0)
    fast_b = _fast_tool("fast_b")
    slow_c = _slow_tool("slow_c", sleep_s=2.0)
    tcs = [
        {"name": "slow_a", "args": {}, "id": "tc_a"},
        {"name": "fast_b", "args": {}, "id": "tc_b"},
        {"name": "slow_c", "args": {}, "id": "tc_c"},
    ]

    loop, history = _make_loop([slow_a, fast_b, slow_c], tcs)
    history.append(HumanMessage(content="go"))
    completed: list[ToolCallCompleted] = []
    async for ev in loop.run_turn(history=history):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)

    assert len(completed) == 3
    assert completed[0].error is not None and "batch timeout" in completed[0].error
    assert completed[1].error is None
    assert completed[1].output == {"done": "fast_b"}
    assert completed[2].error is not None and "batch timeout" in completed[2].error

    events = _events(log_path)
    timeouts = [e for e in events if e["event"] == "batch_timeout"]
    assert len(timeouts) == 1
    j = timeouts[0]
    assert j["cancelled_tool_call_ids"] == ["tc_a", "tc_c"]
    assert j["completed_tool_call_ids"] == ["tc_b"]


@pytest.mark.asyncio
async def test_cancelled_task_still_fires_post_tool_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """post_tool hook chain must see the synthesised error ToolResult too."""
    monkeypatch.setenv("AURA_BATCH_TIMEOUT_SEC", "0.1")
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)

    recorded: list[tuple[str, ToolResult]] = []

    async def _recording_post_tool(
        *, tool: BaseTool, args: dict[str, Any], result: ToolResult,
        state: Any, **_: Any,
    ) -> ToolResult:
        recorded.append((tool.name, result))
        return result

    hooks = HookChain(post_tool=[_recording_post_tool])

    fast = _fast_tool("fast_a")
    slow = _slow_tool("slow_b", sleep_s=2.0)
    tcs = [
        {"name": "fast_a", "args": {}, "id": "tc_fast"},
        {"name": "slow_b", "args": {}, "id": "tc_slow"},
    ]

    loop, history = _make_loop([fast, slow], tcs, hooks=hooks)
    history.append(HumanMessage(content="go"))
    async for _ in loop.run_turn(history=history):
        pass

    # Hook fires for BOTH tools — the cancelled one with the synthesised
    # ToolResult(ok=False, error="batch timeout..."), the completed one
    # with its real result.
    assert len(recorded) == 2
    names = {name for name, _ in recorded}
    assert names == {"fast_a", "slow_b"}

    slow_entry = next(r for n, r in recorded if n == "slow_b")
    assert slow_entry.ok is False
    assert slow_entry.error is not None
    assert "batch timeout" in slow_entry.error

    fast_entry = next(r for n, r in recorded if n == "fast_a")
    assert fast_entry.ok is True
    assert fast_entry.output == {"done": "fast_a"}
