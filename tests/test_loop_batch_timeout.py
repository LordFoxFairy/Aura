"""Tests for B4 — batch wall-clock timeout in AgentLoop._run_batch.

Per-tool ``timeout_sec`` deadlines exist (see test_tool_timeout.py) but the
outer *batch* has no deadline: the slowest concurrency-safe sibling
dominates end-to-end turn latency, and a runaway tool that somehow
escapes its per-tool deadline can stall the whole turn indefinitely.

This module pins the B4 contract:

- Applies whenever ``batch_timeout_sec > 0``, regardless of batch size.
  Phase 1 Task 12 dropped the previous ``len(batch) > 1`` guard so a
  single misbehaving tool that escapes its per-tool deadline cannot
  stall the whole turn forever. Tools that own their own SIGTERM/SIGKILL
  ladder (bash) still run that ladder INSIDE the task — the outer wait
  is a backstop, not a replacement.
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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks import HookChain
from aura.core.loop import AgentLoop
from aura.core.persistence import journal
from aura.core.registry import ToolRegistry
from aura.schemas.events import Final, ToolCallCompleted
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel, FakeTurn, make_minimal_context


class _NoArgs(BaseModel):
    pass


class _RecordingFakeChatModel(FakeChatModel):
    def __init__(self, turns: list[FakeTurn] | None = None, **kwargs: Any) -> None:
        super().__init__(turns=turns, **kwargs)

    @property
    def seen_messages(self) -> list[list[BaseMessage]]:
        return self.__dict__.setdefault("seen_messages", [])  # type: ignore[no-any-return]

    async def _agenerate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any,
    ) -> Any:
        self.seen_messages.append(list(messages))
        return await super()._agenerate(messages, *args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slow_tool(
    name: str, sleep_s: float, *, is_concurrency_safe: bool = True,
) -> BaseTool:
    async def _coro() -> dict[str, Any]:
        await asyncio.sleep(sleep_s)
        return {"done": name}

    return build_tool(
        name=name,
        description=f"slow tool {name}",
        args_schema=_NoArgs,
        coroutine=_coro,
        is_concurrency_safe=is_concurrency_safe,
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
async def test_batch_timeout_appends_balanced_tool_messages_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed fast/slow tools append one ToolMessage per call before Final."""
    monkeypatch.setenv("AURA_BATCH_TIMEOUT_SEC", "60")
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)

    fast = _fast_tool("fast_a")
    slow = _slow_tool("slow_b", sleep_s=2.0)
    tcs = [
        {"name": "fast_a", "args": {}, "id": "tc_fast"},
        {"name": "slow_b", "args": {}, "id": "tc_slow"},
    ]
    model = _RecordingFakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=tcs)),
        FakeTurn(message=AIMessage(content="done")),
    ])
    loop = AgentLoop(
        model=model,
        registry=ToolRegistry([fast, slow]),
        context=make_minimal_context(),
        hooks=HookChain(),
        session_id="batch-timeout-balanced-history",
        batch_timeout_sec=0.1,
    )
    history: list[BaseMessage] = [HumanMessage(content="go")]

    completed: list[ToolCallCompleted] = []
    finals: list[Final] = []
    async for ev in loop.run_turn(history=history):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)
        if isinstance(ev, Final):
            finals.append(ev)

    assert [ev.name for ev in completed] == ["fast_a", "slow_b"]
    assert completed[0].error is None
    assert completed[1].error is not None
    assert "batch timeout" in completed[1].error
    assert [ev.message for ev in finals] == ["done"]

    assert model.ainvoke_calls == 2
    assert len(model.seen_messages) == 2
    second_turn_messages = model.seen_messages[1]
    tool_messages = [
        msg for msg in second_turn_messages if isinstance(msg, ToolMessage)
    ]
    assert [msg.tool_call_id for msg in tool_messages] == ["tc_fast", "tc_slow"]
    assert [msg.status for msg in tool_messages] == ["success", "error"]
    assert "batch timeout" in str(tool_messages[1].content)

    persisted_tool_messages = [
        msg for msg in history if isinstance(msg, ToolMessage)
    ]
    assert [msg.tool_call_id for msg in persisted_tool_messages] == [
        "tc_fast", "tc_slow",
    ]


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
async def test_batch_timeout_fires_for_size_1_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 1 Task 12 — size-1 batches now respect the batch wallclock too.

    Previously the ``len(batch) > 1`` guard let a single misbehaving tool
    that escaped its per-tool deadline (or had none) park the turn
    forever. The bounded wait now fires regardless of batch size; the
    cancelled task gets the same synthesised ``ToolResult`` and the
    journal records a ``batch_timeout`` event with ``size=1``.
    """
    monkeypatch.setenv("AURA_BATCH_TIMEOUT_SEC", "0.1")
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)

    # is_concurrency_safe=False → partitioned into a size-1 batch. No
    # per-tool ``timeout_sec`` so the tool would otherwise run to natural
    # completion (2s) despite the 0.1s batch deadline.
    lone = _slow_tool("lone_unsafe", sleep_s=2.0, is_concurrency_safe=False)
    tcs = [{"name": "lone_unsafe", "args": {}, "id": "tc_lone"}]

    loop, history = _make_loop([lone], tcs)
    history.append(HumanMessage(content="go"))
    completed: list[ToolCallCompleted] = []
    async for ev in loop.run_turn(history=history):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)

    assert len(completed) == 1
    assert completed[0].error is not None
    assert "batch timeout after 0.1s" in completed[0].error

    events = _events(log_path)
    timeouts = [e for e in events if e["event"] == "batch_timeout"]
    assert len(timeouts) == 1
    j = timeouts[0]
    assert j["size"] == 1
    assert j["timeout_sec"] == 0.1
    assert j["cancelled_count"] == 1
    assert j["cancelled_tool_call_ids"] == ["tc_lone"]
    assert j["completed_tool_call_ids"] == []


@pytest.mark.asyncio
async def test_batch_timeout_size_1_disabled_when_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with the size-1 guard removed, a 0/disabled deadline is still
    a true escape hatch — the lone tool runs to completion."""
    monkeypatch.setenv("AURA_BATCH_TIMEOUT_SEC", "0")
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)

    lone = _slow_tool("lone_unsafe", sleep_s=0.05, is_concurrency_safe=False)
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
