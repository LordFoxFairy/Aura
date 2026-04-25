"""Abort infrastructure regression tests (Phase v0.15 Track A).

Covers F-01-001 (history balance on cancel mid-batch), F-01-002 (signal
plumbed to tools), F-05-003 (partial assistant text preserved on cancel),
F-05-004 (user-turn rollback when no AIMessage landed), F-07-005
(parent abort cascades to subagents).

No mocks for the loop — real asyncio so timing bugs surface.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from aura.config.schema import AuraConfig
from aura.core.abort import (
    AbortController,
    AbortException,
    current_abort_signal,
)
from aura.core.agent import Agent
from aura.core.hooks import HookChain
from aura.core.loop import AgentLoop
from aura.core.persistence.storage import SessionStorage
from aura.core.registry import ToolRegistry
from aura.schemas.events import AgentEvent, AssistantDelta, Final
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel, FakeTurn, make_minimal_context


def _make_config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


@pytest.mark.asyncio
async def test_abort_mid_batch_synthesizes_tool_messages() -> None:
    # Three concurrency-safe tools in one batch; one sleeps 5s, the other
    # two sleep 1s. We abort 100ms in. History must end with the AIMessage
    # (3 tool_calls) followed by 3 ToolMessages — at least one synthetic
    # status="error" (abort) — so the next astream call doesn't 400.
    class _P(BaseModel):
        delay: float = 0.0

    started_calls: list[float] = []

    async def _slow(delay: float = 0.0) -> dict[str, Any]:
        started_calls.append(delay)
        await asyncio.sleep(delay)
        return {"slept": delay}

    slow_tool = build_tool(
        name="slow",
        description="sleep then return",
        args_schema=_P,
        coroutine=_slow,
        is_read_only=True,
        is_concurrency_safe=True,
    )

    tool_calls = [
        {"name": "slow", "args": {"delay": 5.0}, "id": "tc_slow"},
        {"name": "slow", "args": {"delay": 1.0}, "id": "tc_mid_a"},
        {"name": "slow", "args": {"delay": 1.0}, "id": "tc_mid_b"},
    ]
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=tool_calls)),
        FakeTurn(message=AIMessage(content="should-not-reach")),
    ])
    registry = ToolRegistry([slow_tool])
    loop_obj = AgentLoop(
        model=model, registry=registry, context=make_minimal_context(),
        hooks=HookChain(),
    )
    abort = AbortController()

    history: list[BaseMessage] = [HumanMessage(content="go")]

    async def _drive() -> list[AgentEvent]:
        events: list[AgentEvent] = []
        try:
            async for ev in loop_obj.run_turn(history=history, abort=abort):
                events.append(ev)
        except (AbortException, asyncio.CancelledError):
            pass
        return events

    drive_task = asyncio.create_task(_drive())
    await asyncio.sleep(0.1)
    abort.abort("user_ctrl_c")
    await drive_task

    # AIMessage with 3 tool_calls + exactly 3 ToolMessages, all aligned to
    # the original tool_call ids so the next provider call is balanced.
    ai_idx = next(
        i for i, m in enumerate(history) if isinstance(m, AIMessage)
    )
    tool_msgs = [m for m in history[ai_idx + 1:] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 3
    ids = {m.tool_call_id for m in tool_msgs}
    assert ids == {"tc_slow", "tc_mid_a", "tc_mid_b"}
    error_msgs = [m for m in tool_msgs if m.status == "error"]
    assert error_msgs, "at least the slow tool must be marked aborted"


@pytest.mark.asyncio
async def test_abort_before_any_ai_message_rolls_back_user_turn(
    tmp_path: Path,
) -> None:
    cfg = _make_config()

    async def _slow_model_invoke() -> AIMessage:  # noqa: ARG001
        await asyncio.sleep(2.0)
        return AIMessage(content="never")

    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="never"))])

    # Patch the FakeChatModel's _agenerate to take 2s so we can abort.
    async def _slow_agenerate(*_a: Any, **_kw: Any) -> Any:
        await asyncio.sleep(2.0)
        raise RuntimeError("should have aborted")

    object.__setattr__(model, "_agenerate", _slow_agenerate)

    agent = Agent(
        config=cfg, model=model, storage=SessionStorage(tmp_path / "db"),
    )

    async def _drive() -> list[AgentEvent]:
        events: list[AgentEvent] = []
        try:
            async for ev in agent.astream("the user prompt"):
                events.append(ev)
        except (AbortException, asyncio.CancelledError):
            pass
        return events

    drive_task = asyncio.create_task(_drive())
    await asyncio.sleep(0.05)
    assert agent.current_abort is not None
    agent.current_abort.abort("user_ctrl_c")
    events = await drive_task

    # No AIMessage was appended → user-turn rolled back from history.
    persisted = agent._storage.load(agent.session_id)
    assert all(
        not (isinstance(m, HumanMessage) and m.content == "the user prompt")
        for m in persisted
    ), f"user turn should have been rolled back; persisted={persisted}"

    # And the cancel flow yielded a Final reason="aborted".
    finals = [e for e in events if isinstance(e, Final)]
    assert finals and finals[-1].reason == "aborted"
    await agent.aclose()


@pytest.mark.asyncio
async def test_abort_after_partial_text_preserves_text(tmp_path: Path) -> None:
    cfg = _make_config()
    # Build a model whose ainvoke hangs forever; we'll inject partial text
    # via the test helper, abort, and assert the AssistantDelta event is
    # surfaced before Final.
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="never"))])

    async def _hang(*_a: Any, **_kw: Any) -> Any:
        await asyncio.sleep(5.0)
        raise RuntimeError("should have aborted")

    object.__setattr__(model, "_agenerate", _hang)

    agent = Agent(
        config=cfg, model=model, storage=SessionStorage(tmp_path / "db"),
    )

    events: list[AgentEvent] = []

    async def _drive() -> None:
        try:
            async for ev in agent.astream("hello"):
                events.append(ev)
        except (AbortException, asyncio.CancelledError):
            pass

    drive_task = asyncio.create_task(_drive())
    await asyncio.sleep(0.05)
    # Buffer a partial as if streaming produced it.
    agent.buffer_partial_assistant_text("partial answer fragment")
    assert agent.current_abort is not None
    agent.current_abort.abort("user_ctrl_c")
    await drive_task

    deltas = [e for e in events if isinstance(e, AssistantDelta)]
    finals = [e for e in events if isinstance(e, Final)]
    assert any(d.text == "partial answer fragment" for d in deltas), (
        f"expected partial AssistantDelta in events; got {events}"
    )
    # The partial delta lands BEFORE Final.
    last_delta_idx = max(
        i for i, e in enumerate(events) if isinstance(e, AssistantDelta)
    )
    final_idx = next(
        i for i, e in enumerate(events) if isinstance(e, Final)
    )
    assert last_delta_idx < final_idx
    assert finals and finals[-1].reason == "aborted"
    await agent.aclose()


@pytest.mark.asyncio
async def test_subagent_abort_cascades(tmp_path: Path) -> None:
    # Build a parent Agent with task_create. Each subagent's first model
    # turn is scripted to hang (5s) so the cascade has stuck children
    # to interrupt within 2s.
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["task_create"]},
    })

    parent_calls = [
        {
            "name": "task_create",
            "args": {
                "description": "child-a",
                "prompt": "do a thing",
                "agent_type": "general-purpose",
            },
            "id": "tc_a",
        },
        {
            "name": "task_create",
            "args": {
                "description": "child-b",
                "prompt": "do a thing",
                "agent_type": "general-purpose",
            },
            "id": "tc_b",
        },
    ]
    parent_model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=parent_calls)),
        FakeTurn(message=AIMessage(content="never")),
    ])

    # The parent's second ainvoke must hang so the abort cascade has
    # work to interrupt. Wrap _agenerate so the first call uses the
    # scripted FakeChatModel turn and subsequent calls hang.
    call_counter = {"n": 0}
    orig_agen = parent_model._agenerate

    async def _conditional(messages: list[BaseMessage], **kw: Any) -> Any:
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            return await orig_agen(messages, **kw)
        await asyncio.sleep(10.0)
        raise RuntimeError("should have aborted")

    object.__setattr__(parent_model, "_agenerate", _conditional)

    agent = Agent(
        config=cfg, model=parent_model,
        storage=SessionStorage(tmp_path / "db"),
    )

    # Inject a child model factory whose ainvoke hangs. The child's
    # tools.enabled is the parent set minus ``task_create``/``task_output``;
    # since the parent's only tool is ``task_create``, the child has zero
    # tools, but ainvoke is still called and that's where we hang.
    async def _child_hanging_agen(*_a: Any, **_kw: Any) -> Any:
        await asyncio.sleep(10.0)
        raise RuntimeError("should have aborted")

    def _child_model_factory() -> Any:
        m = FakeChatModel(turns=[
            FakeTurn(message=AIMessage(content="never")),
        ])
        object.__setattr__(m, "_agenerate", _child_hanging_agen)
        return m

    agent._subagent_factory._model_factory = _child_model_factory

    async def _drive() -> None:
        try:
            async for _ev in agent.astream("go"):
                pass
        except (AbortException, asyncio.CancelledError):
            pass

    drive_task = asyncio.create_task(_drive())
    # Wait for the two subagents to register.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if len(agent._running_aborts) >= 2:
            break
        await asyncio.sleep(0.05)
    assert len(agent._running_aborts) >= 2, (
        f"expected 2 child controllers, got {len(agent._running_aborts)}"
    )
    # Snapshot child controllers BEFORE abort so we can assert .aborted
    # after run_task removes them from the registry on cleanup.
    child_controllers = list(agent._running_aborts.values())

    assert agent.current_abort is not None
    agent.current_abort.abort("user_ctrl_c")
    await asyncio.wait_for(drive_task, timeout=3.0)

    for c in child_controllers:
        assert c.aborted, "every child controller must be flipped"

    # Both task records flipped to cancelled within 2s.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        recs = agent._tasks_store.list()
        if all(r.status in {"cancelled", "failed"} for r in recs):
            break
        await asyncio.sleep(0.05)
    recs = agent._tasks_store.list()
    assert recs, "subagent records should exist"
    cancelled_or_failed = [
        r for r in recs if r.status in {"cancelled", "failed"}
    ]
    assert len(cancelled_or_failed) == len(recs), (
        f"expected all children terminal; got {[r.status for r in recs]}"
    )
    await agent.aclose()


@pytest.mark.asyncio
async def test_signal_aborted_breaks_long_running_tool() -> None:
    # A tool that polls ``current_abort_signal.get().aborted`` should
    # return within 200ms when the signal flips — instead of waiting for
    # its per-tool timeout.
    class _P(BaseModel):
        pass

    started_at: list[float] = []
    finished_at: list[float] = []

    async def _polling() -> dict[str, Any]:
        started_at.append(time.monotonic())
        signal = current_abort_signal.get()
        for _ in range(1000):
            if signal is not None and signal.aborted:
                finished_at.append(time.monotonic())
                raise AbortException("aborted by signal")
            await asyncio.sleep(0.01)
        finished_at.append(time.monotonic())
        return {"ok": True}

    polling_tool = build_tool(
        name="polling",
        description="poll abort signal",
        args_schema=_P,
        coroutine=_polling,
        is_read_only=True,
        is_concurrency_safe=False,
    )

    tool_calls = [{"name": "polling", "args": {}, "id": "tc_poll"}]
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=tool_calls)),
        FakeTurn(message=AIMessage(content="done")),
    ])
    registry = ToolRegistry([polling_tool])
    loop_obj = AgentLoop(
        model=model, registry=registry, context=make_minimal_context(),
        hooks=HookChain(),
    )
    abort = AbortController()

    # Set the contextvar so the tool's polling sees the live controller.
    token = current_abort_signal.set(abort)

    history: list[BaseMessage] = [HumanMessage(content="go")]

    async def _drive() -> None:
        try:
            async for _ in loop_obj.run_turn(history=history, abort=abort):
                pass
        except (AbortException, asyncio.CancelledError):
            pass

    try:
        drive_task = asyncio.create_task(_drive())
        await asyncio.sleep(0.05)
        abort.abort("user_ctrl_c")
        t0 = time.monotonic()
        await drive_task
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, f"polling tool should bail within 0.5s, took {elapsed:.2f}s"
    finally:
        current_abort_signal.reset(token)
