"""Tests for AgentLoop.run_turn — single tool-call round trip."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.events import AgentEvent, Final, ToolCallCompleted, ToolCallStarted
from aura.core.hooks import HookChain
from aura.core.loop import AgentLoop
from aura.core.persistence.storage import SessionStorage
from aura.core.registry import ToolRegistry
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel, FakeTurn, make_minimal_context


class _EchoParams(BaseModel):
    msg: str


def _echo(msg: str) -> dict[str, Any]:
    return {"echoed": msg}


_echo_tool: BaseTool = build_tool(
    name="echo",
    description="echoes input",
    args_schema=_EchoParams,
    func=_echo,
    is_read_only=True,
    is_concurrency_safe=True,
)


def _make_model_and_registry() -> tuple[FakeChatModel, ToolRegistry]:
    turn1 = FakeTurn(message=AIMessage(
        content="",
        tool_calls=[{"name": "echo", "args": {"msg": "hi"}, "id": "tc_1"}],
    ))
    turn2 = FakeTurn(message=AIMessage(content="done"))
    model = FakeChatModel(turns=[turn1, turn2])
    registry = ToolRegistry([_echo_tool])
    return model, registry


@pytest.mark.asyncio
async def test_run_turn_single_tool_call_roundtrip() -> None:
    model, registry = _make_model_and_registry()
    history: list[BaseMessage] = []

    events: list[AgentEvent] = []
    loop = AgentLoop(
        model=model, registry=registry, context=make_minimal_context(),
        hooks=HookChain(),
    )
    async for ev in loop.run_turn(user_prompt="call echo", history=history):
        events.append(ev)

    key_events = [e for e in events if isinstance(e, (ToolCallStarted, ToolCallCompleted, Final))]
    assert len(key_events) == 3
    assert isinstance(key_events[0], ToolCallStarted)
    assert isinstance(key_events[1], ToolCallCompleted)
    assert isinstance(key_events[2], Final)

    assert len(history) == 4

    tool_msg = history[2]
    assert tool_msg.tool_call_id == "tc_1"  # type: ignore[attr-defined]
    assert tool_msg.status == "success"  # type: ignore[attr-defined]
    assert "echoed" in tool_msg.content

    assert history[3].content == "done"


@pytest.mark.asyncio
async def test_run_turn_two_ainvoke_calls_made() -> None:
    model, registry = _make_model_and_registry()
    history: list[BaseMessage] = []

    loop = AgentLoop(
        model=model, registry=registry, context=make_minimal_context(),
        hooks=HookChain(),
    )
    async for _ in loop.run_turn(user_prompt="call echo", history=history):
        pass

    assert model.ainvoke_calls == 2


@pytest.mark.asyncio
async def test_run_turn_tool_call_event_contents() -> None:
    model, registry = _make_model_and_registry()
    history: list[BaseMessage] = []

    events: list[AgentEvent] = []
    loop = AgentLoop(
        model=model, registry=registry, context=make_minimal_context(),
        hooks=HookChain(),
    )
    async for ev in loop.run_turn(user_prompt="call echo", history=history):
        events.append(ev)

    started = next(e for e in events if isinstance(e, ToolCallStarted))
    completed = next(e for e in events if isinstance(e, ToolCallCompleted))

    assert started.name == "echo"
    assert started.input == {"msg": "hi"}

    assert completed.name == "echo"
    assert completed.output == {"echoed": "hi"}
    assert completed.error is None


@pytest.mark.asyncio
async def test_run_turn_tool_call_output_serialized_as_json() -> None:
    model, registry = _make_model_and_registry()
    history: list[BaseMessage] = []

    loop = AgentLoop(
        model=model, registry=registry, context=make_minimal_context(),
        hooks=HookChain(),
    )
    async for _ in loop.run_turn(user_prompt="call echo", history=history):
        pass

    tool_msg_content = history[2].content
    assert isinstance(tool_msg_content, str)
    parsed = json.loads(tool_msg_content)
    assert parsed == {"echoed": "hi"}


@pytest.mark.asyncio
async def test_parallel_safe_tools_run_concurrently(tmp_path: Path) -> None:
    # Each tool sleeps 0.2s; serial execution would take ~0.4s, parallel
    # should finish in <0.35s. Timing is the only observable signal of
    # asyncio.gather dispatch for concurrency-safe tools.
    class _P(BaseModel):
        pass

    async def _slow_read() -> dict[str, Any]:
        await asyncio.sleep(0.2)
        return {"ok": True}

    slow = build_tool(
        name="slow_safe",
        description="slow read-only",
        args_schema=_P,
        coroutine=_slow_read,
        is_read_only=True,
        is_concurrency_safe=True,
    )

    tool_calls = [
        {"name": "slow_safe", "args": {}, "id": "tc_1"},
        {"name": "slow_safe", "args": {}, "id": "tc_2"},
    ]
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=tool_calls)),
        FakeTurn(message=AIMessage(content="done")),
    ])

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["slow_safe"]},
    })
    agent = Agent(
        config=cfg, model=model, storage=SessionStorage(tmp_path / "db"),
        available_tools={"slow_safe": slow},
    )

    start = time.monotonic()
    async for _ in agent.astream("go"):
        pass
    elapsed = time.monotonic() - start
    agent.close()

    assert elapsed < 0.35, f"expected parallel <0.35s, got {elapsed:.2f}s"
