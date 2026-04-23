"""Tests for AgentLoop.run_turn — happy path (text-only, no tool dispatch)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks import HookChain
from aura.core.loop import AgentLoop
from aura.core.registry import ToolRegistry
from aura.schemas.events import AgentEvent, AssistantDelta, Final
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel, FakeTurn, make_minimal_context


@pytest.mark.asyncio
async def test_run_turn_happy_path_single_message() -> None:
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="hello world"))])
    history: list[BaseMessage] = []
    loop = AgentLoop(
        model=model, registry=ToolRegistry(()), context=make_minimal_context(),
        hooks=HookChain(),
    )

    events: list[AgentEvent] = []
    history.append(HumanMessage(content="hi"))
    async for ev in loop.run_turn(history=history):
        events.append(ev)

    assert events == [AssistantDelta(text="hello world"), Final(message="hello world")]
    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "hi"
    assert isinstance(history[1], AIMessage)
    assert history[1].content == "hello world"


@pytest.mark.asyncio
async def test_run_turn_no_registry_skips_bind_tools() -> None:
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="ok"))])
    history: list[BaseMessage] = []
    loop = AgentLoop(
        model=model, registry=ToolRegistry(()), context=make_minimal_context(),
        hooks=HookChain(),
    )

    history.append(HumanMessage(content="hello"))
    async for _ in loop.run_turn(history=history):
        pass

    assert model.seen_bound_tools == []


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


@pytest.mark.asyncio
async def test_run_turn_with_registry_calls_bind_tools_once() -> None:
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="done"))])
    history: list[BaseMessage] = []
    registry = ToolRegistry([_echo_tool])
    loop = AgentLoop(
        model=model, registry=registry, context=make_minimal_context(),
        hooks=HookChain(),
    )

    history.append(HumanMessage(content="echo me"))
    async for _ in loop.run_turn(history=history):
        pass

    assert len(model.seen_bound_tools) == 1
    bound = model.seen_bound_tools[0]
    assert len(bound) == 1
    assert bound[0].name == "echo"


@pytest.mark.asyncio
async def test_run_turn_empty_content_still_emits_final() -> None:
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content=""))])
    history: list[BaseMessage] = []
    loop = AgentLoop(
        model=model, registry=ToolRegistry(()), context=make_minimal_context(),
        hooks=HookChain(),
    )

    events: list[AgentEvent] = []
    history.append(HumanMessage(content="test"))
    async for ev in loop.run_turn(history=history):
        events.append(ev)

    deltas = [e for e in events if isinstance(e, AssistantDelta)]
    assert len(deltas) == 0

    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1
    assert finals[0].message == ""
