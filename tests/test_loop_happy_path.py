"""Tests for aura.core.loop.run_turn — happy path (text-only, no tool dispatch)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from aura.core.events import AgentEvent, AssistantDelta, Final
from aura.core.hooks import HookChain
from aura.core.loop import run_turn
from aura.core.registry import ToolRegistry
from aura.tools.base import AuraTool, ToolResult, build_tool
from tests.conftest import FakeChatModel, FakeTurn


@pytest.mark.asyncio
async def test_run_turn_happy_path_single_message() -> None:
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="hello world"))])
    history: list[BaseMessage] = []

    events: list[AgentEvent] = []
    async for ev in run_turn(
        user_prompt="hi",
        history=history,
        model=model,
        registry=ToolRegistry(()),
        hooks=HookChain(),
    ):
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

    async for _ in run_turn(
        user_prompt="hello",
        history=history,
        model=model,
        registry=ToolRegistry(()),
        hooks=HookChain(),
    ):
        pass

    assert model.seen_bound_tools == []


class _EchoParams(BaseModel):
    msg: str


async def _echo_call(params: BaseModel) -> ToolResult:
    assert isinstance(params, _EchoParams)
    return ToolResult(ok=True, output={"echoed": params.msg})


_echo_tool: AuraTool = build_tool(
    name="echo",
    description="echoes input",
    input_model=_EchoParams,
    call=_echo_call,
    is_read_only=True,
    is_concurrency_safe=True,
)


@pytest.mark.asyncio
async def test_run_turn_with_registry_calls_bind_tools_once() -> None:
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="done"))])
    history: list[BaseMessage] = []
    registry = ToolRegistry([_echo_tool])

    async for _ in run_turn(
        user_prompt="echo me",
        history=history,
        model=model,
        registry=registry,
        hooks=HookChain(),
    ):
        pass

    assert len(model.seen_bound_tools) == 1
    bound_schemas = model.seen_bound_tools[0]
    assert len(bound_schemas) == 1
    assert bound_schemas[0]["function"]["name"] == "echo"


@pytest.mark.asyncio
async def test_run_turn_empty_content_still_emits_final() -> None:
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content=""))])
    history: list[BaseMessage] = []

    events: list[AgentEvent] = []
    async for ev in run_turn(
        user_prompt="test",
        history=history,
        model=model,
        registry=ToolRegistry(()),
        hooks=HookChain(),
    ):
        events.append(ev)

    deltas = [e for e in events if isinstance(e, AssistantDelta)]
    assert len(deltas) == 0

    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1
    assert finals[0].message == ""
