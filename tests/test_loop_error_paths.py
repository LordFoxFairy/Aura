"""Tests for AgentLoop defensive error paths in tool dispatch."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.events import AgentEvent, Final, ToolCallCompleted
from aura.core.hooks import HookChain
from aura.core.loop import AgentLoop
from aura.core.registry import ToolRegistry
from aura.tools.base import ToolResult, build_tool
from tests.conftest import FakeChatModel, FakeTurn


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


def _explode(msg: str) -> dict[str, Any]:
    raise RuntimeError("kaboom")


_exploding_tool: BaseTool = build_tool(
    name="exploder",
    description="always raises",
    args_schema=_EchoParams,
    func=_explode,
    is_destructive=True,
)


@pytest.mark.asyncio
async def test_unknown_tool_name_emits_error_tool_message_and_continues() -> None:
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "ghost", "args": {"msg": "x"}, "id": "tc_1"}
        ])),
        FakeTurn(message=AIMessage(content="recovered")),
    ])
    registry = ToolRegistry([_echo_tool])
    loop = AgentLoop(model=model, registry=registry, hooks=HookChain())

    history: list[BaseMessage] = []
    events: list[AgentEvent] = []
    async for ev in loop.run_turn(user_prompt="go", history=history):
        events.append(ev)

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "tc_1"
    assert tool_msgs[0].status == "error"
    assert "unknown tool" in str(tool_msgs[0].content)
    assert "ghost" in str(tool_msgs[0].content)

    completed = [e for e in events if isinstance(e, ToolCallCompleted)]
    assert len(completed) == 1
    assert completed[0].error is not None

    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1
    assert finals[0].message == "recovered"


@pytest.mark.asyncio
async def test_invalid_args_emit_error_tool_message_and_continues() -> None:
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"wrong_field": 1}, "id": "tc_1"}
        ])),
        FakeTurn(message=AIMessage(content="recovered")),
    ])
    registry = ToolRegistry([_echo_tool])
    loop = AgentLoop(model=model, registry=registry, hooks=HookChain())

    history: list[BaseMessage] = []
    events: list[AgentEvent] = []
    async for ev in loop.run_turn(user_prompt="go", history=history):
        events.append(ev)

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "tc_1"
    assert tool_msgs[0].status == "error"
    assert "invalid args" in str(tool_msgs[0].content)


@pytest.mark.asyncio
async def test_invoke_exception_emits_error_tool_message_with_exception_info() -> None:
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "exploder", "args": {"msg": "x"}, "id": "tc_1"}
        ])),
        FakeTurn(message=AIMessage(content="noted")),
    ])
    registry = ToolRegistry([_exploding_tool])
    loop = AgentLoop(model=model, registry=registry, hooks=HookChain())

    history: list[BaseMessage] = []
    events: list[AgentEvent] = []
    async for ev in loop.run_turn(user_prompt="go", history=history):
        events.append(ev)

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "error"
    content = str(tool_msgs[0].content)
    assert "RuntimeError" in content
    assert "kaboom" in content


@pytest.mark.asyncio
async def test_post_tool_sees_exception_result() -> None:
    seen: list[ToolResult] = []

    async def capture(
        *, tool: BaseTool, args: dict[str, Any], result: ToolResult, state: object,
        **_: object,
    ) -> ToolResult:
        seen.append(result)
        return result

    hooks = HookChain(post_tool=[capture])
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "exploder", "args": {"msg": "x"}, "id": "tc_1"}
        ])),
        FakeTurn(message=AIMessage(content="noted")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_exploding_tool]), hooks=hooks,
    )

    async for _ in loop.run_turn(user_prompt="go", history=[]):
        pass

    assert len(seen) == 1
    assert seen[0].ok is False
    assert seen[0].error is not None
    assert "RuntimeError" in seen[0].error


@pytest.mark.asyncio
async def test_pre_tool_not_fired_for_unknown_tool() -> None:
    calls: list[str] = []

    async def record(
        *, tool: BaseTool, args: dict[str, Any], state: object, **_: object
    ) -> None:
        calls.append(tool.name)
        return None

    hooks = HookChain(pre_tool=[record])
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "ghost", "args": {"msg": "x"}, "id": "tc_1"}
        ])),
        FakeTurn(message=AIMessage(content="done")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_echo_tool]), hooks=hooks,
    )

    async for _ in loop.run_turn(user_prompt="go", history=[]):
        pass

    assert calls == []


@pytest.mark.asyncio
async def test_mixed_tool_calls_each_produce_own_tool_message() -> None:
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"msg": "good"}, "id": "tc_1"},
            {"name": "ghost", "args": {"msg": "x"}, "id": "tc_2"},
            {"name": "echo", "args": {"wrong_field": 1}, "id": "tc_3"},
        ])),
        FakeTurn(message=AIMessage(content="done")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_echo_tool]), hooks=HookChain(),
    )

    history: list[BaseMessage] = []
    async for _ in loop.run_turn(user_prompt="go", history=history):
        pass

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 3
    assert [m.tool_call_id for m in tool_msgs] == ["tc_1", "tc_2", "tc_3"]
    assert tool_msgs[0].status == "success"
    assert tool_msgs[1].status == "error"
    assert tool_msgs[2].status == "error"
    assert "unknown tool" in str(tool_msgs[1].content)
    assert "invalid args" in str(tool_msgs[2].content)
