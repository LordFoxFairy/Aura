"""Integration tests: HookChain wired into run_turn."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from aura.core.events import AgentEvent, ToolCallCompleted
from aura.core.hooks import HookChain
from aura.core.loop import run_turn
from aura.core.registry import ToolRegistry
from aura.tools.base import AuraTool, ToolResult, build_tool
from tests.conftest import FakeChatModel, FakeTurn


class _EchoParams(BaseModel):
    msg: str


_acall_counter = 0


async def _echo_call(params: BaseModel) -> ToolResult:
    global _acall_counter
    _acall_counter += 1
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


def _tool_turn(msg: str = "hi") -> FakeTurn:
    return FakeTurn(message=AIMessage(
        content="",
        tool_calls=[{"name": "echo", "args": {"msg": msg}, "id": "tc_1"}],
    ))


def _final_turn(text: str = "done") -> FakeTurn:
    return FakeTurn(message=AIMessage(content=text))


@pytest.mark.asyncio
async def test_pre_model_fires_before_each_ainvoke() -> None:
    counter = {"n": 0}

    async def count(*, history: list[BaseMessage]) -> None:
        counter["n"] += 1

    model = FakeChatModel(turns=[_tool_turn(), _final_turn()])
    registry = ToolRegistry([_echo_tool])
    hooks = HookChain(pre_model=[count])

    async for _ in run_turn(
        user_prompt="go",
        history=[],
        model=model,
        registry=registry,
        hooks=hooks,
    ):
        pass

    assert counter["n"] == 2


@pytest.mark.asyncio
async def test_post_model_fires_after_each_ainvoke_with_ai_message() -> None:
    seen: list[AIMessage] = []

    async def capture(*, ai_message: AIMessage, history: list[BaseMessage]) -> None:
        seen.append(ai_message)

    model = FakeChatModel(turns=[_tool_turn(), _final_turn()])
    registry = ToolRegistry([_echo_tool])
    hooks = HookChain(post_model=[capture])

    async for _ in run_turn(
        user_prompt="go",
        history=[],
        model=model,
        registry=registry,
        hooks=hooks,
    ):
        pass

    assert len(seen) == 2
    assert seen[0] is not seen[1]


@pytest.mark.asyncio
async def test_pre_tool_fires_before_acall_and_can_deny() -> None:
    global _acall_counter
    _acall_counter = 0
    denied = ToolResult(ok=False, error="denied")

    async def deny(*, tool: AuraTool, params: BaseModel) -> ToolResult | None:
        return denied

    model = FakeChatModel(turns=[_tool_turn(), _final_turn()])
    registry = ToolRegistry([_echo_tool])
    hooks = HookChain(pre_tool=[deny])

    events: list[AgentEvent] = []
    async for ev in run_turn(
        user_prompt="go",
        history=[],
        model=model,
        registry=registry,
        hooks=hooks,
    ):
        events.append(ev)

    assert _acall_counter == 0

    completed = next(e for e in events if isinstance(e, ToolCallCompleted))
    assert completed.error == "denied"


@pytest.mark.asyncio
async def test_post_tool_fires_after_acall_and_can_rewrite_output() -> None:
    async def truncate(*, tool: AuraTool, params: BaseModel, result: ToolResult) -> ToolResult:
        return ToolResult(ok=True, output={"truncated": True})

    model = FakeChatModel(turns=[_tool_turn(), _final_turn()])
    registry = ToolRegistry([_echo_tool])
    hooks = HookChain(post_tool=[truncate])

    history: list[BaseMessage] = []
    async for _ in run_turn(
        user_prompt="go",
        history=history,
        model=model,
        registry=registry,
        hooks=hooks,
    ):
        pass

    import json
    raw = history[2].content
    assert isinstance(raw, str)
    tool_msg_content = json.loads(raw)
    assert tool_msg_content == {"truncated": True}


@pytest.mark.asyncio
async def test_hooks_all_fire_in_order_for_tool_turn() -> None:
    event_log: list[str] = []

    async def pre_model(*, history: list[BaseMessage]) -> None:
        event_log.append("pre_model")

    async def post_model(*, ai_message: AIMessage, history: list[BaseMessage]) -> None:
        event_log.append("post_model")

    async def pre_tool(*, tool: AuraTool, params: BaseModel) -> ToolResult | None:
        event_log.append("pre_tool")
        return None

    async def post_tool(*, tool: AuraTool, params: BaseModel, result: ToolResult) -> ToolResult:
        event_log.append("post_tool")
        return result

    model = FakeChatModel(turns=[_tool_turn(), _final_turn()])
    registry = ToolRegistry([_echo_tool])
    hooks = HookChain(
        pre_model=[pre_model],
        post_model=[post_model],
        pre_tool=[pre_tool],
        post_tool=[post_tool],
    )

    async for _ in run_turn(
        user_prompt="go",
        history=[],
        model=model,
        registry=registry,
        hooks=hooks,
    ):
        pass

    assert event_log[:4] == ["pre_model", "post_model", "pre_tool", "post_tool"]
