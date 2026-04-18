"""Tests for aura.core.hooks.HookChain."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from pydantic import BaseModel

from aura.core.hooks import HookChain
from aura.core.state import LoopState
from aura.tools.base import AuraTool, ToolResult, build_tool


class _P(BaseModel):
    x: int = 0


async def _noop_call(params: BaseModel) -> ToolResult:
    return ToolResult(ok=True, output={})


_stub_tool: AuraTool = build_tool(
    name="stub",
    description="stub",
    input_model=_P,
    call=_noop_call,
)


@pytest.mark.asyncio
async def test_hookchain_empty_is_noop() -> None:
    chain = HookChain()
    history: list[BaseMessage] = []
    ai_msg = AIMessage(content="hi")
    params = _P()
    result = ToolResult(ok=True, output={})
    state = LoopState()

    await chain.run_pre_model(history=history, state=state)
    await chain.run_post_model(ai_message=ai_msg, history=history, state=state)
    decision = await chain.run_pre_tool(tool=_stub_tool, params=params, state=state)
    final = await chain.run_post_tool(
        tool=_stub_tool, params=params, result=result, state=state
    )

    assert history == []
    assert decision is None
    assert final is result


@pytest.mark.asyncio
async def test_pre_model_sees_history_and_can_mutate() -> None:
    async def inject(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        history.append(SystemMessage(content="injected"))

    chain = HookChain(pre_model=[inject])
    history: list[BaseMessage] = []
    await chain.run_pre_model(history=history, state=LoopState())

    assert len(history) == 1
    assert isinstance(history[0], SystemMessage)
    assert history[0].content == "injected"


@pytest.mark.asyncio
async def test_post_model_sees_ai_message() -> None:
    captured: list[AIMessage] = []

    async def capture(
        *, ai_message: AIMessage, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        captured.append(ai_message)

    chain = HookChain(post_model=[capture])
    ai_msg = AIMessage(content="test")
    await chain.run_post_model(ai_message=ai_msg, history=[], state=LoopState())

    assert len(captured) == 1
    assert captured[0] is ai_msg


@pytest.mark.asyncio
async def test_pre_tool_short_circuits_with_tool_result() -> None:
    denied = ToolResult(ok=False, error="denied")

    async def deny(
        *, tool: AuraTool, params: BaseModel, state: LoopState, **_: object
    ) -> ToolResult | None:
        return denied

    chain = HookChain(pre_tool=[deny])
    result = await chain.run_pre_tool(tool=_stub_tool, params=_P(), state=LoopState())

    assert result is denied


@pytest.mark.asyncio
async def test_pre_tool_first_short_circuit_wins() -> None:
    call_log: list[str] = []
    decision = ToolResult(ok=False, error="first")

    async def first(
        *, tool: AuraTool, params: BaseModel, state: LoopState, **_: object
    ) -> ToolResult | None:
        call_log.append("first")
        return decision

    async def second(
        *, tool: AuraTool, params: BaseModel, state: LoopState, **_: object
    ) -> ToolResult | None:
        call_log.append("second")
        return ToolResult(ok=False, error="second")

    chain = HookChain(pre_tool=[first, second])
    result = await chain.run_pre_tool(tool=_stub_tool, params=_P(), state=LoopState())

    assert result is decision
    assert call_log == ["first"]


@pytest.mark.asyncio
async def test_post_tool_chains_in_order() -> None:
    async def append_a(
        *, tool: AuraTool, params: BaseModel, result: ToolResult, state: LoopState, **_: object
    ) -> ToolResult:
        out = list(result.output) if isinstance(result.output, list) else []
        out.append("a")
        return ToolResult(ok=True, output=out)

    async def append_b(
        *, tool: AuraTool, params: BaseModel, result: ToolResult, state: LoopState, **_: object
    ) -> ToolResult:
        out = list(result.output) if isinstance(result.output, list) else []
        out.append("b")
        return ToolResult(ok=True, output=out)

    chain = HookChain(post_tool=[append_a, append_b])
    final = await chain.run_post_tool(
        tool=_stub_tool, params=_P(), result=ToolResult(ok=True, output=[]), state=LoopState()
    )

    assert final.output == ["a", "b"]


@pytest.mark.asyncio
async def test_multiple_hooks_of_same_type_run_in_registration_order() -> None:
    call_log: list[str] = []

    async def first(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        call_log.append("first")

    async def second(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        call_log.append("second")

    chain = HookChain(pre_model=[first, second])
    await chain.run_pre_model(history=[], state=LoopState())

    assert call_log == ["first", "second"]


@pytest.mark.asyncio
async def test_pre_tool_chain_returns_none_when_all_return_none() -> None:
    async def pass_through(
        *, tool: AuraTool, params: BaseModel, state: LoopState, **_: object
    ) -> ToolResult | None:
        return None

    chain = HookChain(pre_tool=[pass_through, pass_through])
    result = await chain.run_pre_tool(tool=_stub_tool, params=_P(), state=LoopState())

    assert result is None


@pytest.mark.asyncio
async def test_hooks_receive_state_kwarg() -> None:
    received: list[LoopState] = []

    async def capture(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        received.append(state)

    hooks = HookChain(pre_model=[capture])
    s = LoopState()
    await hooks.run_pre_model(history=[], state=s)

    assert received == [s]
