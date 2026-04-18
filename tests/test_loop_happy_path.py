"""Tests for aura.core.loop.run_turn — happy path (text-only, no tool dispatch)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from aura.core.events import AgentEvent, AssistantDelta, Final
from aura.core.loop import run_turn
from aura.core.registry import ToolRegistry
from aura.tools.base import AuraTool, ToolResult, build_tool
from tests.conftest import FakeChatModel, FakeTurn, text_chunk  # noqa: E402

# ---------------------------------------------------------------------------
# Test 1: two text chunks → [AssistantDelta, AssistantDelta, Final]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_happy_path_two_text_chunks() -> None:
    turn = FakeTurn(chunks=[text_chunk("hello ", final=False), text_chunk("world", final=True)])
    model = FakeChatModel(turns=[turn])
    history: list[BaseMessage] = []

    events: list[AgentEvent] = []
    async for ev in run_turn(
        user_prompt="hi",
        history=history,
        model=model,
        registry=ToolRegistry(()),
        provider="openai",
    ):
        events.append(ev)

    # Exactly 3 events: delta, delta, final
    assert len(events) == 3
    assert isinstance(events[0], AssistantDelta)
    assert isinstance(events[1], AssistantDelta)
    assert isinstance(events[2], Final)

    # Delta texts in order
    assert events[0].text == "hello "
    assert events[1].text == "world"

    # Final message is concatenated
    assert events[2].message == "hello world"

    # History has 2 messages: Human + AI
    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "hi"
    assert isinstance(history[1], AIMessage)
    assert history[1].content == "hello world"


# ---------------------------------------------------------------------------
# Test 2: empty registry skips bind_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_no_registry_skips_bind_tools() -> None:
    turn = FakeTurn(chunks=[text_chunk("ok", final=True)])
    model = FakeChatModel(turns=[turn])
    history: list[BaseMessage] = []

    async for _ in run_turn(
        user_prompt="hello",
        history=history,
        model=model,
        registry=ToolRegistry(()),
        provider="openai",
    ):
        pass

    # No bind_tools call should have been made
    assert model.seen_bound_tools == []


# ---------------------------------------------------------------------------
# Test 3: non-empty registry calls bind_tools once with correct schema
# ---------------------------------------------------------------------------


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
    turn = FakeTurn(chunks=[text_chunk("done", final=True)])
    model = FakeChatModel(turns=[turn])
    history: list[BaseMessage] = []
    registry = ToolRegistry([_echo_tool])

    async for _ in run_turn(
        user_prompt="echo me",
        history=history,
        model=model,
        registry=registry,
        provider="openai",
    ):
        pass

    # bind_tools called exactly once
    assert len(model.seen_bound_tools) == 1
    bound_schemas = model.seen_bound_tools[0]
    # One tool schema passed
    assert len(bound_schemas) == 1
    assert bound_schemas[0]["function"]["name"] == "echo"


# ---------------------------------------------------------------------------
# Test 4: empty-content chunks are skipped (no AssistantDelta for them)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_empty_content_chunks_are_skipped() -> None:
    turn = FakeTurn(chunks=[text_chunk("", final=False), text_chunk("only", final=True)])
    model = FakeChatModel(turns=[turn])
    history: list[BaseMessage] = []

    events: list[AgentEvent] = []
    async for ev in run_turn(
        user_prompt="test",
        history=history,
        model=model,
        registry=ToolRegistry(()),
        provider="openai",
    ):
        events.append(ev)

    # Only one AssistantDelta (for "only"), not two
    deltas = [e for e in events if isinstance(e, AssistantDelta)]
    assert len(deltas) == 1
    assert deltas[0].text == "only"

    # Final is still emitted
    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1
    assert finals[0].message == "only"
