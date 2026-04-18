"""Tests for aura.core.loop.run_turn — single tool-call round trip (Task 17)."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from aura.core.events import AgentEvent, Final, ToolCallCompleted, ToolCallStarted
from aura.core.loop import run_turn
from aura.core.registry import ToolRegistry
from aura.tools.base import AuraTool, ToolResult, build_tool
from tests.conftest import FakeChatModel, FakeTurn, text_chunk, tool_call_chunk

# ---------------------------------------------------------------------------
# Shared test fixture: _echo_tool built via build_tool factory
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


def _make_model_and_registry() -> tuple[FakeChatModel, ToolRegistry]:
    turn1 = FakeTurn(chunks=[tool_call_chunk("echo", '{"msg": "hi"}', "tc_1")])
    turn2 = FakeTurn(chunks=[text_chunk("done", final=True)])
    model = FakeChatModel(turns=[turn1, turn2])
    registry = ToolRegistry([_echo_tool])
    return model, registry


# ---------------------------------------------------------------------------
# Test 1: full round trip — history shape + event kinds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_single_tool_call_roundtrip() -> None:
    model, registry = _make_model_and_registry()
    history: list[BaseMessage] = []

    events: list[AgentEvent] = []
    async for ev in run_turn(
        user_prompt="call echo",
        history=history,
        model=model,
        registry=registry,
        provider="openai",
    ):
        events.append(ev)

    # Event kinds include ToolCallStarted, ToolCallCompleted, Final in that order
    key_events = [e for e in events if isinstance(e, (ToolCallStarted, ToolCallCompleted, Final))]
    assert len(key_events) == 3
    assert isinstance(key_events[0], ToolCallStarted)
    assert isinstance(key_events[1], ToolCallCompleted)
    assert isinstance(key_events[2], Final)

    # History: [Human, AI(tool_calls=[tc_1]), Tool(tool_call_id=tc_1), AI(final text)]
    assert len(history) == 4

    # history[2] is the ToolMessage
    tool_msg = history[2]
    assert tool_msg.tool_call_id == "tc_1"  # type: ignore[attr-defined]
    assert tool_msg.status == "success"  # type: ignore[attr-defined]
    assert "echoed" in tool_msg.content

    # history[3] is the final AIMessage
    assert history[3].content == "done"


# ---------------------------------------------------------------------------
# Test 2: ToolCallStarted and ToolCallCompleted event contents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_tool_call_event_contents() -> None:
    model, registry = _make_model_and_registry()
    history: list[BaseMessage] = []

    events: list[AgentEvent] = []
    async for ev in run_turn(
        user_prompt="call echo",
        history=history,
        model=model,
        registry=registry,
        provider="openai",
    ):
        events.append(ev)

    started = next(e for e in events if isinstance(e, ToolCallStarted))
    completed = next(e for e in events if isinstance(e, ToolCallCompleted))

    assert started.name == "echo"
    assert started.input == {"msg": "hi"}

    assert completed.name == "echo"
    assert completed.output == {"echoed": "hi"}
    assert completed.error is None


# ---------------------------------------------------------------------------
# Test 3: ToolMessage.content is valid JSON serialization of tool output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_tool_call_output_serialized_as_json() -> None:
    model, registry = _make_model_and_registry()
    history: list[BaseMessage] = []

    async for _ in run_turn(
        user_prompt="call echo",
        history=history,
        model=model,
        registry=registry,
        provider="openai",
    ):
        pass

    tool_msg_content = history[2].content
    assert isinstance(tool_msg_content, str)
    parsed = json.loads(tool_msg_content)
    assert parsed == {"echoed": "hi"}


# ---------------------------------------------------------------------------
# Test 4: model.astream was called exactly twice (one per loop iteration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_two_astream_calls_made() -> None:
    model, registry = _make_model_and_registry()
    history: list[BaseMessage] = []

    async for _ in run_turn(
        user_prompt="call echo",
        history=history,
        model=model,
        registry=registry,
        provider="openai",
    ):
        pass

    assert model.astream_calls == 2
