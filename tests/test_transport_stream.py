"""Tests for streaming Agent output through transport adapters."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from aura.schemas.events import AssistantDelta, Final, ToolCallStarted
from aura.schemas.state import LoopSlots
from aura.transport.agui import AguiAdapter
from aura.transport.sse import encode_json_sse
from aura.transport.stream import (
    stream_agent_agui,
    stream_agent_agui_sse,
    stream_agent_wire,
    stream_agent_wire_sse,
)


class _FakeAgent:
    current_model = "fake:model"
    mode = "default"
    pinned_tokens_estimate = 0
    context_window = 100
    state = SimpleNamespace(slots=LoopSlots())

    async def astream(self, prompt: str) -> Any:
        assert prompt == "hello"
        yield AssistantDelta("hi")
        yield ToolCallStarted("read_file", {"path": "README.md"}, id="tc_1")
        yield Final("done")


class _FailingAgent(_FakeAgent):
    async def astream(self, prompt: str) -> Any:
        assert prompt == "hello"
        yield AssistantDelta("before failure")
        raise RuntimeError("provider went away")


def _clock(values: list[float]) -> Iterator[float]:
    yield from values


@pytest.mark.asyncio
async def test_stream_agent_wire_serializes_events_and_final_state() -> None:
    ticks = _clock([10.0, 12.5])

    events = [
        event
        async for event in stream_agent_wire(
            _FakeAgent(),
            "hello",
            clock=lambda: next(ticks),
        )
    ]

    assert events == [
        {"event": "assistant_delta", "text": "hi"},
        {
            "event": "tool_call_started",
            "id": "tc_1",
            "name": "read_file",
            "input": {"path": "README.md"},
        },
        {"event": "final", "message": "done", "reason": "natural"},
        {
            "event": "aura_state",
            "model": "fake:model",
            "mode": "default",
            "cwd": events[-1]["cwd"],
            "tokens": {
                "last_input": 0,
                "last_output": 0,
                "last_cache_read": 0,
                "total_input": 0,
                "total_output": 0,
                "total_cache_read": 0,
                "turn_count": 0,
            },
            "pinned": 0,
            "window": 100,
            "last_turn_seconds": 2.5,
        },
    ]


@pytest.mark.asyncio
async def test_stream_agent_agui_wraps_wire_stream() -> None:
    ticks = _clock([1.0, 1.25])
    adapter = AguiAdapter(run_id="run-1")

    events = [
        event
        async for event in stream_agent_agui(
            _FakeAgent(),
            "hello",
            adapter=adapter,
            clock=lambda: next(ticks),
        )
    ]

    assert events[0]["type"] == "RUN_STARTED"
    assert {"type": "TEXT_MESSAGE_CONTENT", "messageId": "run-1-message", "delta": "hi"} in events
    assert {
        "type": "TOOL_CALL_START",
        "toolCallId": "tc_1",
        "toolCallName": "read_file",
    } in events
    state_index = next(i for i, event in enumerate(events) if event["type"] == "STATE_SNAPSHOT")
    finished_index = next(i for i, event in enumerate(events) if event["type"] == "RUN_FINISHED")
    assert state_index < finished_index
    assert events[-1]["type"] == "RUN_FINISHED"


@pytest.mark.asyncio
async def test_stream_agent_agui_events_encode_as_sse_frames() -> None:
    ticks = _clock([1.0, 1.25])
    adapter = AguiAdapter(run_id="run-1")

    frames = [
        encode_json_sse(event, event="agui")
        async for event in stream_agent_agui(
            _FakeAgent(),
            "hello",
            adapter=adapter,
            clock=lambda: next(ticks),
        )
    ]

    assert frames[0] == (
        'event: agui\ndata: {"type":"RUN_STARTED","runId":"run-1",'
        '"threadId":"run-1-thread"}\n\n'
    )
    assert any('"type":"TOOL_CALL_START"' in frame for frame in frames)
    assert any('"type":"STATE_SNAPSHOT"' in frame for frame in frames)


@pytest.mark.asyncio
async def test_stream_agent_wire_sse_frames_wire_events() -> None:
    ticks = _clock([1.0, 1.25])

    frames = [
        frame
        async for frame in stream_agent_wire_sse(
            _FakeAgent(),
            "hello",
            event="aura",
            clock=lambda: next(ticks),
        )
    ]

    assert frames[0] == 'event: aura\ndata: {"event":"assistant_delta","text":"hi"}\n\n'
    assert any('"event":"aura_state"' in frame for frame in frames)


@pytest.mark.asyncio
async def test_stream_agent_wire_sse_converts_exceptions_to_error_frame() -> None:
    frames = [
        frame
        async for frame in stream_agent_wire_sse(
            _FailingAgent(),
            "hello",
            event="aura",
        )
    ]

    assert frames[-1] == (
        'event: aura\ndata: {"event":"error",'
        '"message":"RuntimeError: provider went away"}\n\n'
    )


@pytest.mark.asyncio
async def test_stream_agent_agui_sse_frames_agui_events() -> None:
    ticks = _clock([1.0, 1.25])
    adapter = AguiAdapter(run_id="run-sse")

    frames = [
        frame
        async for frame in stream_agent_agui_sse(
            _FakeAgent(),
            "hello",
            event="agui",
            adapter=adapter,
            clock=lambda: next(ticks),
        )
    ]

    assert frames[0] == (
        'event: agui\ndata: {"type":"RUN_STARTED","runId":"run-sse",'
        '"threadId":"run-sse-thread"}\n\n'
    )
    state_index = next(i for i, frame in enumerate(frames) if '"type":"STATE_SNAPSHOT"' in frame)
    finished_index = next(i for i, frame in enumerate(frames) if '"type":"RUN_FINISHED"' in frame)
    assert state_index < finished_index


@pytest.mark.asyncio
async def test_stream_agent_agui_errors_close_run() -> None:
    adapter = AguiAdapter(run_id="run-err")

    events = [
        event
        async for event in stream_agent_agui(
            _FailingAgent(),
            "hello",
            adapter=adapter,
        )
    ]

    assert events[-2] == {
        "type": "TEXT_MESSAGE_END",
        "messageId": "run-err-message",
    }
    assert events[-1] == {
        "type": "RUN_ERROR",
        "message": "RuntimeError: provider went away",
    }
