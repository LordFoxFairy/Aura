"""Tests for streaming Agent output through transport adapters."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from aura.adapters.protocol.bridge import AguiAdapter
from aura.adapters.protocol.stream import (
    encode_json_sse,
    stream_agent_agui,
    stream_agent_agui_sse,
    stream_agent_wire,
    stream_agent_wire_sse,
)
from aura.schemas.events import AssistantDelta, Final, ToolCallStarted
from aura.schemas.state import LoopSlots


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


class _CoordinationAgent(_FakeAgent):
    session_id = "parent-1"

    def __init__(self) -> None:
        self._pending_protocol_events = [
            {
                "event": "coordination",
                "family": "subagent",
                "action": "task_notification",
                "subagent_id": "task-1",
                "parent_id": "parent-1",
                "payload": {
                    "task_id": "task-1",
                    "status": "completed",
                    "summary": "child-final",
                    "description": "probe",
                    "terminal": True,
                },
            },
            {
                "event": "coordination",
                "family": "team",
                "action": "message_sent",
                "team_id": "demo",
                "member_id": "scout",
                "payload": {
                    "msg_id": "msg-1",
                    "sender": "leader",
                    "recipient": "scout",
                    "body": "ping",
                    "kind": "text",
                    "sent_at": 123.0,
                },
            },
        ]

    @property
    def pending_protocol_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._pending_protocol_events)

    def drain_protocol_events(self) -> list[dict[str, Any]]:
        drained = list(self._pending_protocol_events)
        self._pending_protocol_events.clear()
        return drained

    async def astream(self, prompt: str) -> Any:
        assert prompt == "hello"
        yield AssistantDelta("hi")
        yield Final("done")


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

    state_event = cast(dict[str, Any], events[-1])

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
            "cwd": state_event["cwd"],
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
async def test_stream_agent_wire_drains_coordination_events_from_runtime_paths() -> None:
    ticks = _clock([10.0, 12.5])

    events = [
        event
        async for event in stream_agent_wire(
            _CoordinationAgent(),
            "hello",
            clock=lambda: next(ticks),
        )
    ]

    assert events == [
        {
            "event": "coordination",
            "family": "subagent",
            "action": "task_notification",
            "subagent_id": "task-1",
            "parent_id": "parent-1",
            "payload": {
                "task_id": "task-1",
                "status": "completed",
                "summary": "child-final",
                "description": "probe",
                "terminal": True,
            },
        },
        {
            "event": "coordination",
            "family": "team",
            "action": "message_sent",
            "team_id": "demo",
            "member_id": "scout",
            "payload": {
                "msg_id": "msg-1",
                "sender": "leader",
                "recipient": "scout",
                "body": "ping",
                "kind": "text",
                "sent_at": 123.0,
            },
        },
        {"event": "assistant_delta", "text": "hi"},
        {"event": "final", "message": "done", "reason": "natural"},
        {
            "event": "aura_state",
            "model": "fake:model",
            "mode": "default",
            "cwd": cast(dict[str, Any], events[-1])["cwd"],
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
async def test_stream_agent_agui_drains_coordination_events_from_runtime_paths() -> None:
    ticks = _clock([1.0, 1.25])
    adapter = AguiAdapter(run_id="run-coord")

    events = [
        event
        async for event in stream_agent_agui(
            _CoordinationAgent(),
            "hello",
            adapter=adapter,
            clock=lambda: next(ticks),
        )
    ]

    assert events[0] == {
        "type": "RUN_STARTED",
        "runId": "run-coord",
        "threadId": "run-coord-thread",
    }
    assert {
        "type": "CUSTOM",
        "name": "aura.subagent.task_notification",
        "value": {
            "event": "coordination",
            "family": "subagent",
            "action": "task_notification",
            "subagent_id": "task-1",
            "parent_id": "parent-1",
            "payload": {
                "task_id": "task-1",
                "status": "completed",
                "summary": "child-final",
                "description": "probe",
                "terminal": True,
            },
        },
    } in events
    assert {
        "type": "CUSTOM",
        "name": "aura.team.message_sent",
        "value": {
            "event": "coordination",
            "family": "team",
            "action": "message_sent",
            "team_id": "demo",
            "member_id": "scout",
            "payload": {
                "msg_id": "msg-1",
                "sender": "leader",
                "recipient": "scout",
                "body": "ping",
                "kind": "text",
                "sent_at": 123.0,
            },
        },
    } in events


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
