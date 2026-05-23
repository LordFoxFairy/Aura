"""Tests for streaming Agent output through ``stream_agent_wire``."""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from aura.infrastructure.wire.stream import (
    encode_sse,
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
async def test_stream_agent_wire_drops_skipped_no_op_compact_events() -> None:
    class _NoisyCompactAgent(_FakeAgent):
        async def astream(self, prompt: str) -> Any:
            assert prompt == "hello"
            yield {
                "event": "compact_event",
                "trigger": "microcompact",
                "tokens_before": 100,
                "tokens_after": 100,
                "outcome": "skipped",
                "duration_ms": 0.1,
            }
            yield AssistantDelta("hi")
            yield Final("done")
            yield {
                "event": "compact_event",
                "trigger": "auto",
                "tokens_before": 200,
                "tokens_after": 200,
                "outcome": "skipped",
                "duration_ms": 0.1,
            }

    ticks = _clock([10.0, 11.0])
    events = [
        event
        async for event in stream_agent_wire(
            _NoisyCompactAgent(),
            "hello",
            clock=lambda: next(ticks),
        )
    ]
    kinds = [e["event"] for e in events]
    assert kinds == ["assistant_delta", "final", "aura_state"]


@pytest.mark.asyncio
async def test_stream_agent_wire_keeps_compact_events_that_changed_tokens() -> None:
    class _RealCompactAgent(_FakeAgent):
        async def astream(self, prompt: str) -> Any:
            assert prompt == "hello"
            yield AssistantDelta("hi")
            yield Final("done")
            yield {
                "event": "compact_event",
                "trigger": "auto",
                "tokens_before": 5000,
                "tokens_after": 1200,
                "outcome": "ok",
                "duration_ms": 42.0,
            }
            yield {
                "event": "compact_event",
                "trigger": "microcompact",
                "tokens_before": 800,
                "tokens_after": 600,
                "outcome": "skipped",
                "duration_ms": 0.2,
            }

    ticks = _clock([10.0, 11.0])
    events = [
        event
        async for event in stream_agent_wire(
            _RealCompactAgent(),
            "hello",
            clock=lambda: next(ticks),
        )
    ]
    kinds = [e["event"] for e in events]
    assert kinds == ["assistant_delta", "final", "compact_event", "compact_event", "aura_state"]


@pytest.mark.asyncio
async def test_stream_agent_wire_emits_aura_state_last_even_after_post_final_events() -> None:
    class _PostFinalAgent(_FakeAgent):
        async def astream(self, prompt: str) -> Any:
            assert prompt == "hello"
            yield AssistantDelta("hi")
            yield Final("done")
            yield {
                "event": "compact_event",
                "trigger": "auto",
                "tokens_before": 5000,
                "tokens_after": 1200,
                "outcome": "ok",
                "duration_ms": 42.0,
            }

    ticks = _clock([10.0, 11.0])
    events = [
        event
        async for event in stream_agent_wire(
            _PostFinalAgent(),
            "hello",
            clock=lambda: next(ticks),
        )
    ]
    assert events[-1]["event"] == "aura_state"


def test_encode_sse_emits_single_aura_frame() -> None:
    frame = encode_sse(cast(Any, {"event": "foo", "x": 1}))
    assert frame == 'event: aura\ndata: {"event": "foo", "x": 1}\n\n'


@pytest.mark.asyncio
async def test_stream_agent_wire_sse_yields_well_formed_frames() -> None:
    ticks = _clock([10.0, 12.5])
    frames = [
        frame
        async for frame in stream_agent_wire_sse(
            _FakeAgent(),
            "hello",
            clock=lambda: next(ticks),
        )
    ]
    assert len(frames) == 4
    for frame in frames:
        assert frame.startswith("event: aura\ndata: ")
        assert frame.endswith("\n\n")
        payload = json.loads(frame.split("\ndata: ", 1)[1].rstrip("\n"))
        assert "event" in payload
    assert json.loads(frames[0].split("\ndata: ", 1)[1].rstrip("\n")) == {
        "event": "assistant_delta",
        "text": "hi",
    }
