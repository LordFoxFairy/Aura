"""Tests for mapping Aura wire events to AG-UI-style events."""

from __future__ import annotations

import json
from types import SimpleNamespace

from aura.adapters.protocol.agui import AguiAdapter
from aura.adapters.protocol.bridge import AguiAdapter as BridgeAguiAdapter
from aura.adapters.protocol.bridge import AguiEventBridge
from aura.domain.protocol.events import (
    AuraStateEvent,
    PermissionRequestEvent,
    SubagentProtocolEvent,
    TeamProtocolEvent,
)
from aura.schemas.events import AssistantDelta, Final
from aura.schemas.state import LoopSlots


def test_agui_adapter_is_bridge_owned_implementation() -> None:
    assert AguiAdapter is BridgeAguiAdapter
    assert AguiAdapter.__module__ == "aura.adapters.protocol.bridge"


def test_agui_adapter_wraps_text_stream_in_run_and_message_events() -> None:
    adapter = AguiAdapter(run_id="run-1", thread_id="thread-1")

    assert adapter.start_run() == [{
        "type": "RUN_STARTED",
        "runId": "run-1",
        "threadId": "thread-1",
    }]
    text_events = adapter.convert({"event": "assistant_delta", "text": "hi"})
    assert text_events == [
        {
            "type": "TEXT_MESSAGE_START",
            "messageId": "run-1-message",
            "role": "assistant",
        },
        {
            "type": "TEXT_MESSAGE_CONTENT",
            "messageId": "run-1-message",
            "delta": "hi",
        },
    ]
    assert adapter.convert({
        "event": "final",
        "message": "done",
        "reason": "natural",
    }) == [
        {
            "type": "TEXT_MESSAGE_END",
            "messageId": "run-1-message",
        },
        {
            "type": "RUN_FINISHED",
            "runId": "run-1",
            "threadId": "thread-1",
            "result": {"reason": "natural"},
        },
    ]


def test_agui_adapter_defaults_to_string_thread_id() -> None:
    adapter = AguiAdapter(run_id="run-1")

    assert adapter.start_run()[0]["threadId"] == "run-1-thread"
    assert adapter.finish_run()[0]["threadId"] == "run-1-thread"


def test_agui_adapter_maps_tool_events_and_progress() -> None:
    adapter = AguiAdapter(run_id="run-2")

    assert adapter.convert({
        "event": "tool_call_started",
        "id": "tc_1",
        "name": "bash",
        "input": {"command": "echo hi"},
    }) == [
        {
            "type": "TOOL_CALL_START",
            "toolCallId": "tc_1",
            "toolCallName": "bash",
        },
        {
            "type": "TOOL_CALL_ARGS",
            "toolCallId": "tc_1",
            "delta": "{\"command\":\"echo hi\"}",
        },
        {
            "type": "TOOL_CALL_END",
            "toolCallId": "tc_1",
        },
    ]
    assert adapter.convert({
        "event": "tool_call_progress",
        "id": "tc_1",
        "name": "bash",
        "stream": "stdout",
        "chunk": "hi\n",
    }) == [{
        "type": "CUSTOM",
        "name": "aura.tool.progress",
        "value": {
            "toolCallId": "tc_1",
            "toolName": "bash",
            "stream": "stdout",
            "chunk": "hi\n",
        },
    }]
    # Phase 2 Task 9: AG-UI ``TOOL_CALL_RESULT.content`` carries a
    # JSON-encoded structured payload {text, error}.
    assert adapter.convert({
        "event": "tool_call_completed",
        "id": "tc_1",
        "name": "bash",
        "content": {"text": "{\"stdout\":\"hi\\n\"}", "error": False},
    }) == [{
        "type": "TOOL_CALL_RESULT",
        "messageId": "run-2-message",
        "toolCallId": "tc_1",
        "content": "{\"text\":\"{\\\"stdout\\\":\\\"hi\\\\n\\\"}\",\"error\":false}",
        "role": "tool",
    }]


def test_agui_adapter_tool_completed_error_payload_carries_error_flag() -> None:
    # Phase 2 Task 9: a failing tool's ``TOOL_CALL_RESULT.content`` is
    # the JSON-encoded structured payload — clients can decode and
    # see the boolean ``error`` flag (no more guessing whether bare
    # text is success-output or an error message).
    adapter = AguiAdapter(run_id="run-err")

    out = adapter.convert({
        "event": "tool_call_completed",
        "id": "tc_1",
        "name": "bash",
        "content": {"text": "permission denied", "error": True},
    })

    assert len(out) == 1
    assert out[0]["type"] == "TOOL_CALL_RESULT"
    parsed = json.loads(out[0]["content"])
    assert parsed == {"text": "permission denied", "error": True}


def test_agui_adapter_gives_idless_tool_calls_distinct_fallback_ids() -> None:
    adapter = AguiAdapter(run_id="run-2")

    first = adapter.convert({
        "event": "tool_call_started",
        "name": "bash",
        "input": {"cmd": "one"},
    })
    second = adapter.convert({
        "event": "tool_call_started",
        "name": "bash",
        "input": {"cmd": "two"},
    })
    first_completed = adapter.convert({
        "event": "tool_call_completed",
        "name": "bash",
        "content": {"text": "{\"ok\":true}", "error": False},
    })
    second_completed = adapter.convert({
        "event": "tool_call_completed",
        "name": "bash",
        "content": {"text": "{\"ok\":true}", "error": False},
    })

    first_id = first[0]["toolCallId"]
    second_id = second[0]["toolCallId"]
    assert first_id != second_id
    assert first_completed[0]["toolCallId"] == first_id
    assert second_completed[0]["toolCallId"] == second_id


def test_agui_adapter_maps_aura_specific_events_to_custom() -> None:
    adapter = AguiAdapter(run_id="run-3")
    permission_request: PermissionRequestEvent = {
        "event": "permission_request",
        "id": "perm_1",
        "tool": "write_file",
        "args": {},
        "rule_hint": "write_file",
        "is_destructive": False,
    }
    aura_state: AuraStateEvent = {
        "event": "aura_state",
        "model": "",
        "mode": "default",
        "cwd": "/tmp",
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
        "window": 0,
        "last_turn_seconds": 0.0,
    }

    assert adapter.convert(permission_request) == [{
        "type": "CUSTOM",
        "name": "aura.permission.request",
        "value": permission_request,
    }]
    assert adapter.convert(aura_state) == [{
        "type": "STATE_SNAPSHOT",
        "snapshot": aura_state,
    }]


def test_agui_adapter_maps_coordination_placeholders_to_family_specific_custom_events() -> None:
    adapter = AguiAdapter(run_id="run-coord")
    subagent_event: SubagentProtocolEvent = {
        "event": "coordination",
        "family": "subagent",
        "action": "task_notification",
        "subagent_id": "sa_1",
        "payload": {
            "task_id": "sa_1",
            "status": "completed",
            "summary": None,
            "description": "coordination probe",
            "terminal": True,
        },
    }
    team_event: TeamProtocolEvent = {
        "event": "coordination",
        "family": "team",
        "action": "message_sent",
        "team_id": "team_1",
        "member_id": "scout",
        "payload": {
            "msg_id": "msg_1",
            "sender": "leader",
            "recipient": "scout",
            "body": "ping",
            "kind": "text",
            "sent_at": 123.0,
        },
    }

    assert adapter.convert(subagent_event) == [{
        "type": "CUSTOM",
        "name": "aura.subagent.task_notification",
        "value": subagent_event,
    }]
    assert adapter.convert(team_event) == [{
        "type": "CUSTOM",
        "name": "aura.team.message_sent",
        "value": team_event,
    }]


def test_agui_adapter_error_closes_open_text_and_run() -> None:
    adapter = AguiAdapter(run_id="run-4")
    adapter.convert({"event": "assistant_delta", "text": "partial"})

    assert adapter.error("boom") == [
        {
            "type": "TEXT_MESSAGE_END",
            "messageId": "run-4-message",
        },
        {
            "type": "RUN_ERROR",
            "message": "boom",
        },
    ]


def test_agui_event_bridge_owns_run_start_state_and_final_ordering() -> None:
    bridge = AguiEventBridge(adapter=AguiAdapter(run_id="run-bridge"), clock=lambda: 11.5)
    agent = SimpleNamespace(
        current_model="fake:model",
        mode="default",
        pinned_tokens_estimate=0,
        context_window=100,
        state=SimpleNamespace(slots=LoopSlots()),
    )

    events = []
    events.extend(bridge.start())
    events.extend(bridge.emit(AssistantDelta("hi"), agent=agent))
    events.extend(bridge.emit(Final("done"), agent=agent, turn_started_at=10.0))

    assert [event["type"] for event in events] == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "STATE_SNAPSHOT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
