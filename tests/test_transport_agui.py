"""Tests for mapping Aura wire events to AG-UI-style events."""

from __future__ import annotations

from aura.transport.agui import AguiAdapter


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
    assert adapter.convert({
        "event": "tool_call_completed",
        "id": "tc_1",
        "name": "bash",
        "output": {"stdout": "hi\n"},
        "error": None,
    }) == [{
        "type": "TOOL_CALL_RESULT",
        "messageId": "run-2-message",
        "toolCallId": "tc_1",
        "content": "{\"stdout\":\"hi\\n\"}",
        "role": "tool",
    }]


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
        "output": {"ok": True},
        "error": None,
    })
    second_completed = adapter.convert({
        "event": "tool_call_completed",
        "name": "bash",
        "output": {"ok": True},
        "error": None,
    })

    first_id = first[0]["toolCallId"]
    second_id = second[0]["toolCallId"]
    assert first_id != second_id
    assert first_completed[0]["toolCallId"] == first_id
    assert second_completed[0]["toolCallId"] == second_id


def test_agui_adapter_maps_aura_specific_events_to_custom() -> None:
    adapter = AguiAdapter(run_id="run-3")

    assert adapter.convert({
        "event": "permission_request",
        "id": "perm_1",
        "tool": "write_file",
    }) == [{
        "type": "CUSTOM",
        "name": "aura.permission.request",
        "value": {
            "event": "permission_request",
            "id": "perm_1",
            "tool": "write_file",
        },
    }]
    assert adapter.convert({"event": "aura_state", "mode": "default"}) == [{
        "type": "STATE_SNAPSHOT",
        "snapshot": {"event": "aura_state", "mode": "default"},
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
