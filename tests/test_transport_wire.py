"""Tests for the transport-neutral Aura wire event contract."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import get_args

from rich.console import Console

from aura.domain.task import TaskNotification
from aura.infrastructure.wire.event_dto import (
    AssistantDeltaEvent,
    CompactEvent,
    ErrorEvent,
    FinalEvent,
    PermissionAuditEvent,
    PermissionRequestEvent,
    SubagentProgressEvent,
    SubagentProtocolEvent,
    SubagentStartedEvent,
    TeamProtocolEvent,
    ToolCallCompletedEvent,
    ToolCallProgressEvent,
    ToolCallStartedEvent,
    UnknownEvent,
    WireEvent,
)
from aura.infrastructure.wire.wire import (
    agent_state_to_wire,
    event_to_wire,
    permission_request_to_wire,
    task_notification_to_wire,
    task_progress_to_wire,
    task_started_to_wire,
)
from aura.schemas.events import (
    AgentEvent,
    AssistantDelta,
    Final,
    PermissionAudit,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)
from cli.render import Renderer
from desktop.host import headless


class _UnsafeArg:
    def __str__(self) -> str:
        return "unsafe-value"


def test_wire_event_is_explicit_union_of_protocol_families() -> None:
    members = set(get_args(WireEvent))

    assert AssistantDeltaEvent in members
    assert ToolCallStartedEvent in members
    assert ToolCallProgressEvent in members
    assert ToolCallCompletedEvent in members
    assert PermissionRequestEvent in members
    assert PermissionAuditEvent in members
    assert FinalEvent in members
    assert CompactEvent in members
    assert ErrorEvent in members
    assert SubagentProtocolEvent in members
    assert SubagentStartedEvent in members
    assert SubagentProgressEvent in members
    assert TeamProtocolEvent in members
    assert UnknownEvent in members


def test_event_to_wire_preserves_desktop_event_shapes() -> None:
    assert event_to_wire.__module__ == "aura.infrastructure.wire.wire"
    assert event_to_wire(AssistantDelta("hi")) == {
        "event": "assistant_delta",
        "text": "hi",
    }
    assert event_to_wire(
        ToolCallStarted("read_file", {"path": "README.md"}, id="tc_1"),
    ) == {
        "event": "tool_call_started",
        "id": "tc_1",
        "name": "read_file",
        "input": {"path": "README.md"},
    }
    assert event_to_wire(
        ToolCallProgress("bash", "stdout", "ok\n", id="tc_2"),
    ) == {
        "event": "tool_call_progress",
        "id": "tc_2",
        "name": "bash",
        "stream": "stdout",
        "chunk": "ok\n",
    }
    assert event_to_wire(
        ToolCallCompleted("grep", {"matches": 1}, id="tc_3"),
    ) == {
        "event": "tool_call_completed",
        "id": "tc_3",
        "name": "grep",
        "content": {"output": {"matches": 1}, "error": False},
    }
    assert event_to_wire(PermissionAudit("bash", "auto-allowed: rule")) == {
        "event": "permission_audit",
        "tool": "bash",
        "text": "auto-allowed: rule",
    }
    assert event_to_wire(Final("done")) == {
        "event": "final",
        "message": "done",
        "reason": "natural",
    }


def test_event_to_wire_omits_absent_optional_ids_for_compatibility() -> None:
    assert "id" not in event_to_wire(ToolCallStarted("read_file", {}))
    assert "id" not in event_to_wire(ToolCallProgress("bash", "stderr", "x"))
    assert "id" not in event_to_wire(ToolCallCompleted("bash", None))


def test_event_to_wire_tool_completed_error_carries_structured_content() -> None:
    # On error: ``content.output`` is the error message string and
    # ``content.error`` is True — one shape for both success and failure.
    payload = event_to_wire(
        ToolCallCompleted("bash", output=None, error="boom: rm refused"),
    )
    assert payload["event"] == "tool_call_completed"
    assert payload["name"] == "bash"
    assert payload["content"] == {"output": "boom: rm refused", "error": True}
    # Top-level split fields are gone — only ``content`` carries result data.
    assert "output" not in payload
    assert "error" not in payload


def test_event_to_wire_omits_empty_optional_ids_for_compatibility() -> None:
    assert "id" not in event_to_wire(ToolCallStarted("read_file", {}, id=""))
    assert "id" not in event_to_wire(ToolCallProgress("bash", "stderr", "x", id=""))
    assert "id" not in event_to_wire(ToolCallCompleted("bash", None, id=""))


def test_permission_request_to_wire_uses_frontend_contract_and_safe_args() -> None:
    payload = permission_request_to_wire(
        request_id="perm_1",
        tool="bash",
        args={"cmd": _UnsafeArg()},
        rule_hint='bash:{"cmd":"echo hi"}',
        is_destructive=True,
    )

    assert payload == {
        "event": "permission_request",
        "id": "perm_1",
        "tool": "bash",
        "args": {"cmd": "unsafe-value"},
        "rule_hint": 'bash:{"cmd":"echo hi"}',
        "is_destructive": True,
    }


def test_task_notification_to_wire_maps_terminal_subagent_notification() -> None:
    payload = task_notification_to_wire(
        TaskNotification(
            task_id="task_12345678",
            status="completed",
            summary="child-final",
            description="probe",
        ),
    )

    assert payload == {
        "event": "coordination",
        "family": "subagent",
        "action": "task_notification",
        "subagent_id": "task_12345678",
        "payload": {
            "task_id": "task_12345678",
            "status": "completed",
            "summary": "child-final",
            "description": "probe",
            "terminal": True,
        },
    }


def test_task_started_to_wire_shapes_live_progress_event() -> None:
    payload = task_started_to_wire(
        task_id="task_abcdef",
        description="probe",
        parent_session_id="parent_sess",
        started_at=1234.5,
        parent_id="parent_sess",
    )

    assert payload == {
        "event": "coordination",
        "family": "subagent",
        "action": "task_started",
        "subagent_id": "task_abcdef",
        "payload": {
            "task_id": "task_abcdef",
            "description": "probe",
            "parent_session_id": "parent_sess",
            "started_at": 1234.5,
        },
        "parent_id": "parent_sess",
    }


def test_task_progress_to_wire_carries_tool_name_and_count() -> None:
    payload = task_progress_to_wire(
        task_id="task_abcdef",
        tool_name="read_file",
        activity_count=3,
    )

    assert payload == {
        "event": "coordination",
        "family": "subagent",
        "action": "task_progress",
        "subagent_id": "task_abcdef",
        "payload": {
            "task_id": "task_abcdef",
            "tool_name": "read_file",
            "activity_count": 3,
        },
    }


def test_agent_state_to_wire_uses_stable_numeric_shape() -> None:
    from aura.schemas.state import LoopSlots, TokenStats

    agent = SimpleNamespace(
        state=SimpleNamespace(slots=LoopSlots(
            token_stats=TokenStats(
                last_input_tokens=10,
                last_output_tokens=20,
                last_cache_read_tokens=3,
                total_input_tokens=100,
                total_output_tokens=200,
                total_cache_read_tokens=30,
                turn_count=4,
            ),
        )),
        current_model="openai:gpt-4o-mini",
        mode="default",
        pinned_tokens_estimate=123,
        context_window=456,
    )

    payload = agent_state_to_wire(agent, 1.5)

    assert payload["event"] == "aura_state"
    assert payload["model"] == "openai:gpt-4o-mini"
    assert payload["mode"] == "default"
    assert payload["tokens"]["last_input"] == 10
    assert payload["tokens"]["total_output"] == 200
    assert payload["pinned"] == 123
    assert payload["window"] == 456
    assert payload["last_turn_seconds"] == 1.5


def test_cli_renderer_and_desktop_wire_accept_same_agent_event_set() -> None:
    events: list[AgentEvent] = [
        AssistantDelta("hello"),
        ToolCallStarted("read_file", {"path": "README.md"}, id="tc_read"),
        PermissionAudit("read_file", "auto-allowed: rule `read_file`"),
        ToolCallProgress("bash", "stdout", "ok\n", id="tc_bash"),
        ToolCallCompleted("read_file", {"content": "hello"}, id="tc_read"),
        Final("done"),
    ]
    console_file = io.StringIO()
    renderer = Renderer(
        Console(file=console_file, force_terminal=False, width=120),
    )

    for event in events:
        renderer.on_event(event)
        assert headless._event_to_dict(event) == event_to_wire(event)
    renderer.finish()

    rendered = console_file.getvalue()
    assert "read_file" in rendered
    assert "auto-allowed" in rendered
