"""Serialize internal Aura events to their stable external wire shape."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol, TypeGuard

from aura.domain.events import (
    AgentEvent,
    AssistantDelta,
    Final,
    PermissionAudit,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)
from aura.domain.task import TaskNotification
from aura.domain.team import TeamMessage
from aura.infrastructure.wire.events import (
    AuraStateEvent,
    CompactEvent,
    CoordinationEvent,
    JSONValue,
    PermissionRequestEvent,
    ToolCallCompletedEvent,
    ToolCallProgressEvent,
    ToolCallStartedEvent,
    WireEvent,
)


class HasTokenStats(Protocol):
    last_input_tokens: int
    last_output_tokens: int
    last_cache_read_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    turn_count: int


class HasLoopSlots(Protocol):
    token_stats: HasTokenStats


class HasLoopState(Protocol):
    slots: HasLoopSlots


class HasAgentState(Protocol):
    @property
    def state(self) -> HasLoopState: ...

    @property
    def current_model(self) -> str: ...

    @property
    def mode(self) -> str: ...

    @property
    def pinned_tokens_estimate(self) -> int: ...

    @property
    def context_window(self) -> int: ...


def _is_wire_event(event: object) -> TypeGuard[WireEvent]:
    return isinstance(event, dict) and isinstance(event.get("event"), str)


def event_to_wire(event: AgentEvent | WireEvent | object) -> WireEvent:
    """Convert one internal event into Aura's external wire shape."""
    if _is_wire_event(event):
        return event
    if isinstance(event, AssistantDelta):
        return {"event": "assistant_delta", "text": event.text}
    if isinstance(event, ToolCallStarted):
        started_event: ToolCallStartedEvent = {
            "event": "tool_call_started",
            "name": event.name,
            "input": event.input,
        }
        if event.id:
            started_event["id"] = event.id
        return started_event
    if isinstance(event, ToolCallProgress):
        progress_event: ToolCallProgressEvent = {
            "event": "tool_call_progress",
            "name": event.name,
            "stream": event.stream,
            "chunk": event.chunk,
        }
        if event.id:
            progress_event["id"] = event.id
        return progress_event
    if isinstance(event, ToolCallCompleted):
        # content.output is raw JSON so the frontend decodes once, not twice.
        is_error = event.error is not None
        output: JSONValue = str(event.error) if is_error else _json_safe(event.output)
        completed_event: ToolCallCompletedEvent = {
            "event": "tool_call_completed",
            "name": event.name,
            "content": {"output": output, "error": is_error},
        }
        if event.id:
            completed_event["id"] = event.id
        return completed_event
    if isinstance(event, PermissionAudit):
        return {
            "event": "permission_audit",
            "tool": event.tool,
            "text": event.text,
        }
    if isinstance(event, Final):
        return {
            "event": "final",
            "message": event.message,
            "reason": event.reason,
        }
    return {"event": "unknown", "type": type(event).__name__}


def permission_request_to_wire(
    *,
    request_id: str,
    tool: str,
    args: object,
    rule_hint: str,
    is_destructive: bool,
) -> PermissionRequestEvent:
    """Build the external permission prompt event used by interactive UIs."""
    return {
        "event": "permission_request",
        "id": request_id,
        "tool": tool,
        "args": _json_safe(args),
        "rule_hint": rule_hint,
        "is_destructive": bool(is_destructive),
    }


def compact_event_to_wire(
    *,
    trigger: str,
    tokens_before: int,
    tokens_after: int,
    outcome: str,
    duration_ms: float,
) -> CompactEvent:
    return {
        "event": "compact_event",
        "trigger": trigger,
        "tokens_before": int(tokens_before),
        "tokens_after": int(tokens_after),
        "outcome": outcome,
        "duration_ms": float(duration_ms),
    }


def agent_state_to_wire(agent: HasAgentState, last_turn_seconds: float) -> AuraStateEvent:
    """Snapshot agent state into the external ``aura_state`` event."""
    stats = agent.state.slots.token_stats
    return {
        "event": "aura_state",
        "model": agent.current_model,
        "mode": agent.mode,
        "cwd": str(Path.cwd()),
        "tokens": {
            "last_input": int(stats.last_input_tokens),
            "last_output": int(stats.last_output_tokens),
            "last_cache_read": int(stats.last_cache_read_tokens),
            "total_input": int(stats.total_input_tokens),
            "total_output": int(stats.total_output_tokens),
            "total_cache_read": int(stats.total_cache_read_tokens),
            "turn_count": int(stats.turn_count),
        },
        "pinned": int(agent.pinned_tokens_estimate),
        "window": int(agent.context_window),
        "last_turn_seconds": float(last_turn_seconds),
    }


def task_notification_to_wire(
    notification: TaskNotification,
    *,
    parent_id: str | None = None,
) -> CoordinationEvent:
    """Map a terminal subagent task notification to coordination wire shape."""
    if notification.status == "running":
        raise ValueError("task_notification_to_wire requires a terminal status")
    status: Literal["completed", "failed", "cancelled"] = (
        "completed" if notification.status == "completed"
        else "failed" if notification.status == "failed"
        else "cancelled"
    )
    payload: CoordinationEvent = {
        "event": "coordination",
        "family": "subagent",
        "action": "task_notification",
        "subagent_id": notification.task_id,
        "payload": {
            "task_id": notification.task_id,
            "status": status,
            "summary": notification.summary,
            "description": notification.description,
            "terminal": True,
        },
    }
    if parent_id:
        payload["parent_id"] = parent_id
    return payload


def task_started_to_wire(
    *,
    task_id: str,
    description: str,
    parent_session_id: str,
    started_at: float,
    parent_id: str | None = None,
) -> CoordinationEvent:
    """Build the live ``task_started`` coordination event."""
    payload: CoordinationEvent = {
        "event": "coordination",
        "family": "subagent",
        "action": "task_started",
        "subagent_id": task_id,
        "payload": {
            "task_id": task_id,
            "description": description,
            "parent_session_id": parent_session_id,
            "started_at": float(started_at),
        },
    }
    if parent_id:
        payload["parent_id"] = parent_id
    return payload


def task_progress_to_wire(
    *,
    task_id: str,
    tool_name: str,
    activity_count: int,
    parent_id: str | None = None,
) -> CoordinationEvent:
    """Build a per-tool-start ``task_progress`` coordination event."""
    payload: CoordinationEvent = {
        "event": "coordination",
        "family": "subagent",
        "action": "task_progress",
        "subagent_id": task_id,
        "payload": {
            "task_id": task_id,
            "tool_name": tool_name,
            "activity_count": int(activity_count),
        },
    }
    if parent_id:
        payload["parent_id"] = parent_id
    return payload


def team_message_to_wire(
    message: TeamMessage,
    *,
    team_id: str,
    member_id: str | None = None,
) -> CoordinationEvent:
    """Map a team mailbox send payload to coordination wire shape."""
    action: Literal["message_sent", "control_sent"] = (
        "message_sent" if message.kind == "text" else "control_sent"
    )
    return {
        "event": "coordination",
        "family": "team",
        "action": action,
        "team_id": team_id,
        "member_id": member_id or message.recipient,
        "payload": {
            "msg_id": message.msg_id,
            "sender": message.sender,
            "recipient": message.recipient,
            "body": message.body,
            "kind": message.kind,
            "sent_at": float(message.sent_at),
        },
    }


def _is_json_value(value: object) -> TypeGuard[JSONValue]:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in value.items()
        )
    return False


def _json_safe(value: object) -> JSONValue:
    try:
        raw = json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return {"_repr": repr(value)}
    if _is_json_value(raw):
        return raw
    return {"_repr": repr(raw)}
