"""Serialize internal Aura events to their stable external wire shape."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

from aura.domain.task import TaskNotification
from aura.domain.team import TeamMessage
from aura.infrastructure.wire.event_dto import (
    AuraStateEvent,
    CompactEvent,
    PermissionRequestEvent,
    SubagentProtocolEvent,
    TeamProtocolEvent,
    WireEvent,
)
from aura.schemas.events import (
    AssistantDelta,
    Final,
    PermissionAudit,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)


def event_to_wire(event: Any) -> WireEvent:
    """Convert one internal event into Aura's external wire shape."""
    if isinstance(event, dict):
        return cast(WireEvent, event)
    if isinstance(event, AssistantDelta):
        return {"event": "assistant_delta", "text": event.text}
    if isinstance(event, ToolCallStarted):
        payload: dict[str, Any] = {
            "event": "tool_call_started",
            "name": event.name,
            "input": event.input,
        }
        if event.id:
            payload["id"] = event.id
        return cast(WireEvent, payload)
    if isinstance(event, ToolCallProgress):
        payload = {
            "event": "tool_call_progress",
            "name": event.name,
            "stream": event.stream,
            "chunk": event.chunk,
        }
        if event.id:
            payload["id"] = event.id
        return cast(WireEvent, payload)
    if isinstance(event, ToolCallCompleted):
        is_error = event.error is not None
        if is_error:
            text = str(event.error)
        else:
            try:
                text = json.dumps(event.output, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                text = repr(event.output)
        payload = {
            "event": "tool_call_completed",
            "name": event.name,
            "content": {"text": text, "error": is_error},
        }
        if event.id:
            payload["id"] = event.id
        return cast(WireEvent, payload)
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
            "reason": getattr(event, "reason", "natural"),
        }
    return {"event": "unknown", "type": type(event).__name__}


def permission_request_to_wire(
    *,
    request_id: str,
    tool: str,
    args: Any,
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


def agent_state_to_wire(agent: Any, last_turn_seconds: float) -> AuraStateEvent:
    """Snapshot agent state into the external ``aura_state`` event."""
    stats = agent.state.slots.token_stats
    return {
        "event": "aura_state",
        "model": agent.current_model or "",
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
        "pinned": int(agent.pinned_tokens_estimate or 0),
        "window": int(agent.context_window or 0),
        "last_turn_seconds": float(last_turn_seconds),
    }


def task_notification_to_wire(
    notification: TaskNotification,
    *,
    parent_id: str | None = None,
) -> SubagentProtocolEvent:
    """Map a terminal subagent task notification to coordination wire shape."""
    if notification.status == "running":
        raise ValueError("task_notification_to_wire requires a terminal status")
    status: Literal["completed", "failed", "cancelled"] = (
        "completed" if notification.status == "completed"
        else "failed" if notification.status == "failed"
        else "cancelled"
    )
    payload: SubagentProtocolEvent = {
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


def team_message_to_wire(
    message: TeamMessage,
    *,
    team_id: str,
    member_id: str | None = None,
) -> TeamProtocolEvent:
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


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return {"_repr": repr(value)}
