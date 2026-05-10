"""Aura's transport-neutral JSON event contract.

This module is the single place where internal ``AgentEvent`` dataclasses are
converted to JSON-friendly dictionaries. Desktop/headless, future SSE, and
AG-UI adapters should depend on this layer instead of re-serializing
``AgentEvent`` independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aura.schemas.events import (
    AssistantDelta,
    Final,
    PermissionAudit,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)


def event_to_wire(event: Any) -> dict[str, Any]:
    """Convert one internal event into Aura's stable external shape."""
    if isinstance(event, AssistantDelta):
        return {"event": "assistant_delta", "text": event.text}
    if isinstance(event, ToolCallStarted):
        started_payload: dict[str, Any] = {
            "event": "tool_call_started",
            "name": event.name,
            "input": event.input,
        }
        if event.id:
            started_payload["id"] = event.id
        return started_payload
    if isinstance(event, ToolCallProgress):
        progress_payload: dict[str, Any] = {
            "event": "tool_call_progress",
            "name": event.name,
            "stream": event.stream,
            "chunk": event.chunk,
        }
        if event.id:
            progress_payload["id"] = event.id
        return progress_payload
    if isinstance(event, ToolCallCompleted):
        # Phase 2 Task 9 — unified tool-result wire shape. The legacy
        # ``output``/``error`` split is replaced by a single
        # ``content: {"text": ..., "error": bool}`` payload so the
        # frontend has ONE shape to render (and ``error: true`` drives
        # the red banner). ``text`` is the same string the model sees
        # in the ToolMessage (success → JSON of output, failure →
        # error+hint), so the frontend never has to re-format JSON
        # for display.
        is_error = event.error is not None
        if is_error:
            text = str(event.error)
        else:
            try:
                text = json.dumps(event.output, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                text = repr(event.output)
        completed_payload: dict[str, Any] = {
            "event": "tool_call_completed",
            "name": event.name,
            "content": {"text": text, "error": is_error},
        }
        if event.id:
            completed_payload["id"] = event.id
        return completed_payload
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
) -> dict[str, Any]:
    """Build the external permission prompt event used by interactive UIs."""
    return {
        "event": "permission_request",
        "id": request_id,
        "tool": tool,
        "args": _json_safe(args),
        "rule_hint": rule_hint,
        "is_destructive": bool(is_destructive),
    }


def agent_state_to_wire(agent: Any, last_turn_seconds: float) -> dict[str, Any]:
    """Snapshot agent state into the external ``aura_state`` event.

    Reads from the typed :class:`aura.schemas.state.TokenStats` slot
    (``state.slots.token_stats``); the wire shape (``last_input`` etc.)
    is preserved verbatim so external JSON consumers see no change.
    """
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


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return {"_repr": repr(value)}
