"""AG-UI-style adapter for Aura wire events."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any
from uuid import uuid4

from aura.domain.protocol.events import WireEvent


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


class AguiAdapter:
    """Stateful converter from Aura wire events to AG-UI-style events."""

    def __init__(self, *, run_id: str | None = None, thread_id: str | None = None):
        self.run_id = run_id or uuid4().hex
        self.thread_id = thread_id or f"{self.run_id}-thread"
        self._message_id = f"{self.run_id}-message"
        self._text_open = False
        self._run_finished = False
        self._next_tool_fallback = 0
        self._open_tool_fallbacks: dict[str, list[str]] = defaultdict(list)

    def start_run(self) -> list[dict[str, Any]]:
        return [{
            "type": "RUN_STARTED",
            "runId": self.run_id,
            "threadId": self.thread_id,
        }]

    def convert(self, event: WireEvent) -> list[dict[str, Any]]:
        kind = event.get("event")
        if kind == "assistant_delta":
            out: list[dict[str, Any]] = []
            if not self._text_open:
                self._text_open = True
                out.append({
                    "type": "TEXT_MESSAGE_START",
                    "messageId": self._message_id,
                    "role": "assistant",
                })
            out.append({
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": self._message_id,
                "delta": str(event.get("text", "")),
            })
            return out
        if kind == "final":
            return self.finish_run(reason=str(event.get("reason") or "natural"))
        if kind == "error":
            return self.error(str(event.get("message") or "unknown error"))
        if kind == "tool_call_started":
            tool_call_id = self._start_tool_call_id(event)
            return [
                {
                    "type": "TOOL_CALL_START",
                    "toolCallId": tool_call_id,
                    "toolCallName": str(event.get("name") or ""),
                },
                {
                    "type": "TOOL_CALL_ARGS",
                    "toolCallId": tool_call_id,
                    "delta": _compact_json(event.get("input", {})),
                },
                {
                    "type": "TOOL_CALL_END",
                    "toolCallId": tool_call_id,
                },
            ]
        if kind == "tool_call_progress":
            return [{
                "type": "CUSTOM",
                "name": "aura.tool.progress",
                "value": {
                    "toolCallId": self._active_tool_call_id(event),
                    "toolName": str(event.get("name") or ""),
                    "stream": str(event.get("stream") or ""),
                    "chunk": str(event.get("chunk") or ""),
                },
            }]
        if kind == "tool_call_completed":
            content = event.get("content") or {"text": "", "error": False}
            return [{
                "type": "TOOL_CALL_RESULT",
                "messageId": self._message_id,
                "toolCallId": self._complete_tool_call_id(event),
                "content": _compact_json(content),
                "role": "tool",
            }]
        if kind == "aura_state":
            return [{"type": "STATE_SNAPSHOT", "snapshot": event}]
        if kind == "permission_request":
            return [{
                "type": "CUSTOM",
                "name": "aura.permission.request",
                "value": event,
            }]
        if kind == "permission_audit":
            return [{
                "type": "CUSTOM",
                "name": "aura.permission.audit",
                "value": event,
            }]
        if kind == "compact_event":
            return [{
                "type": "CUSTOM",
                "name": "aura.compact.event",
                "value": event,
            }]
        if kind == "coordination":
            return [{
                "type": "CUSTOM",
                "name": _coordination_custom_event_name(event),
                "value": event,
            }]
        return [{"type": "CUSTOM", "name": "aura.event", "value": event}]

    def finish_run(self, *, reason: str = "natural") -> list[dict[str, Any]]:
        if self._run_finished:
            return []
        out = self._close_text_if_open()
        out.append({
            "type": "RUN_FINISHED",
            "runId": self.run_id,
            "threadId": self.thread_id,
            "result": {"reason": reason},
        })
        self._run_finished = True
        return out

    def error(self, message: str) -> list[dict[str, Any]]:
        if self._run_finished:
            return []
        out = self._close_text_if_open()
        out.append({
            "type": "RUN_ERROR",
            "message": message,
        })
        self._run_finished = True
        return out

    def _close_text_if_open(self) -> list[dict[str, Any]]:
        if not self._text_open:
            return []
        self._text_open = False
        return [{
            "type": "TEXT_MESSAGE_END",
            "messageId": self._message_id,
        }]

    def _start_tool_call_id(self, event: WireEvent) -> str:
        if explicit := _explicit_tool_call_id(event):
            return explicit
        name = str(event.get("name") or "tool")
        self._next_tool_fallback += 1
        fallback = f"{name}-{self._next_tool_fallback}"
        self._open_tool_fallbacks[name].append(fallback)
        return fallback

    def _active_tool_call_id(self, event: WireEvent) -> str:
        if explicit := _explicit_tool_call_id(event):
            return explicit
        name = str(event.get("name") or "tool")
        if self._open_tool_fallbacks[name]:
            return self._open_tool_fallbacks[name][-1]
        return f"{name}-unknown"

    def _complete_tool_call_id(self, event: WireEvent) -> str:
        if explicit := _explicit_tool_call_id(event):
            return explicit
        name = str(event.get("name") or "tool")
        if self._open_tool_fallbacks[name]:
            return self._open_tool_fallbacks[name].pop(0)
        return f"{name}-unknown"


def _explicit_tool_call_id(event: WireEvent) -> str | None:
    raw = event.get("id")
    if isinstance(raw, str) and raw:
        return raw
    return None


def _coordination_custom_event_name(event: WireEvent) -> str:
    family = event.get("family")
    if isinstance(family, str) and family:
        return f"aura.{family}.event"
    return "aura.coordination.event"
