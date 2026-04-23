"""Event types emitted by the agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AssistantDelta:
    """Incremental text output from the assistant."""

    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    """Tool invocation initiated."""

    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolCallProgress:
    """Incremental output chunk streamed while a tool is still running.

    Currently produced by the ``bash`` tool: each stdout/stderr chunk
    surfaces to the renderer in near-realtime so a long-running command
    doesn't sit silently for 30s+ before the user sees any output.

    ``stream`` is ``"stdout"`` or ``"stderr"``; ``chunk`` is a decoded
    UTF-8 fragment (may contain partial multibyte sequences replaced
    with U+FFFD). Chunks are NOT newline-aligned — callers that want
    one-line-per-print must buffer themselves.
    """

    name: str
    stream: Literal["stdout", "stderr"]
    chunk: str


@dataclass(frozen=True)
class ToolCallCompleted:
    """Tool invocation finished with result or error."""

    name: str
    output: Any
    error: str | None = None


@dataclass(frozen=True)
class Final:
    """Final concatenated text response from the agent loop.

    ``reason`` marks WHY the loop stopped — ``"natural"`` (model emitted no
    more tool_calls) or ``"max_turns"`` (the loop hit its turn cap). The CLI
    dims a "stopped: max turns reached" line when reason != natural.
    """

    message: str
    reason: Literal["natural", "max_turns"] = "natural"


@dataclass(frozen=True)
class PermissionAudit:
    """Dim audit line shown after a ToolCallStarted for auto-allow decisions.

    Only emitted for reasons where no prompt was shown (``rule_allow``,
    ``mode_bypass``). User-prompted decisions
    (``user_accept`` / ``user_always`` / ``user_deny``) need no audit line —
    the prompt itself was the audit.
    """

    tool: str
    text: str


AgentEvent = (
    AssistantDelta
    | ToolCallStarted
    | ToolCallProgress
    | ToolCallCompleted
    | Final
    | PermissionAudit
)
"""Type alias for all events emitted by the agent loop."""
