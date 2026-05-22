"""Event types emitted by the agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AssistantDelta:
    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    name: str
    input: dict[str, Any]
    id: str | None = None


@dataclass(frozen=True)
class ToolCallProgress:
    name: str
    stream: Literal["stdout", "stderr"]
    chunk: str
    id: str | None = None


@dataclass(frozen=True)
class ToolCallCompleted:
    name: str
    output: Any
    error: str | None = None
    id: str | None = None


@dataclass(frozen=True)
class Final:
    message: str
    reason: Literal[
        "natural", "max_turns", "aborted", "length_recovery_exhausted",
    ] = "natural"


@dataclass(frozen=True)
class PermissionAudit:
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
