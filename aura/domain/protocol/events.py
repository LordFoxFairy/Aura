"""Canonical Aura protocol event payload contracts."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypeAlias, TypedDict


class AssistantDeltaEvent(TypedDict):
    event: Literal["assistant_delta"]
    text: str


class ToolCallStartedEvent(TypedDict):
    event: Literal["tool_call_started"]
    name: str
    input: dict[str, Any]
    id: NotRequired[str]


class ToolCallProgressEvent(TypedDict):
    event: Literal["tool_call_progress"]
    name: str
    stream: str
    chunk: str
    id: NotRequired[str]


class ToolCallContent(TypedDict):
    text: str
    error: bool


class ToolCallCompletedEvent(TypedDict):
    event: Literal["tool_call_completed"]
    name: str
    content: ToolCallContent
    id: NotRequired[str]


ToolCallEvent: TypeAlias = (
    ToolCallStartedEvent
    | ToolCallProgressEvent
    | ToolCallCompletedEvent
)


class PermissionRequestEvent(TypedDict):
    event: Literal["permission_request"]
    id: str
    tool: str
    args: dict[str, Any]
    rule_hint: str
    is_destructive: bool


class PermissionAuditEvent(TypedDict):
    event: Literal["permission_audit"]
    tool: str
    text: str


PermissionEvent: TypeAlias = PermissionRequestEvent | PermissionAuditEvent


class FinalEvent(TypedDict):
    event: Literal["final"]
    message: str
    reason: str


class TokenUsageSnapshot(TypedDict):
    last_input: int
    last_output: int
    last_cache_read: int
    total_input: int
    total_output: int
    total_cache_read: int
    turn_count: int


class AuraStateEvent(TypedDict):
    event: Literal["aura_state"]
    model: str
    mode: str
    cwd: str
    tokens: TokenUsageSnapshot
    pinned: int
    window: int
    last_turn_seconds: float


class CompactEvent(TypedDict):
    event: Literal["compact_event"]
    trigger: str
    tokens_before: int
    tokens_after: int
    outcome: str
    duration_ms: float


class ErrorEvent(TypedDict):
    event: Literal["error"]
    message: str


class UnknownEvent(TypedDict):
    event: Literal["unknown"]
    type: str


class SubagentProtocolEvent(TypedDict):
    """Placeholder coordination envelope for future subagent protocol traffic."""

    event: Literal["coordination"]
    family: Literal["subagent"]
    action: str
    subagent_id: NotRequired[str]
    parent_id: NotRequired[str]
    payload: NotRequired[dict[str, Any]]


class TeamProtocolEvent(TypedDict):
    """Placeholder coordination envelope for future team protocol traffic."""

    event: Literal["coordination"]
    family: Literal["team"]
    action: str
    team_id: NotRequired[str]
    member_id: NotRequired[str]
    payload: NotRequired[dict[str, Any]]


CoordinationEvent: TypeAlias = SubagentProtocolEvent | TeamProtocolEvent

CommonProtocolEvent: TypeAlias = (
    AssistantDeltaEvent
    | ToolCallEvent
    | PermissionEvent
    | FinalEvent
    | AuraStateEvent
    | CompactEvent
    | ErrorEvent
    | UnknownEvent
)

WireEvent: TypeAlias = CommonProtocolEvent | CoordinationEvent

__all__ = [
    "AssistantDeltaEvent",
    "AuraStateEvent",
    "CommonProtocolEvent",
    "CompactEvent",
    "CoordinationEvent",
    "ErrorEvent",
    "FinalEvent",
    "PermissionAuditEvent",
    "PermissionEvent",
    "PermissionRequestEvent",
    "SubagentProtocolEvent",
    "TeamProtocolEvent",
    "TokenUsageSnapshot",
    "ToolCallCompletedEvent",
    "ToolCallContent",
    "ToolCallEvent",
    "ToolCallProgressEvent",
    "ToolCallStartedEvent",
    "UnknownEvent",
    "WireEvent",
]
