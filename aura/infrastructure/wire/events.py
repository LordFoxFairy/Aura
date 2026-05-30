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
    # On error: output is the error message string and error=True.
    output: Any
    error: bool


class ToolCallCompletedEvent(TypedDict):
    event: Literal["tool_call_completed"]
    name: str
    content: ToolCallContent
    id: NotRequired[str]


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


class FinalEvent(TypedDict):
    event: Literal["final"]
    message: str
    reason: str


class AuraStateEvent(TypedDict):
    event: Literal["aura_state"]
    model: str
    mode: str
    cwd: str
    tokens: dict[str, int]
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


class _SubagentPayload(TypedDict):
    task_id: str
    status: Literal["completed", "failed", "cancelled"]
    summary: str | None
    description: str
    terminal: Literal[True]


class _SubagentStartedPayload(TypedDict):
    task_id: str
    description: str
    parent_session_id: str
    started_at: float


class _SubagentProgressPayload(TypedDict):
    task_id: str
    tool_name: str
    activity_count: int


class TeamMessagePayload(TypedDict):
    msg_id: str
    sender: str
    recipient: str
    body: str
    kind: Literal["text", "shutdown_request", "shutdown_response"]
    sent_at: float


class LifecyclePayload(TypedDict):
    state: str
    previous_state: NotRequired[str]
    reason: NotRequired[str]


CoordinationPayload: TypeAlias = (
    _SubagentPayload
    | _SubagentStartedPayload
    | _SubagentProgressPayload
    | TeamMessagePayload
    | LifecyclePayload
)


class CoordinationEvent(TypedDict):
    event: Literal["coordination"]
    family: Literal["subagent", "team"]
    action: Literal[
        "task_started",
        "task_progress",
        "task_notification",
        "message_sent",
        "control_sent",
        "team_lifecycle",
        "member_lifecycle",
    ]
    payload: CoordinationPayload
    subagent_id: NotRequired[str]
    parent_id: NotRequired[str]
    team_id: NotRequired[str]
    member_id: NotRequired[str]


WireEvent: TypeAlias = (
    AssistantDeltaEvent
    | ToolCallStartedEvent
    | ToolCallProgressEvent
    | ToolCallCompletedEvent
    | PermissionRequestEvent
    | PermissionAuditEvent
    | FinalEvent
    | AuraStateEvent
    | CompactEvent
    | ErrorEvent
    | UnknownEvent
    | CoordinationEvent
)
