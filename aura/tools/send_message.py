"""send_message — append a message to a teammate's mailbox."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.domain.team import (
    BROADCAST_RECIPIENT,
    MAX_BODY_CHARS,
    TEAM_LEADER_NAME,
)
from aura.schemas.tool import ToolError, ToolMetadata

SendMessageKind = Literal["text", "shutdown_request"]


class SendMessageParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    to: str = Field(
        min_length=1,
        max_length=64,
        description="Recipient member name, 'leader', or 'broadcast'.",
    )
    body: str = Field(
        min_length=1,
        max_length=MAX_BODY_CHARS,
        description="Plain-text message body.",
    )
    kind: SendMessageKind = Field(
        default="text",
        description="'text' for normal messages; 'shutdown_request' asks recipient to exit.",
    )


def _preview(args: dict[str, Any]) -> str:
    to = args.get("to", "?")
    body = args.get("body", "")
    snippet = body[:40].replace("\n", " ")
    return f"send_message → {to}: {snippet}"


class SendMessage(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "send_message"
    description: str = (
        "Send a message to another teammate, the team leader, or broadcast "
        "to every active member. Returns immediately after the message is "
        "appended to the recipient's mailbox; the recipient consumes it on "
        "their next loop iteration. Only available inside a team — outside "
        "a team this tool errors."
    )
    args_schema: type[BaseModel] = SendMessageParams
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=False,
        is_concurrency_safe=False,
        rule_matcher=None,
        args_preview=_preview,
        timeout_sec=None,
    )

    _agent: Any = PrivateAttr()

    def __init__(self, *, agent: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agent = agent

    def _run(
        self, to: str, body: str, kind: SendMessageKind = "text",
    ) -> dict[str, Any]:
        raise NotImplementedError("send_message is async-only; use ainvoke")

    async def _arun(
        self, to: str, body: str, kind: SendMessageKind = "text",
    ) -> dict[str, Any]:
        manager = getattr(self._agent, "team", None)
        if manager is None or not getattr(manager, "is_active", False):
            raise ToolError(
                "send_message: the calling agent is not in a team. "
                "Create a team via /team create first.",
            )
        sender = getattr(self._agent, "_team_member_name", None) or TEAM_LEADER_NAME
        record = manager.team
        valid_names: set[str] = {TEAM_LEADER_NAME, BROADCAST_RECIPIENT}
        if record is not None:
            valid_names.update(m.name for m in record.members)
        if to not in valid_names:
            raise ToolError(
                f"send_message: unknown recipient {to!r}; "
                f"valid: {sorted(valid_names)}",
            )
        try:
            sent = manager.send(
                sender=sender, recipient=to, body=body, kind=kind,
            )
        except ValueError as exc:
            raise ToolError(f"send_message: {exc}") from exc
        first = sent[0]
        return {
            "msg_id": first.msg_id,
            "recipient": to,
            "sender": sender,
            "sent_at": first.sent_at,
            "fanout": len(sent),
        }
