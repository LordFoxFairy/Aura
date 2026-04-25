"""send_message — the LLM-facing surface for team comms.

Phase A: text + (model-facing) shutdown_request only. The tool resolves
the active team via the calling Agent's ``team`` attribute; absence
raises ``ToolError`` so the LLM sees a clear message rather than a
silent no-op.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.core.teams.types import (
    BROADCAST_RECIPIENT,
    MAX_BODY_CHARS,
    TEAM_LEADER_NAME,
)
from aura.schemas.tool import ToolError, tool_metadata

# Surface only the kinds the model is allowed to emit. ``shutdown_response``
# is internal (the runtime emits it implicitly by exiting), so we hide it
# from the schema — exposing it would let a teammate forge an "ack"
# without actually shutting down.
SendMessageKind = Literal["text", "shutdown_request"]


class SendMessageParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    to: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "Recipient member name, the literal 'leader' to message the "
            "team leader, or 'broadcast' to fan out to every active "
            "member of the team."
        ),
    )
    body: str = Field(
        min_length=1,
        max_length=MAX_BODY_CHARS,
        description=(
            "Plain-text message body. Wrap structured data in markdown / "
            "code fences; the team channel is text-only in Phase A."
        ),
    )
    kind: SendMessageKind = Field(
        default="text",
        description=(
            "Use 'text' for normal messages. 'shutdown_request' asks the "
            "recipient to exit cleanly (used by the leader to drain a "
            "teammate before remove)."
        ),
    )


def _preview(args: dict[str, Any]) -> str:
    to = args.get("to", "?")
    body = args.get("body", "")
    snippet = body[:40].replace("\n", " ")
    return f"send_message → {to}: {snippet}"


class SendMessage(BaseTool):
    """Append a message to a teammate's mailbox.

    Stateful: the calling Agent injects itself via ``__init__`` so the
    tool can resolve ``agent.team`` at invoke time. Outside a team the
    tool is still bound (the LLM keeps seeing it in the schema) but
    invocation raises a clear ToolError.
    """

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
    metadata: dict[str, Any] | None = tool_metadata(
        is_destructive=False,
        is_concurrency_safe=False,
        max_result_size_chars=400,
        args_preview=_preview,
    )

    _agent: Any = PrivateAttr()

    def __init__(self, *, agent: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # ``Agent`` reference; typed ``Any`` to dodge the Agent → tools
        # → Agent import cycle. The only attribute we touch is ``.team``
        # which is set / cleared by ``Agent.join_team`` / ``leave_team``.
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
        # Sender resolution: a teammate Agent has ``_team_member_name``
        # stamped by ``join_team``; the leader doesn't have one and is
        # the sole sender outside that. We surface "leader" explicitly
        # so message envelopes are unambiguous.
        sender = getattr(self._agent, "_team_member_name", None) or TEAM_LEADER_NAME
        # Validate ``to`` against the live membership before hitting the
        # mailbox so a typo'd name returns a usable error rather than
        # appending a message no one will ever read.
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
