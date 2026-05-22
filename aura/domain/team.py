"""Pydantic models for team state.

``TeamRecord`` is the on-disk source of truth (``config.json``);
``TeammateMember`` describes one teammate; ``TeamMessage`` is one mailbox
JSONL line. Bodies cap at 4 KB so the POSIX append stays within the
write(2) atomicity floor across kernels.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_BODY_CHARS: int = 4_000
MAX_MEMBERS: int = 16
TEAM_LEADER_NAME: str = "leader"
BROADCAST_RECIPIENT: str = "broadcast"

TeamMessageKind = Literal["text", "shutdown_request", "shutdown_response"]

# ``"in_process"`` = asyncio task on the leader's loop;
# ``"pane"`` = subprocess inside a tmux pane (requires $TMUX + tmux on PATH).
BackendType = Literal["in_process", "pane"]


class TeammateMember(BaseModel):
    """One member entry inside :class:`TeamRecord`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    agent_type: str = "general-purpose"
    system_prompt: str | None = None
    model_name: str | None = None
    created_at: float = Field(default_factory=time.time)
    is_active: bool = True
    backend_type: BackendType = "in_process"
    # Populated by the pane backend at spawn (``%<int>`` from tmux);
    # ``None`` for in-process members.
    tmux_pane_id: str | None = None


class TeamRecord(BaseModel):
    """Top-level on-disk state for one team.

    ``cwd`` is captured at create time so a teammate spawned later
    resolves memory / rules from the same root the leader saw at creation.
    """

    model_config = ConfigDict(extra="forbid")

    team_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=64)
    leader_session_id: str
    members: list[TeammateMember] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    cwd: str = "."


class TeamMessage(BaseModel):
    """One mailbox JSONL line.

    ``msg_id`` uniquely identifies the message in the ``.seen`` cursor.
    ``recipient`` may be ``"broadcast"``; the manager fans that out at send.
    """

    model_config = ConfigDict(extra="forbid")

    msg_id: str
    sender: str
    recipient: str
    body: str = Field(min_length=1, max_length=MAX_BODY_CHARS)
    kind: TeamMessageKind = "text"
    sent_at: float = Field(default_factory=time.time)
