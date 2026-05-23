"""Team state models. Body cap = 4 KB to stay under POSIX write(2) atomicity."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_BODY_CHARS: int = 4_000
MAX_MEMBERS: int = 16
TEAM_LEADER_NAME: str = "leader"
BROADCAST_RECIPIENT: str = "broadcast"

TeamMessageKind = Literal["text", "shutdown_request", "shutdown_response"]

# in_process = asyncio task on leader loop; pane = tmux subprocess.
BackendType = Literal["in_process", "pane"]


class TeammateMember(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    agent_type: str = "general-purpose"
    system_prompt: str | None = None
    model_name: str | None = None
    created_at: float = Field(default_factory=time.time)
    is_active: bool = True
    backend_type: BackendType = "in_process"
    # %<int> from tmux for pane backend; None for in_process.
    tmux_pane_id: str | None = None


class TeamRecord(BaseModel):
    # cwd captured at create-time so later-spawned teammates resolve memory
    # and rules from the same root the leader saw.
    model_config = ConfigDict(extra="forbid")

    team_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=64)
    leader_session_id: str
    members: list[TeammateMember] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    cwd: str = "."


class TeamMessage(BaseModel):
    # recipient="broadcast" is fanned out by the manager at send time.
    model_config = ConfigDict(extra="forbid")

    msg_id: str
    sender: str
    recipient: str
    body: str = Field(min_length=1, max_length=MAX_BODY_CHARS)
    kind: TeamMessageKind = "text"
    sent_at: float = Field(default_factory=time.time)
