"""Pydantic models for team state.

Persistence shape mirrors the design proposal §3.2: a ``TeamRecord`` is the
on-disk source of truth (``config.json``), a ``TeammateMember`` describes one
in-process teammate, a ``TeamMessage`` is one mailbox JSONL line.

We use ``BaseModel`` (not frozen dataclass) so JSON round-tripping picks up
field-level validation for free — bodies cap at 4 KB to keep POSIX
``write(2)`` append below the 512-byte/4 KB atomicity floor across kernels.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: Phase A constants — kept here so both manager and tool see the same caps.
#: 4000 chars is conservative: typical TeamMessage line < 1500 bytes including
#: envelopes; cap covers most real model outputs without forcing the LLM to
#: write tiny replies. A larger body is split across multiple sends or routed
#: through a shared file (Phase B).
MAX_BODY_CHARS: int = 4_000
MAX_MEMBERS: int = 16
TEAM_LEADER_NAME: str = "leader"
BROADCAST_RECIPIENT: str = "broadcast"

TeamMessageKind = Literal["text", "shutdown_request", "shutdown_response"]

#: Backend identifier for a teammate's runtime. ``"in_process"`` runs the
#: teammate as an asyncio task in the leader's process (default; matches
#: claude-code's ``InProcessBackend``). ``"pane"`` spawns the teammate as
#: a separate subprocess inside a tmux pane via ``tmux split-window``,
#: enabling the visual "switch into team" UX (claude-code's
#: ``PaneBackend``).
BackendType = Literal["in_process", "pane"]


class TeammateMember(BaseModel):
    """One member entry inside :class:`TeamRecord`.

    ``name`` is the unique slug within the team — the recipient string callers
    use in :class:`SendMessage`. ``agent_type`` selects the flavor (matches
    :mod:`aura.core.tasks.agent_types`). ``model_name`` overrides the leader's
    model for this teammate; ``None`` inherits.

    ``backend_type`` selects the runtime strategy (default ``"in_process"``).
    ``tmux_pane_id`` is populated by the pane backend at spawn time
    (``%<int>`` from ``tmux split-window -P -F '#{pane_id}'``); ``None``
    for in-process members.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    agent_type: str = "general-purpose"
    system_prompt: str | None = None
    model_name: str | None = None
    created_at: float = Field(default_factory=time.time)
    is_active: bool = True
    backend_type: BackendType = "in_process"
    tmux_pane_id: str | None = None


class TeamRecord(BaseModel):
    """Top-level on-disk state for one team.

    ``leader_session_id`` is the parent Agent's ``session_id`` — the team
    is owned by exactly one session and dies when that session exits
    (Phase A invariant). ``cwd`` is captured at create time so a teammate
    spawned later resolves memory / rules from the same root the leader
    saw at creation.
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

    ``msg_id`` is uuid4 hex — used by the ``.seen`` cursor to mark consumed.
    ``sender`` is the member name OR ``"leader"``. ``recipient`` is a member
    name OR ``"broadcast"``; the manager fans broadcast out to every active
    member's mailbox at send time.
    """

    model_config = ConfigDict(extra="forbid")

    msg_id: str
    sender: str
    recipient: str
    body: str = Field(min_length=1, max_length=MAX_BODY_CHARS)
    kind: TeamMessageKind = "text"
    sent_at: float = Field(default_factory=time.time)
