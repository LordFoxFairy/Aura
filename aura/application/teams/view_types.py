"""Pure read-model snapshots shared between TeamManager and the ``/team`` surface."""

from __future__ import annotations

from dataclasses import dataclass

from aura.domain.team import TeamMessage


@dataclass(frozen=True)
class TeammateMemberStatus:
    name: str
    agent_type: str
    model_spec: str | None
    status: str
    tokens_used: int
    last_active: float | None
    lifecycle_state: str


@dataclass(frozen=True)
class TeamViewSnapshot:
    team_id: str
    name: str
    members: list[TeammateMemberStatus]
    recent_messages: list[TeamMessage]
    subagent_count: int
    transcript_count: int
