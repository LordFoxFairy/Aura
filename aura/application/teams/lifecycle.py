"""Coordination-event queue and team/member lifecycle-state emission for TeamManager."""

from __future__ import annotations

from collections.abc import Callable

from aura.application.teams.state import Member
from aura.domain.team import TeamRecord
from aura.infrastructure.wire.events import CoordinationEvent, LifecyclePayload


class LifecycleEmitter:
    def __init__(
        self,
        *,
        team: Callable[[], TeamRecord | None],
        members: dict[str, Member],
    ) -> None:
        self._team = team
        self._members = members
        self._pending: list[CoordinationEvent] = []
        self._team_lifecycle_state: str | None = None

    @property
    def pending(self) -> tuple[CoordinationEvent, ...]:
        return tuple(self._pending)

    def reset_team_state(self) -> None:
        self._team_lifecycle_state = None

    def append(self, event: CoordinationEvent) -> None:
        self._pending.append(event)

    def drain(self) -> list[CoordinationEvent]:
        drained = list(self._pending)
        self._pending.clear()
        return drained

    def emit_team(self, state: str) -> None:
        team = self._team()
        if team is None:
            return
        payload: LifecyclePayload = {"state": state}
        if self._team_lifecycle_state is not None:
            payload["previous_state"] = self._team_lifecycle_state
        self._pending.append({
            "event": "coordination",
            "family": "team",
            "action": "team_lifecycle",
            "team_id": team.team_id,
            "payload": payload,
        })
        self._team_lifecycle_state = state

    def emit_member(
        self, name: str, state: str, *, reason: str | None = None,
    ) -> None:
        team = self._team()
        if team is None:
            return
        payload: LifecyclePayload = {"state": state}
        member = self._members.get(name)
        previous = None if member is None else member.lifecycle_state
        if previous is not None:
            payload["previous_state"] = previous
        if reason is not None:
            payload["reason"] = reason
        self._pending.append({
            "event": "coordination",
            "family": "team",
            "action": "member_lifecycle",
            "team_id": team.team_id,
            "member_id": name,
            "payload": payload,
        })
        if member is None:
            self._members[name] = Member(lifecycle_state=state)
        else:
            member.lifecycle_state = state
