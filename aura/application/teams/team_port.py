"""Structural contract a teammate Agent sees of its bound TeamManager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from aura.application.tasks.store import TasksStore
from aura.domain.team import TeamMessage, TeamMessageKind, TeamRecord
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.wire.events import CoordinationEvent


@dataclass(frozen=True)
class TeammateBinding:
    task_id: str
    tasks_store: TasksStore


@runtime_checkable
class TeamPort(Protocol):
    """Narrow team surface for send_message, the teammate runtime, and backends."""

    @property
    def is_active(self) -> bool: ...

    @property
    def team(self) -> TeamRecord | None: ...

    @property
    def storage(self) -> SessionStorage: ...

    def post_message(self, msg: TeamMessage) -> None: ...

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        body: str,
        kind: TeamMessageKind = "text",
    ) -> list[TeamMessage]: ...

    def confirm_shutdown(self, member_name: str, *, body: str = "") -> None: ...

    @property
    def pending_protocol_events(self) -> tuple[CoordinationEvent, ...]: ...

    def drain_protocol_events(self) -> list[CoordinationEvent]: ...

    async def cleanup_session_teams(self) -> None: ...
