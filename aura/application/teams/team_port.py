"""Structural contract a teammate Agent sees of its bound TeamManager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from aura.application.tasks.store import TasksStore
from aura.domain.team import TeamMessage, TeamMessageKind, TeamRecord


@dataclass(frozen=True)
class TeammateBinding:
    task_id: str
    tasks_store: TasksStore


@runtime_checkable
class TeamPort(Protocol):
    """Narrow team surface used by send_message and the teammate runtime."""

    @property
    def is_active(self) -> bool: ...

    @property
    def team(self) -> TeamRecord | None: ...

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        body: str,
        kind: TeamMessageKind = "text",
    ) -> list[TeamMessage]: ...

    def confirm_shutdown(self, member_name: str, *, body: str = "") -> None: ...
