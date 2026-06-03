"""Backend Protocol — strategy interface for spawning a teammate."""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from aura.application.session import AgentSession
from aura.application.teams.mailbox import MailboxNotifier
from aura.application.teams.team_port import TeamPort
from aura.domain.abort import AbortController
from aura.domain.team import BackendType, TeammateMember
from aura.infrastructure.persistence.storage import SessionStorage


@runtime_checkable
class BackendHandle(Protocol):
    # pane_id populated only by the pane backend; in-process returns None for uniform access.
    pane_id: str | None

    async def shutdown(self, *, timeout_sec: float = 5.0) -> bool:
        """True iff the teammate exited cooperatively; False on timeout (backend force-killed)."""
        ...

    async def force_kill(self) -> None:
        """Idempotent tear-down; no wait for cooperative exit."""
        ...

    def is_alive(self) -> bool:
        ...


@runtime_checkable
class TeammateBackend(Protocol):
    backend_type: BackendType

    async def spawn(
        self,
        *,
        team_id: str,
        member: TeammateMember,
        agent: AgentSession,
        manager: TeamPort,
        storage: SessionStorage,
        stop_event: asyncio.Event,
        abort: AbortController,
        seed_prompt: str | None,
        notifier: MailboxNotifier | None = None,
    ) -> BackendHandle:
        """Spawn the teammate; backend MAY mutate ``member.tmux_pane_id`` before returning."""
        ...


__all__ = ["BackendHandle", "TeammateBackend"]
