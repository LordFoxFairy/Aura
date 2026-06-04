"""InProcessBackend — runs :func:`run_teammate` as an asyncio task on the leader's loop."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

from aura.application.session import AgentSession
from aura.application.teams.mailbox import MailboxNotifier
from aura.application.teams.runtime import run_teammate
from aura.application.teams.team_port import TeamPort
from aura.domain.abort import AbortController
from aura.domain.team import BackendType, TeammateMember
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.teams.types import BackendHandle


@dataclass
class InProcessHandle(BackendHandle):
    task: asyncio.Task[None]
    # Same instances installed on the runtime — graceful sets the event, force aborts.
    stop_event: asyncio.Event
    abort: AbortController
    pane_id: str | None = None  # always None for in-process

    async def shutdown(self, *, timeout_sec: float = 5.0) -> bool:
        """Cooperative stop; returns True on natural exit, False after force-cancel on timeout."""
        if self.task.done():
            return True
        self.stop_event.set()
        try:
            await asyncio.wait_for(
                asyncio.shield(self.task),
                timeout=timeout_sec,
            )
            return True
        except TimeoutError:
            await self.force_kill()
            return False
        except asyncio.CancelledError:
            return self.task.done()

    async def force_kill(self) -> None:
        if not self.abort.aborted:
            with contextlib.suppress(Exception):
                self.abort.abort("force_kill")
        if not self.task.done():
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.task

    def is_alive(self) -> bool:
        return not self.task.done()


class InProcessBackend:
    backend_type: BackendType = "in_process"

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
    ) -> InProcessHandle:
        # Async only to satisfy the Protocol; no I/O happens here.
        return self.spawn_sync(
            team_id=team_id,
            member=member,
            agent=agent,
            manager=manager,
            storage=storage,
            stop_event=stop_event,
            abort=abort,
            seed_prompt=seed_prompt,
            notifier=notifier,
        )

    def spawn_sync(
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
    ) -> InProcessHandle:
        """Synchronous variant for sync callers; ``notifier=None`` falls back to filesystem-poll."""
        # Accepted for Protocol uniformity; the runtime reaches the manager via agent.team.
        del manager
        task: asyncio.Task[None] = asyncio.create_task(
            run_teammate(
                agent=agent,
                team_id=team_id,
                member_name=member.name,
                storage=storage,
                stop_event=stop_event,
                abort=abort,
                seed_prompt=seed_prompt,
                notifier=notifier,
            ),
            name=f"aura-teammate-{member.name}",
        )
        return InProcessHandle(
            task=task,
            stop_event=stop_event,
            abort=abort,
        )


__all__ = ["InProcessBackend", "InProcessHandle"]
