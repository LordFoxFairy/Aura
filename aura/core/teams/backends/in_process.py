"""InProcessBackend — wraps :func:`run_teammate` in an asyncio task.

This is the original Aura behaviour, preserved bit-for-bit. The leader's
event loop owns the teammate task; cancellation cascades naturally
through the leader's :class:`~aura.core.abort.AbortController` chain.

Lifecycle map:

==========================  ========================================
Manager call                Backend translation
==========================  ========================================
``add_member`` / ``spawn``  ``asyncio.create_task(run_teammate(...))``
``aremove_member`` graceful ``stop_event.set()`` + wait for task done
``remove_member`` force     ``abort.abort()`` + ``task.cancel()``
``cleanup_session_teams``   ``force_kill`` then ``rm -rf <team_dir>``
==========================  ========================================
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aura.core.teams.backends.types import BackendHandle
from aura.core.teams.runtime import run_teammate

if TYPE_CHECKING:
    from aura.core.abort import AbortController
    from aura.core.agent import Agent
    from aura.core.persistence.storage import SessionStorage
    from aura.core.teams.manager import TeamManager
    from aura.core.teams.types import BackendType, TeammateMember


@dataclass
class InProcessHandle(BackendHandle):
    """Handle wrapping the asyncio.Task running ``run_teammate``.

    ``stop_event`` and ``abort`` are the same instances installed on the
    runtime — graceful shutdown sets the event; force-kill aborts and
    cancels the task directly.
    """

    task: asyncio.Task[None]
    stop_event: asyncio.Event
    abort: AbortController
    pane_id: str | None = None  # always None for in-process

    async def shutdown(self, *, timeout_sec: float = 5.0) -> bool:
        """Cooperative stop: fire ``stop_event``, await task.

        Returns ``True`` on natural exit within ``timeout_sec``;
        ``False`` after force-cancelling on timeout. Idempotent.
        """
        if self.task.done():
            return True
        self.stop_event.set()
        try:
            await asyncio.wait_for(
                asyncio.shield(self.task), timeout=timeout_sec,
            )
            return True
        except TimeoutError:
            await self.force_kill()
            return False
        except asyncio.CancelledError:
            return self.task.done()

    async def force_kill(self) -> None:
        """Abort + cancel; safe on already-completed tasks."""
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
    """Singleton in-process backend.

    Stateless — the spawn args carry every piece of context the runtime
    needs. The manager holds one shared instance via
    :func:`~aura.core.teams.backends.registry.get_backend`.
    """

    backend_type: BackendType = "in_process"

    async def spawn(
        self,
        *,
        team_id: str,
        member: TeammateMember,
        agent: Agent,
        manager: TeamManager,
        storage: SessionStorage,
        stop_event: asyncio.Event,
        abort: AbortController,
        seed_prompt: str | None,
    ) -> InProcessHandle:
        """Schedule ``run_teammate`` and return a wired-up handle.

        Async to match the :class:`TeammateBackend` Protocol; performs
        no I/O so the coroutine completes synchronously the moment
        the event loop steps it. Sync callers should use
        :meth:`spawn_sync` to avoid the await ceremony.
        """
        return self.spawn_sync(
            team_id=team_id,
            member=member,
            agent=agent,
            manager=manager,
            storage=storage,
            stop_event=stop_event,
            abort=abort,
            seed_prompt=seed_prompt,
        )

    def spawn_sync(
        self,
        *,
        team_id: str,
        member: TeammateMember,
        agent: Agent,
        manager: TeamManager,
        storage: SessionStorage,
        stop_event: asyncio.Event,
        abort: AbortController,
        seed_prompt: str | None,
    ) -> InProcessHandle:
        """Synchronous spawn — wraps ``asyncio.create_task`` directly.

        Used by :meth:`TeamManager.add_member` (sync entry) so we don't
        have to drive an async-no-op coroutine from inside an already-
        running event loop. Same return shape as :meth:`spawn`.
        """
        # ``manager`` is unused here — the runtime reaches the manager
        # back through ``agent.team`` if it needs to send shutdown_response.
        # Accepting it keeps the Protocol uniform across backends.
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
            ),
            name=f"aura-teammate-{member.name}",
        )
        return InProcessHandle(
            task=task,
            stop_event=stop_event,
            abort=abort,
        )


__all__ = ["InProcessBackend", "InProcessHandle"]
