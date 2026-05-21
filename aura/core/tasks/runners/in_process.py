"""InProcessTeammateTask — in-process teammate runner with shared mailbox.

Translates claude-code's ``tasks/InProcessTeammateTask`` into Python.
The runner adapts :func:`aura.core.teams.runtime.run_teammate` (the
long-lived mailbox-driven loop) into the same ``start() /
wait_for_terminal() / abort()`` shape the other runners expose.

This is deliberately a thin wrapper — the heavy lifting lives in
:mod:`aura.core.teams.runtime` because the in-process teammate model
predates the runner abstraction. Phase B may invert the dependency
(runtime extracted out of teams/ into here); today the runtime stays
where every team-aware consumer already imports it from.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from aura.core.abort import AbortController
from aura.core.persistence.storage import SessionStorage

if TYPE_CHECKING:
    from aura.core.agent import Agent


class InProcessTeammateTask:
    """Long-lived in-process teammate, joined to a team mailbox.

    Unlike :class:`LocalAgentTask` (one prompt, one terminal outcome),
    a teammate task runs until either:

    - ``stop_event`` is set (graceful — exits at the next mailbox poll
      boundary or after the current turn finishes).
    - ``abort`` fires (cascade from leader — short-circuits via
      :class:`AbortException` through ``Agent.astream``).
    - The runtime consumes a ``shutdown_request`` mailbox message.
    """

    def __init__(
        self,
        *,
        agent: Agent,
        team_id: str,
        member_name: str,
        storage: SessionStorage,
        stop_event: asyncio.Event | None = None,
        abort: AbortController | None = None,
        seed_prompt: str | None = None,
    ) -> None:
        self._agent = agent
        self._team_id = team_id
        self._member_name = member_name
        self._storage = storage
        self._stop_event = stop_event or asyncio.Event()
        self._abort = abort or AbortController()
        self._seed_prompt = seed_prompt
        self._task: asyncio.Task[None] | None = None

    def start(self) -> asyncio.Task[None]:
        """Schedule the teammate runtime loop on the current event loop."""
        if self._task is None:
            # Lazy import to avoid the import cycle: runtime.py imports
            # mailbox + teams.types, and this module is itself reachable
            # from teams.manager through the runners package.
            from aura.core.teams.runtime import run_teammate

            self._task = asyncio.create_task(
                run_teammate(
                    agent=self._agent,
                    team_id=self._team_id,
                    member_name=self._member_name,
                    storage=self._storage,
                    stop_event=self._stop_event,
                    abort=self._abort,
                    seed_prompt=self._seed_prompt,
                ),
                name=f"aura-teammate-{self._member_name}",
            )
        return self._task

    async def wait_for_terminal(self) -> None:
        """Await the runtime task. No-op if never started or already done."""
        if self._task is None:
            return
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    def abort(self) -> None:
        """Graceful shutdown: set stop_event AND fire abort controller.

        The runtime's main loop polls ``stop_event`` between mailbox
        slices, so a clean stop arrives within :data:`_POLL_SLICE_SEC`.
        Firing the AbortController ALSO cuts a turn that's already in
        flight — without it the loop would happily finish the current
        ``astream`` even after the operator asked to stop.
        """
        self._stop_event.set()
        if not self._abort.aborted:
            self._abort.abort("teammate_runner_abort")

    @property
    def stop_event(self) -> asyncio.Event:
        return self._stop_event

    @property
    def abort_controller(self) -> AbortController:
        return self._abort

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task


__all__ = ["InProcessTeammateTask"]
