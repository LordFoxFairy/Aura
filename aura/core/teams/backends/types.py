"""Backend Protocol — strategy interface for spawning a teammate.

Two implementations satisfy this Protocol:

- :class:`~aura.core.teams.backends.in_process.InProcessBackend` — wraps
  ``run_teammate`` in an asyncio task on the leader's loop.
- :class:`~aura.core.teams.backends.pane.PaneBackend` — splits a tmux
  pane and runs ``python -m aura.cli.teammate_entrypoint`` inside it.

Designed so the manager can dispatch on ``member.backend_type`` without
caring which implementation is in use. Mirrors claude-code's
``backends/registry.ts`` Strategy pattern; the Protocol here lets us
swap fakes in unit tests without inheritance.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aura.core.abort import AbortController
    from aura.core.agent import Agent
    from aura.core.persistence.storage import SessionStorage
    from aura.core.teams.manager import TeamManager
    from aura.core.teams.types import BackendType, TeammateMember


@runtime_checkable
class BackendHandle(Protocol):
    """Process-shaped handle returned by :meth:`TeammateBackend.spawn`.

    The manager stores one handle per member and uses it for graceful /
    forced shutdown. ``pane_id`` is populated only by the pane backend;
    in-process handles return ``None`` so the same accessor is safe to
    read regardless of strategy.
    """

    pane_id: str | None

    async def shutdown(self, *, timeout_sec: float = 5.0) -> bool:
        """Graceful shutdown.

        Returns ``True`` when the teammate exited cooperatively within
        ``timeout_sec``, ``False`` when the wait timed out and the
        backend force-killed instead.
        """
        ...

    async def force_kill(self) -> None:
        """Tear the teammate down without waiting for cooperative exit.

        Idempotent — calling on an already-dead handle is a no-op.
        """
        ...

    def is_alive(self) -> bool:
        """Cheap liveness probe — does NOT block."""
        ...


@runtime_checkable
class TeammateBackend(Protocol):
    """Strategy for running a teammate's lifecycle.

    Each backend is a singleton (see
    :mod:`aura.core.teams.backends.registry`); ``backend_type`` is a
    class-level constant matching the
    :data:`~aura.core.teams.types.BackendType` literal the registry
    keys on.
    """

    backend_type: BackendType

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
    ) -> BackendHandle:
        """Spawn the teammate; return a :class:`BackendHandle`.

        The handle is stored on the manager and used for shutdown
        later. Backends MAY mutate ``member.tmux_pane_id`` (or other
        backend-specific bookkeeping) BEFORE returning so the manager
        can persist a complete record on its next ``_persist`` call —
        but the canonical place for that field is the model itself.
        """
        ...


__all__ = ["BackendHandle", "TeammateBackend"]
