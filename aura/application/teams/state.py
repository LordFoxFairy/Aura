"""Per-name runtime handles for one teammate, shared by manager and its collaborators."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from aura.application.session import AgentSession


class TeamError(ValueError):
    pass


@runtime_checkable
class BackendHandleLike(Protocol):
    async def force_kill(self) -> None: ...


@dataclass
class Member:
    # lifecycle_state outlives the runtime handles: set before spawn, retained
    # past teardown ("terminated") until the team is cleared.
    lifecycle_state: str | None = None
    task_id: str | None = None
    agent: AgentSession | None = None
    stop_event: asyncio.Event | None = None
    backend: BackendHandleLike | None = None
    shutdown_ack: asyncio.Future[bool] | None = None
    shutdown_waiter: asyncio.Task[bool] | None = None
