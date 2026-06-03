"""Frozen DI container for stateful tools — leaf, no tool-factory deps."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from aura.application.loop_state import LoopState
from aura.application.tasks.spawn_port import SpawnPort
from aura.application.tasks.store import TasksStore
from aura.application.teams.team_port import TeamPort
from aura.infrastructure.persistence.storage import SessionStorage
from aura.tools.ask_user import UserAsker


@dataclass(frozen=True)
class ToolRuntime:
    """Frozen DI container; factories pick the fields they need."""

    state: LoopState
    asker: UserAsker | None = None
    tasks_store: TasksStore | None = None
    spawner: SpawnPort | None = None
    running_tasks: dict[str, asyncio.Task[None]] | None = None
    running_shells: dict[str, asyncio.subprocess.Process] | None = None
    transcript_storage: SessionStorage | None = None
    # Providers (not snapshots): ToolRuntime is built once; team binding is set later by join_team.
    team_provider: Callable[[], TeamPort | None] | None = None
    member_name_provider: Callable[[], str | None] | None = None
