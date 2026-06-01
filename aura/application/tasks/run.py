"""Fire-and-forget driver for one :class:`LocalAgentTask`."""

from __future__ import annotations

from aura.application.tasks.runners.local_agent import (
    DEFAULT_SUBAGENT_TIMEOUT_SEC,
    LocalAgentTask,
    run_local_agent,
)
from aura.application.tasks.spawn_port import SpawnPort
from aura.application.tasks.store import TasksStore
from aura.infrastructure.persistence.storage import SessionStorage


async def run_task(
    store: TasksStore,
    factory: SpawnPort,
    task_id: str,
    *,
    timeout_sec: float | None = None,
    transcript_storage: SessionStorage | None = None,
    summary_interval_sec: float | None = None,
    parent_session_id: str | None = None,
    cwd: str | None = None,
) -> None:
    """Drive one LocalAgentTask to terminal."""
    await run_local_agent(
        store=store,
        factory=factory,
        task_id=task_id,
        timeout_sec=timeout_sec,
        transcript_storage=transcript_storage,
        summary_interval_sec=summary_interval_sec,
        parent_session_id=parent_session_id,
        cwd=cwd,
    )


__all__ = [
    "DEFAULT_SUBAGENT_TIMEOUT_SEC",
    "LocalAgentTask",
    "run_task",
]
