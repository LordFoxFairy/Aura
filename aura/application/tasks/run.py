"""Shared subagent driver — thin entry point on top of the runner package.

Historically this module owned the entire subagent run loop (744 lines
of lifecycle plumbing for the local-agent path). After the runner
split it is reduced to a façade that decides which runner topology to
dispatch to. The actual lifecycle implementation now lives in
:mod:`aura.application.tasks.runners.local_agent` (LocalAgentTask),
:mod:`aura.application.teams.runtime` (InProcessTeammateTask), and
:mod:`aura.application.tasks.runners.remote` (RemoteAgentTask).

:func:`run_task` is the entry point Aura code paths call; it delegates
to the in-process LocalAgentTask path. The function-shape is preserved
so existing callers (``asyncio.create_task(run_task(...))``) continue
working without restructuring; new code can use
:class:`LocalAgentTask` directly for the explicit
``start/abort/wait_for_terminal`` interface.

Invariants (unchanged from the pre-split contract):

- Designed to be scheduled via ``asyncio.create_task`` and NEVER
  awaited by the spawning tool (fire-and-forget). ``task_create``
  records the handle so ``Agent.close()`` can cancel it.
- Exceptions bubbling out of the subagent are caught and written to
  the record's ``error`` field; they do NOT propagate to the parent's
  loop.
- ``CancelledError`` is the exception that DOES propagate — the parent
  asked us to stop. The record flips to ``cancelled`` first so
  ``/tasks`` reflects reality, then we re-raise.

Wall-clock timeout: every local-agent run is wrapped in
``asyncio.timeout`` with a defense-in-depth ceiling so a stalled
model / pathological tool / infinite small-sleep loop can't strand a
record in ``running`` forever. Default is
:data:`DEFAULT_SUBAGENT_TIMEOUT_SEC` (5 minutes); override via the
``AURA_SUBAGENT_TIMEOUT_SEC`` env var (``<= 0`` disables the cap).
"""

from __future__ import annotations

from aura.application.tasks.factory import SubagentFactory
from aura.application.tasks.runners.local_agent import (
    DEFAULT_SUBAGENT_TIMEOUT_SEC,
    LocalAgentTask,
    _run_local_agent,
)
from aura.application.tasks.store import TasksStore
from aura.infrastructure.persistence.storage import SessionStorage


async def run_task(
    store: TasksStore,
    factory: SubagentFactory,
    task_id: str,
    *,
    timeout_sec: float | None = None,
    transcript_storage: SessionStorage | None = None,
    summary_interval_sec: float | None = None,
    parent_session_id: str | None = None,
    cwd: str | None = None,
) -> None:
    """Drive one LocalAgentTask to terminal.

    Thin shim over :func:`_run_local_agent` — kept as a free function
    (not a method on :class:`LocalAgentTask`) so callers that
    ``asyncio.create_task(run_task(...))`` directly keep working
    without restructuring. The class-based shape exists for tests and
    new code that wants the explicit ``start/abort/wait`` interface.
    """
    await _run_local_agent(
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
