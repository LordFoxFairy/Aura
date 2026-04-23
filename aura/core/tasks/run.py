"""run_task — the subagent dispatch core.

Invariants:

- Designed to be scheduled via ``asyncio.create_task`` and NEVER awaited
  by the tool that spawned it (fire-and-forget). The ``task_create`` tool
  records the handle so the parent Agent can cancel it on close.
- Exceptions bubbling out of the subagent are caught and written to the
  record's ``error`` field; they do NOT propagate to the parent's loop.
- ``CancelledError`` is the exception that DOES propagate — it's the
  parent telling us "stop now." We mark the record cancelled first so
  ``/tasks`` reflects reality, then re-raise.
"""

from __future__ import annotations

import asyncio

from aura.core.persistence import journal
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.store import TasksStore
from aura.schemas.events import Final, ToolCallStarted


async def run_task(
    store: TasksStore,
    factory: SubagentFactory,
    task_id: str,
) -> None:
    record = store.get(task_id)
    if record is None:
        return
    # Bug fix (integration test): ``factory.spawn(...)`` USED to live
    # outside the try/except, so if spawn itself raised (e.g. ValueError
    # from ``agent_type="explore"`` requiring a tool the parent hadn't
    # enabled), ``mark_failed`` was never called and the TaskRecord
    # stayed permanently in ``running`` status. Move spawn into the try;
    # guard ``agent.close()`` with a None-check so the except blocks
    # don't trip over an unbound name when spawn itself failed.
    agent = None
    try:
        agent = factory.spawn(
            record.prompt,
            agent_type=record.agent_type or "general-purpose",
        )
        final_text = ""
        async for event in agent.astream(record.prompt):
            if isinstance(event, ToolCallStarted):
                # Progress is for the PARENT's ``task_get`` polling —
                # cheap write, stays in parent's event loop, no cross-process
                # sync. Child tool invocations are the most meaningful unit
                # of "what is this subagent doing right now".
                store.record_activity(task_id, event.name)
            elif isinstance(event, Final):
                final_text = event.message
        store.mark_completed(task_id, final_text)
    except asyncio.CancelledError:
        store.mark_cancelled(task_id)
        journal.write("subagent_cancelled", task_id=task_id)
        if agent is not None:
            agent.close()
        raise
    except Exception as exc:  # noqa: BLE001
        store.mark_failed(task_id, f"{type(exc).__name__}: {exc}")
        journal.write(
            "subagent_failed",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        if agent is not None:
            agent.close()
    else:
        journal.write("subagent_completed", task_id=task_id)
        if agent is not None:
            agent.close()
