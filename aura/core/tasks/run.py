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
from aura.schemas.events import Final


async def run_task(
    store: TasksStore,
    factory: SubagentFactory,
    task_id: str,
) -> None:
    record = store.get(task_id)
    if record is None:
        return
    agent = factory.spawn(record.prompt)
    try:
        final_text = ""
        async for event in agent.astream(record.prompt):
            if isinstance(event, Final):
                final_text = event.message
        store.mark_completed(task_id, final_text)
    except asyncio.CancelledError:
        store.mark_cancelled(task_id)
        journal.write("subagent_cancelled", task_id=task_id)
        agent.close()
        raise
    except Exception as exc:  # noqa: BLE001
        store.mark_failed(task_id, f"{type(exc).__name__}: {exc}")
        journal.write(
            "subagent_failed",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        agent.close()
    else:
        journal.write("subagent_completed", task_id=task_id)
        agent.close()
