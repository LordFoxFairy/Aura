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

from aura.core.hooks import HookChain
from aura.core.persistence import journal
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.store import TasksStore
from aura.schemas.events import Final, ToolCallStarted


async def _fire_post_subagent(
    hooks: HookChain | None,
    *,
    task_id: str,
    status: str,
    final_text: str,
    error: str | None,
) -> None:
    """Best-effort invoke of the parent's ``post_subagent`` chain.

    Called from each terminal branch of ``run_task`` with the terminal
    status + whatever text/error was captured. Exceptions get journaled
    + swallowed; the subagent's record is already persisted before this
    fires so a buggy hook cannot corrupt the transition.
    """
    if hooks is None or not hooks.post_subagent:
        return
    try:
        await hooks.run_post_subagent(
            task_id=task_id,
            status=status,
            final_text=final_text,
            error=error,
        )
    except Exception as exc:  # noqa: BLE001
        journal.write(
            "post_subagent_hook_failed",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_task(
    store: TasksStore,
    factory: SubagentFactory,
    task_id: str,
    *,
    parent_hooks: HookChain | None = None,
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
    final_text = ""
    try:
        agent = factory.spawn(
            record.prompt,
            agent_type=record.agent_type or "general-purpose",
        )
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
        # Fire BEFORE re-raising so the hook sees the cancellation even
        # when the parent is tearing down.
        await _fire_post_subagent(
            parent_hooks,
            task_id=task_id,
            status="cancelled",
            final_text=final_text,
            error=None,
        )
        raise
    except Exception as exc:  # noqa: BLE001
        err_msg = f"{type(exc).__name__}: {exc}"
        store.mark_failed(task_id, err_msg)
        journal.write("subagent_failed", task_id=task_id, error=err_msg)
        if agent is not None:
            agent.close()
        await _fire_post_subagent(
            parent_hooks,
            task_id=task_id,
            status="failed",
            final_text=final_text,
            error=err_msg,
        )
    else:
        journal.write("subagent_completed", task_id=task_id)
        if agent is not None:
            agent.close()
        await _fire_post_subagent(
            parent_hooks,
            task_id=task_id,
            status="completed",
            final_text=final_text,
            error=None,
        )
