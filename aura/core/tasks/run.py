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

Wall-clock timeout (U4)
-----------------------
Every subagent run is wrapped in an ``asyncio.timeout`` with a
defense-in-depth ceiling so a stalled model call / pathological tool /
infinite small-sleep loop can't leave a task in ``running`` forever.
Defaults to :data:`DEFAULT_SUBAGENT_TIMEOUT_SEC` (5 minutes); operators
can override via the ``AURA_SUBAGENT_TIMEOUT_SEC`` environment variable
(float, ``0`` or negative disables the ceiling — escape hatch for
long-running agents). On timeout the record is flipped to ``failed``
with ``error="subagent_timeout: exceeded <N>s"`` and the
``subagent_timeout`` journal event fires.

Matches the spirit of claude-code's ``AbortController`` contract
(``src/tools/AgentTool/runAgent.ts``) — claude-code uses an abort
controller instead of a wallclock, but the escape is the same: a
stalled child never blocks the parent loop forever.
"""

from __future__ import annotations

import asyncio
import os
import time

from aura.core.persistence import journal
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.store import TasksStore
from aura.schemas.events import Final, ToolCallStarted

# Default wall-clock ceiling for a single subagent run. 5 minutes is loose
# enough that legitimate research / multi-turn explorations finish well
# under the cap, while tight enough that an operator doesn't watch a
# silently-stuck task for hours before noticing. Picked to match
# ``PermissionsConfig.prompt_timeout_sec`` (same order of magnitude —
# "long enough to not fight legitimate use, short enough to matter").
DEFAULT_SUBAGENT_TIMEOUT_SEC: float = 300.0

# Environment variable override. Float seconds. ``<= 0`` disables the
# ceiling (escape hatch for specialized long-running subagents — use at
# your own risk; a stuck task can still be killed by ``task_stop`` or
# ``Agent.close``). Parsed at every call site so a long-lived process can
# pick up an env change without restart.
_TIMEOUT_ENV_VAR = "AURA_SUBAGENT_TIMEOUT_SEC"


def _resolve_timeout(override: float | None) -> float | None:
    """Pick the effective wallclock timeout.

    Precedence (highest first):

    1. Explicit ``override`` kwarg to :func:`run_task` — tests inject a
       sub-second ceiling so the regression suite doesn't have to wait 5
       minutes to observe the timeout branch.
    2. ``AURA_SUBAGENT_TIMEOUT_SEC`` environment variable.
    3. :data:`DEFAULT_SUBAGENT_TIMEOUT_SEC`.

    Returns ``None`` when the resolved value is ``<= 0``, meaning "no
    timeout — run until natural completion / cancellation." Keeping the
    escape hatch explicit is intentional: operators running overnight
    investigations shouldn't have to patch the module, and an errant
    ``0`` shouldn't fire a deny-all timeout on every task.
    """
    if override is not None:
        return override if override > 0 else None
    raw = os.environ.get(_TIMEOUT_ENV_VAR)
    if raw is not None:
        try:
            parsed = float(raw)
        except ValueError:
            # Malformed env — journal and fall through to default. Not a
            # hard error: we're in a fire-and-forget path where raising
            # would orphan the TaskRecord in ``running``.
            journal.write(
                "subagent_timeout_env_invalid",
                var=_TIMEOUT_ENV_VAR,
                value=raw,
            )
        else:
            return parsed if parsed > 0 else None
    return DEFAULT_SUBAGENT_TIMEOUT_SEC


async def run_task(
    store: TasksStore,
    factory: SubagentFactory,
    task_id: str,
    *,
    timeout_sec: float | None = None,
) -> None:
    record = store.get(task_id)
    if record is None:
        return
    # Resolve the ceiling ONCE per call so an env flip mid-run doesn't
    # race against the timeout context manager. Pass the resolved number
    # into the error message + journal so the operator sees exactly how
    # long we waited.
    effective_timeout = _resolve_timeout(timeout_sec)
    # V13-T1C: subagent_start fires before spawn so operators observing the
    # journal see a clear "run_task began for X" signal even if spawn itself
    # raises (the lifecycle pairs with one of subagent_completed /
    # subagent_cancelled / subagent_timeout / subagent_failed, all of which
    # now carry a ``duration_sec`` field computed from ``start_monotonic``).
    start_monotonic = time.monotonic()
    journal.write(
        "subagent_start",
        task_id=task_id,
        agent_type=record.agent_type or "general-purpose",
        prompt_chars=len(record.prompt),
    )
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
            task_id=task_id,
        )
        # ``asyncio.timeout(None)`` is a no-op context — the escape hatch
        # (env / kwarg set ``<= 0``) flows through as ``None`` and we get
        # the legacy "wait forever" behavior with no extra indirection.
        async with asyncio.timeout(effective_timeout):
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
        journal.write(
            "subagent_cancelled",
            task_id=task_id,
            duration_sec=round(time.monotonic() - start_monotonic, 3),
        )
        if agent is not None:
            # ``aclose`` is the async-safe teardown. ``close()`` on a live
            # event loop raises ``RuntimeError`` when the child Agent has
            # an MCP manager — today subagents don't wire MCP so the old
            # sync ``close()`` happened to work, but the moment a subagent
            # inherits MCP (an expected near-future change) every terminal
            # branch here would crash ``run_task`` mid-cleanup, orphaning
            # connections and leaving the TaskRecord in ``running``.
            await agent.aclose()
        raise
    except TimeoutError as exc:
        # Wall-clock ceiling hit. ``asyncio.timeout`` raises TimeoutError
        # after the enclosed scope is cancelled — the child's astream got
        # a CancelledError, its own except-CancelledError branches cleaned
        # up, and now we convert that into a ``failed`` record so the
        # parent's task_output sees a terminal state with a clear reason
        # rather than "stuck running forever".
        err_msg = (
            f"subagent_timeout: exceeded {effective_timeout}s "
            f"(set AURA_SUBAGENT_TIMEOUT_SEC to override; "
            f"<=0 disables the ceiling)"
        )
        store.mark_failed(task_id, err_msg)
        journal.write(
            "subagent_timeout",
            task_id=task_id,
            timeout_sec=effective_timeout,
            duration_sec=round(time.monotonic() - start_monotonic, 3),
            # Preserve the underlying cause for deep debugging; mostly
            # ``TimeoutError()`` with no message, but a future re-raise
            # inside astream could carry something more useful.
            cause=f"{type(exc).__name__}: {exc}",
        )
        if agent is not None:
            # ``aclose`` is the async-safe teardown. ``close()`` on a live
            # event loop raises ``RuntimeError`` when the child Agent has
            # an MCP manager — today subagents don't wire MCP so the old
            # sync ``close()`` happened to work, but the moment a subagent
            # inherits MCP (an expected near-future change) every terminal
            # branch here would crash ``run_task`` mid-cleanup, orphaning
            # connections and leaving the TaskRecord in ``running``.
            await agent.aclose()
    except Exception as exc:  # noqa: BLE001
        err_msg = f"{type(exc).__name__}: {exc}"
        store.mark_failed(task_id, err_msg)
        journal.write(
            "subagent_failed",
            task_id=task_id,
            error=err_msg,
            duration_sec=round(time.monotonic() - start_monotonic, 3),
        )
        if agent is not None:
            # ``aclose`` is the async-safe teardown. ``close()`` on a live
            # event loop raises ``RuntimeError`` when the child Agent has
            # an MCP manager — today subagents don't wire MCP so the old
            # sync ``close()`` happened to work, but the moment a subagent
            # inherits MCP (an expected near-future change) every terminal
            # branch here would crash ``run_task`` mid-cleanup, orphaning
            # connections and leaving the TaskRecord in ``running``.
            await agent.aclose()
    else:
        journal.write(
            "subagent_completed",
            task_id=task_id,
            duration_sec=round(time.monotonic() - start_monotonic, 3),
            final_text_chars=len(final_text),
        )
        if agent is not None:
            # ``aclose`` is the async-safe teardown. ``close()`` on a live
            # event loop raises ``RuntimeError`` when the child Agent has
            # an MCP manager — today subagents don't wire MCP so the old
            # sync ``close()`` happened to work, but the moment a subagent
            # inherits MCP (an expected near-future change) every terminal
            # branch here would crash ``run_task`` mid-cleanup, orphaning
            # connections and leaving the TaskRecord in ``running``.
            await agent.aclose()
