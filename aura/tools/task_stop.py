"""task_stop — cancel a still-running task (subagent or shell).

Mirrors claude-code's ``TaskStopTool``: the tool looks up the detached
``asyncio.Task`` handle on the parent Agent's running-tasks map, calls
``.cancel()``, and awaits it (with a timeout so a stuck child doesn't
block the parent's loop).

For ``shell`` tasks (spawned by ``bash_background``), the tool instead
reaches into the ``_running_shells`` map that holds the
``asyncio.subprocess.Process`` handle, delivers SIGTERM → 3s → SIGKILL,
and lets the bash_background coroutine flip the record to cancelled
when it observes the early exit.

The actual "mark cancelled on the record" happens inside the background
coroutine (``run_task`` for subagents, ``bash_background``'s spawned
watcher for shell) when it catches the early termination — that
guarantees the record always reflects the true lifecycle regardless of
who cancelled it (this tool, ``Agent.close()``, or Ctrl+C). This tool
only arms the cancel; it doesn't touch the store directly for
``running``-handle cases.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.core.tasks.store import TasksStore
from aura.schemas.tool import ToolError, tool_metadata

# Upper bound on how long we wait for the child to unwind after
# ``.cancel()``. Most subagents unwind in a single event-loop tick; this
# cap just stops the tool from hanging forever if a child has misbehaving
# shielded coroutines.
_CANCEL_TIMEOUT_SECONDS = 2.0
# Between SIGTERM and SIGKILL for shell tasks. Mirrors bash_background's
# own shutdown ladder so manual stop and timeout-kill behave identically.
_SHELL_TERM_GRACE = 3.0


class TaskStopParams(BaseModel):
    task_id: str = Field(
        ..., min_length=1, description="Task id returned by task_create.",
    )


def _preview(args: dict[str, Any]) -> str:
    tid = args.get("task_id", "?")
    return f"task_stop: {tid[:8]}"


class TaskStop(BaseTool):
    """Cancel a running subagent; error if task is unknown or already done."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "task_stop"
    description: str = (
        "Cancel a still-running subagent task by id. Raises ToolError if "
        "the task is unknown or already in a terminal state."
    )
    args_schema: type[BaseModel] = TaskStopParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_destructive=False,
        # Mutates the running-tasks map + triggers a state transition on
        # the TaskRecord; not safe to run alongside its siblings.
        is_concurrency_safe=False,
        max_result_size_chars=500,
        args_preview=_preview,
    )
    store: TasksStore
    # See task_create for why ``running`` / ``running_shells`` are
    # PrivateAttrs rather than pydantic fields — identity-sharing with the
    # owning Agent. pydantic v2 would deep-copy them during model validation
    # and break the live-reference contract.
    _running: dict[str, asyncio.Task[None]] = PrivateAttr()
    _running_shells: dict[str, asyncio.subprocess.Process] = PrivateAttr()

    def __init__(
        self,
        *,
        store: TasksStore,
        running: dict[str, asyncio.Task[None]],
        running_shells: dict[str, asyncio.subprocess.Process] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(store=store, **kwargs)
        self._running = running
        self._running_shells = running_shells if running_shells is not None else {}

    @property
    def running(self) -> dict[str, asyncio.Task[None]]:
        return self._running

    @property
    def running_shells(self) -> dict[str, asyncio.subprocess.Process]:
        return self._running_shells

    def _run(self, task_id: str) -> dict[str, Any]:
        raise NotImplementedError("task_stop is async-only; use ainvoke")

    async def _arun(self, task_id: str) -> dict[str, Any]:
        rec = self.store.get(task_id)
        if rec is None:
            raise ToolError(f"unknown task_id: {task_id!r}")
        if rec.status != "running":
            raise ToolError(
                f"task {task_id[:8]} is already in terminal state "
                f"{rec.status!r}; nothing to stop",
            )
        if rec.kind == "shell":
            return await self._stop_shell(task_id)
        return await self._stop_subagent(task_id)

    async def _stop_subagent(self, task_id: str) -> dict[str, Any]:
        handle = self._running.get(task_id)
        if handle is None or handle.done():
            # Store says running but we have no live handle — race with the
            # cleanup done-callback. Fall through to a direct mark so the
            # record doesn't get stuck in "running" forever.
            self.store.mark_cancelled(task_id)
            return {"task_id": task_id, "status": "cancelled"}
        handle.cancel()
        try:
            # Shield the await from the caller's own cancel so we get a
            # clean read on the child's terminal state. The child raises
            # CancelledError when it finishes unwinding, which is expected
            # and swallowed here; any other exception surfaces for triage.
            await asyncio.wait_for(
                asyncio.shield(handle), timeout=_CANCEL_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            # Child didn't unwind in time — force the record flag anyway
            # so the LLM sees a consistent state. The detached task will
            # eventually finish and run_task will no-op on an already-final
            # record.
            self.store.mark_cancelled(task_id)
        return {"task_id": task_id, "status": "cancelled"}

    async def _stop_shell(self, task_id: str) -> dict[str, Any]:
        proc = self._running_shells.get(task_id)
        if proc is None or proc.returncode is not None:
            # Either the process already exited (race with the watcher's
            # cleanup) or it was never registered. Flip the record so the
            # LLM doesn't see a stuck "running".
            self.store.mark_cancelled(task_id)
            return {"task_id": task_id, "status": "cancelled"}
        # SIGTERM → grace → SIGKILL. Errors from terminate/kill are
        # swallowed — we're in a cleanup path and cannot raise over
        # the caller's intent.
        with contextlib.suppress(ProcessLookupError, Exception):
            proc.terminate()
        with contextlib.suppress(TimeoutError, Exception):
            await asyncio.wait_for(proc.wait(), timeout=_SHELL_TERM_GRACE)
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, Exception):
                proc.kill()
            with contextlib.suppress(TimeoutError, Exception):
                await asyncio.wait_for(proc.wait(), timeout=_SHELL_TERM_GRACE)
        # The bash_background watcher may still be looping on line reads;
        # mark cancelled here so the record is terminal immediately, and
        # the watcher's own "final transition" path no-ops on an already-
        # terminal record.
        self.store.mark_cancelled(task_id)
        return {"task_id": task_id, "status": "cancelled"}
