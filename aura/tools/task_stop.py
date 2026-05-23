"""task_stop — cancel a running task (subagent or shell)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.application.tasks.store import TasksStore
from aura.schemas.tool import ToolError, ToolMetadata

_CANCEL_TIMEOUT_SECONDS = 2.0
_SHELL_TERM_GRACE = 3.0


class TaskStopParams(BaseModel):
    task_id: str = Field(
        ..., min_length=1, description="Task id returned by task_create.",
    )


def _preview(args: dict[str, Any]) -> str:
    tid = args.get("task_id", "?")
    return f"task_stop: {tid[:8]}"


class TaskStop(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "task_stop"
    description: str = (
        "Cancel a still-running subagent task by id. Raises ToolError if "
        "the task is unknown or already in a terminal state."
    )
    args_schema: type[BaseModel] = TaskStopParams  # pyright: ignore[reportIncompatibleVariableOverride]  # langchain BaseTool declares args_schema as mutable ArgsSchema|None; subclass narrows widely on purpose.
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=False,
        is_concurrency_safe=False,
        rule_matcher=None,
        args_preview=_preview,
        timeout_sec=None,
    )
    store: TasksStore
    # PrivateAttr: pydantic v2 deep-copies regular dict fields, breaking identity-share with Agent.
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
            self.store.mark_cancelled(task_id)
            return {"task_id": task_id, "status": "cancelled"}
        handle.cancel()
        try:
            await asyncio.wait_for(
                asyncio.shield(handle), timeout=_CANCEL_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            self.store.mark_cancelled(task_id)
        return {"task_id": task_id, "status": "cancelled"}

    async def _stop_shell(self, task_id: str) -> dict[str, Any]:
        proc = self._running_shells.get(task_id)
        if proc is None or proc.returncode is not None:
            self.store.mark_cancelled(task_id)
            return {"task_id": task_id, "status": "cancelled"}
        with contextlib.suppress(ProcessLookupError, Exception):
            proc.terminate()
        with contextlib.suppress(TimeoutError, Exception):
            await asyncio.wait_for(proc.wait(), timeout=_SHELL_TERM_GRACE)
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, Exception):
                proc.kill()
            with contextlib.suppress(TimeoutError, Exception):
                await asyncio.wait_for(proc.wait(), timeout=_SHELL_TERM_GRACE)
        self.store.mark_cancelled(task_id)
        return {"task_id": task_id, "status": "cancelled"}
