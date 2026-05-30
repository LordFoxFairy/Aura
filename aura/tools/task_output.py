"""task_output — snapshot a subagent's state, optionally waiting."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, TypedDict

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from aura.application.tasks.store import TasksStore
from aura.domain.abort import current_abort_signal
from aura.domain.task import TaskRecord, TaskStatus
from aura.domain.tool import ToolError, ToolMetadata

_WAIT_TIMEOUT_MIN = 0.01
_WAIT_TIMEOUT_MAX = 600.0
_WAIT_TIMEOUT_DEFAULT = 60.0


class TaskOutputParams(BaseModel):
    task_id: str = Field(
        ..., min_length=1, description="Task id returned by task_create.",
    )
    wait: bool = Field(
        default=False,
        description="When True, block until terminal state or timeout.",
    )
    timeout: float | None = Field(
        default=_WAIT_TIMEOUT_DEFAULT,
        ge=_WAIT_TIMEOUT_MIN, le=_WAIT_TIMEOUT_MAX,
        description="When wait=True, max seconds to block. Ignored when wait=False.",
    )


def _preview(args: dict[str, Any]) -> str:
    tid = args.get("task_id", "?")
    wait = " (waiting)" if args.get("wait") else ""
    return f"task_output: {tid[:8]}{wait}"


class TaskOutputResult(TypedDict):
    task_id: str
    description: str
    status: TaskStatus
    final_result: str | None
    error: str | None
    terminal: bool
    observed_at: float | None


def _snapshot(rec: TaskRecord, *, terminal: bool) -> TaskOutputResult:
    return {
        "task_id": rec.id,
        "description": rec.description,
        "status": rec.status,
        "final_result": rec.final_result,
        "error": rec.error,
        "terminal": terminal,
        "observed_at": rec.observed_at,
    }


class TaskOutput(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "task_output"
    description: str = (
        "Fetch the current status / transcript / final_result of a subagent "
        "task. Default: instant snapshot. Set wait=True to block until the "
        "task reaches a terminal state (timeout-bounded; default 60s)."
    )
    args_schema: type[BaseModel] = TaskOutputParams  # pyright: ignore[reportIncompatibleVariableOverride]  # langchain BaseTool declares args_schema as mutable ArgsSchema|None; subclass narrows widely on purpose.
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        rule_matcher=None,
        args_preview=_preview,
        timeout_sec=None,
    )
    store: TasksStore

    def _run(
        self,
        task_id: str,
        wait: bool = False,
        timeout: float | None = _WAIT_TIMEOUT_DEFAULT,
    ) -> TaskOutputResult:
        raise NotImplementedError("task_output is async-only; use ainvoke")

    async def _arun(
        self,
        task_id: str,
        wait: bool = False,
        timeout: float | None = _WAIT_TIMEOUT_DEFAULT,
    ) -> TaskOutputResult:
        rec = self.store.get(task_id)
        if rec is None:
            raise ToolError(f"unknown task_id: {task_id!r}")

        if not wait or rec.status != "running":
            if rec.status != "running":
                self.store.mark_observed(task_id)
            return _snapshot(rec, terminal=rec.status != "running")

        effective_timeout = (
            timeout if timeout is not None else _WAIT_TIMEOUT_DEFAULT
        )
        terminal_event = self.store.terminal_event(task_id)
        abort = current_abort_signal.get()

        async def _wait_terminal() -> None:
            await terminal_event.wait()

        tasks: list[asyncio.Task[Any]] = [
            asyncio.create_task(_wait_terminal()),
        ]
        if abort is not None:
            async def _wait_abort() -> None:
                await abort.signal.wait()
            tasks.append(asyncio.create_task(_wait_abort()))

        try:
            _, pending = await asyncio.wait(
                tasks,
                timeout=effective_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise
        for t in pending:
            t.cancel()
        for t in tasks:
            if t.cancelled() or t.done():
                continue
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

        rec = self.store.get(task_id)
        if rec is None:  # pragma: no cover — store never deletes records, re-fetch is total
            raise ToolError(f"unknown task_id: {task_id!r}")

        if abort is not None and abort.aborted:
            if rec.status != "running":
                self.store.mark_observed(task_id)
            payload = _snapshot(rec, terminal=rec.status != "running")
            payload["error"] = "parent_aborted"
            return payload

        if rec.status != "running":
            self.store.mark_observed(task_id)
        return _snapshot(rec, terminal=rec.status != "running")
