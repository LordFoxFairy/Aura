"""task_create — spawn a subagent fire-and-forget.

Returns the ``task_id`` the instant the subagent is scheduled, without
awaiting its completion. Parent's loop continues immediately; the subagent
runs as a detached :class:`asyncio.Task` in the same event loop. Callers
poll via ``task_output`` and/or the ``/tasks`` command.

State is injected at Agent construction time — see
``Agent.__init__``'s stateful-tools wiring block.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.run import run_task
from aura.core.tasks.store import TasksStore
from aura.schemas.tool import tool_metadata


class TaskCreateParams(BaseModel):
    description: str = Field(
        ..., min_length=1, max_length=100,
        description="Short name shown in /tasks list.",
    )
    prompt: str = Field(
        ..., min_length=1,
        description="The prompt the subagent should work on.",
    )


def _preview(args: dict[str, Any]) -> str:
    desc = args.get("description", "?")
    return f"subagent: {desc}"


class TaskCreate(BaseTool):
    """Spawn a subagent; return its task_id immediately."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "task_create"
    description: str = (
        "Spawn a subagent to work on a focused subtask. Returns a task_id "
        "immediately — use task_output(task_id) to fetch progress / result. "
        "The subagent runs in the background; your turn continues."
    )
    args_schema: type[BaseModel] = TaskCreateParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_destructive=False,
        # State-mutating (adds to TasksStore + creates an asyncio.Task on
        # the loop); can't be batched with siblings.
        is_concurrency_safe=False,
        max_result_size_chars=1000,
        args_preview=_preview,
    )
    # Stateful deps: store + factory + the running-tasks map (shared with
    # the owning Agent so Agent.close() can cancel). ``running`` lives as a
    # PrivateAttr because pydantic v2 eagerly deep-copies dict fields during
    # model validation, which would break the identity-sharing contract
    # with the owning Agent. Accepted as a regular kwarg via __init__ below.
    store: TasksStore
    factory: SubagentFactory
    _running: dict[str, asyncio.Task[None]] = PrivateAttr()

    def __init__(
        self,
        *,
        store: TasksStore,
        factory: SubagentFactory,
        running: dict[str, asyncio.Task[None]],
        **kwargs: Any,
    ) -> None:
        super().__init__(store=store, factory=factory, **kwargs)
        self._running = running

    @property
    def running(self) -> dict[str, asyncio.Task[None]]:
        return self._running

    def _run(self, description: str, prompt: str) -> dict[str, Any]:
        raise NotImplementedError("task_create is async-only; use ainvoke")

    async def _arun(self, description: str, prompt: str) -> dict[str, Any]:
        record = self.store.create(description=description, prompt=prompt)
        # Fire-and-forget. The handle is kept in ``self._running`` so the
        # parent Agent can cancel it on close / Ctrl+C. We also attach a
        # done-callback to clean the map on natural completion so the
        # dict doesn't grow unbounded across a long session.
        task: asyncio.Task[None] = asyncio.create_task(
            run_task(self.store, self.factory, record.id),
            name=f"aura-subagent-{record.id[:8]}",
        )
        self._running[record.id] = task

        def _cleanup(t: asyncio.Task[None]) -> None:
            self._running.pop(record.id, None)
            # Swallow any residual exception so the event loop's default
            # "Task exception was never retrieved" warning doesn't fire —
            # run_task already wrote the failure into the record.
            if not t.cancelled():
                t.exception()

        task.add_done_callback(_cleanup)
        return {
            "task_id": record.id,
            "description": record.description,
            "status": "running",
        }
