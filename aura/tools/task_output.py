"""task_output — snapshot a subagent's current state.

Read-only / concurrency-safe: just reads the TasksStore. No side effects,
no event-loop work. The LLM uses this to poll a task it previously spawned
via ``task_create``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from aura.core.tasks.store import TasksStore
from aura.schemas.tool import ToolError, tool_metadata


class TaskOutputParams(BaseModel):
    task_id: str = Field(
        ..., min_length=1, description="Task id returned by task_create.",
    )


def _preview(args: dict[str, Any]) -> str:
    tid = args.get("task_id", "?")
    return f"task_output: {tid[:8]}"


class TaskOutput(BaseTool):
    """Return the current transcript / result snapshot for a subagent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "task_output"
    description: str = (
        "Fetch the current status / transcript / final_result of a subagent "
        "task. Call this to poll a task_id returned by task_create."
    )
    args_schema: type[BaseModel] = TaskOutputParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_read_only=True,
        is_concurrency_safe=True,
        max_result_size_chars=8000,
        args_preview=_preview,
    )
    store: TasksStore

    def _run(self, task_id: str) -> dict[str, Any]:
        return self._fetch(task_id)

    async def _arun(self, task_id: str) -> dict[str, Any]:
        return self._fetch(task_id)

    def _fetch(self, task_id: str) -> dict[str, Any]:
        rec = self.store.get(task_id)
        if rec is None:
            raise ToolError(f"unknown task_id: {task_id!r}")
        return {
            "task_id": rec.id,
            "description": rec.description,
            "status": rec.status,
            "final_result": rec.final_result,
            "error": rec.error,
        }
