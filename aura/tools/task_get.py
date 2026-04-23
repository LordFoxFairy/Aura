"""task_get — full structured snapshot of a single TaskRecord.

A richer sibling of ``task_output``: same read-only, concurrency-safe
shape, but returns the lifecycle metadata (``started_at``, ``finished_at``,
``duration_seconds``, ``parent_id``, ``progress``) that a caller polling
a subagent needs to reason about "is this still moving", "did it ever
finish", "what was the child actually doing".

``messages`` is expensive to serialize (full BaseMessage transcript) and
floods the parent's context; the default response omits it. Set
``include_messages=True`` to get the transcript as a list of
``{type, content}`` dicts — used sparingly, typically only once the
caller has decided to debug a failed task.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from aura.core.tasks.store import TasksStore
from aura.core.tasks.types import TaskRecord
from aura.schemas.tool import ToolError, tool_metadata


class TaskGetParams(BaseModel):
    task_id: str = Field(
        ..., min_length=1, description="Task id returned by task_create.",
    )
    include_messages: bool = Field(
        default=False,
        description=(
            "When True, include the full subagent transcript. Off by "
            "default — the transcript is chatty and usually not needed."
        ),
    )


def _preview(args: dict[str, Any]) -> str:
    tid = args.get("task_id", "?")
    return f"task_get: {tid[:8]}"


def _serialize_messages(rec: TaskRecord) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in rec.messages:
        # BaseMessage.type is the string the langchain serialisation layer
        # uses (``human``/``ai``/``tool``/``system``); content is whatever
        # the model layer handed back (str or list for multi-part).
        out.append({"type": msg.type, "content": msg.content})
    return out


def _serialize(rec: TaskRecord, *, include_messages: bool) -> dict[str, Any]:
    duration: float | None = None
    if rec.finished_at is not None:
        duration = rec.finished_at - rec.started_at
    payload: dict[str, Any] = {
        "task_id": rec.id,
        "parent_id": rec.parent_id,
        "description": rec.description,
        "kind": rec.kind,
        "status": rec.status,
        "started_at": rec.started_at,
        "finished_at": rec.finished_at,
        "duration_seconds": duration,
        "final_result": rec.final_result,
        "error": rec.error,
        "progress": {
            "tool_count": rec.progress.tool_count,
            "token_count": rec.progress.token_count,
            "line_count": rec.progress.line_count,
            "last_activity_at": rec.progress.last_activity_at,
            # Copy — caller should not be able to mutate the live ring.
            "recent_activities": list(rec.progress.recent_activities),
        },
    }
    if include_messages:
        payload["messages"] = _serialize_messages(rec)
    return payload


class TaskGet(BaseTool):
    """Return a full snapshot of a subagent's TaskRecord."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "task_get"
    description: str = (
        "Fetch the full status / lifecycle metadata / progress snapshot of "
        "a subagent task. Complements task_output (which returns only the "
        "final result). Pass include_messages=True to also fetch the full "
        "child transcript."
    )
    args_schema: type[BaseModel] = TaskGetParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_read_only=True,
        is_concurrency_safe=True,
        max_result_size_chars=16000,
        args_preview=_preview,
    )
    store: TasksStore

    def _run(self, task_id: str, include_messages: bool = False) -> dict[str, Any]:
        return self._fetch(task_id, include_messages)

    async def _arun(
        self, task_id: str, include_messages: bool = False,
    ) -> dict[str, Any]:
        return self._fetch(task_id, include_messages)

    def _fetch(self, task_id: str, include_messages: bool) -> dict[str, Any]:
        rec = self.store.get(task_id)
        if rec is None:
            raise ToolError(f"unknown task_id: {task_id!r}")
        return _serialize(rec, include_messages=include_messages)
