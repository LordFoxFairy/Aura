"""task_get — full structured snapshot of a TaskRecord."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from aura.application.tasks.store import TasksStore
from aura.domain.task import TaskRecord
from aura.schemas.tool import ToolError, ToolMetadata


class TaskGetParams(BaseModel):
    task_id: str = Field(
        ..., min_length=1, description="Task id returned by task_create.",
    )
    include_messages: bool = Field(
        default=False,
        description="When True, include the full subagent transcript.",
    )


def _preview(args: dict[str, Any]) -> str:
    tid = args.get("task_id", "?")
    return f"task_get: {tid[:8]}"


def _serialize_messages(rec: TaskRecord) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in rec.messages:
        out.append({"type": msg.type, "content": msg.content})
    return out


def _serialize(rec: TaskRecord, *, include_messages: bool) -> dict[str, Any]:
    duration: float | None = None
    if rec.finished_at is not None:
        duration = rec.finished_at - rec.started_at
    payload: dict[str, Any] = {
        "task_id": rec.id,
        "description": rec.description,
        "kind": rec.kind,
        "model_spec": rec.model_spec,
        "agent_type": rec.agent_type or "general-purpose",
        "status": rec.status,
        "started_at": rec.started_at,
        "finished_at": rec.finished_at,
        "observed_at": rec.observed_at,
        "duration_seconds": duration,
        "final_result": rec.final_result,
        "error": rec.error,
        "transcript_path": (
            str(rec.transcript_path) if rec.transcript_path is not None else None
        ),
        "progress": {
            "tool_count": rec.progress.tool_count,
            "token_count": rec.progress.token_count,
            "line_count": rec.progress.line_count,
            "last_activity_at": rec.progress.last_activity_at,
            "recent_activities": list(rec.progress.recent_activities),
            "input_tokens": rec.progress.input_tokens,
            "output_tokens": rec.progress.output_tokens,
            "latest_summary": rec.progress.latest_summary,
            "summary_updated_at": rec.progress.summary_updated_at,
        },
    }
    if include_messages:
        payload["messages"] = _serialize_messages(rec)
    return payload


class TaskGet(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "task_get"
    description: str = (
        "Fetch the full status / lifecycle metadata / progress snapshot of "
        "a subagent task. Complements task_output (which returns only the "
        "final result). Pass include_messages=True to also fetch the full "
        "child transcript."
    )
    args_schema: type[BaseModel] = TaskGetParams
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        rule_matcher=None,
        args_preview=_preview,
        timeout_sec=None,
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
        if rec.status != "running":
            self.store.mark_observed(task_id)
        return _serialize(rec, include_messages=include_messages)
