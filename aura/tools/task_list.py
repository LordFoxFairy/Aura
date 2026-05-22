"""task_list — enumerate recent tasks with status counts."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from aura.application.tasks.store import TasksStore
from aura.domain.task import TaskKind, TaskRecord, TaskStatus
from aura.schemas.tool import ToolMetadata

_StatusFilter = Literal["all", "running", "completed", "failed", "cancelled"]
_KindFilter = Literal["all", "subagent", "shell", "teammate"]


class TaskListParams(BaseModel):
    status: _StatusFilter = Field(
        default="all",
        description="Filter to one lifecycle state, or 'all'.",
    )
    kind: _KindFilter = Field(
        default="all",
        description="Filter by task kind ('subagent', 'shell', 'teammate', or 'all').",
    )
    limit: int = Field(
        default=20, ge=1, le=200,
        description="Maximum number of tasks to return (newest first).",
    )


def _preview(args: dict[str, Any]) -> str:
    bits = [args.get("status", "all")]
    k = args.get("kind", "all")
    if k != "all":
        bits.append(f"kind={k}")
    return f"task_list: {', '.join(bits)}"


def _row(rec: TaskRecord) -> dict[str, Any]:
    return {
        "id": rec.id,
        "status": rec.status,
        "kind": rec.kind,
        "description": rec.description,
        "started_at": rec.started_at,
        "observed_at": rec.observed_at,
    }


_ALL_STATUSES: tuple[TaskStatus, ...] = (
    "running", "completed", "failed", "cancelled",
)


class TaskList(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "task_list"
    description: str = (
        "List recent subagent, shell, and teammate tasks with per-status counts. Pass "
        "status='running' (or 'completed'/'failed'/'cancelled') to filter; "
        "kind='subagent'/'shell'/'teammate' to filter by task kind; default "
        "'all' returns everything. limit caps the window (default 20)."
    )
    args_schema: type[BaseModel] = TaskListParams
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
        status: _StatusFilter = "all",
        kind: _KindFilter = "all",
        limit: int = 20,
    ) -> dict[str, Any]:
        return self._fetch(status, kind, limit)

    async def _arun(
        self,
        status: _StatusFilter = "all",
        kind: _KindFilter = "all",
        limit: int = 20,
    ) -> dict[str, Any]:
        return self._fetch(status, kind, limit)

    def _fetch(
        self,
        status: _StatusFilter,
        kind: _KindFilter,
        limit: int,
    ) -> dict[str, Any]:
        # Counts span the full fleet so callers see overall state even when filtering.
        all_records = self.store.list()
        counts = {
            s: sum(1 for r in all_records if r.status == s)
            for s in _ALL_STATUSES
        }
        filter_status: TaskStatus | None = (
            None if status == "all" else status
        )
        filter_kind: TaskKind | None = None if kind == "all" else kind
        records = self.store.list(
            status=filter_status, kind=filter_kind, limit=limit,
        )
        return {"tasks": [_row(r) for r in records], "counts": counts}
