"""task_list — enumerate recent subagent tasks with status counts.

Complements ``/tasks`` (which is for humans) with a structured-data
version the LLM can call. Returns a tiny summary per task + a counts
dict so the model can reason about the fleet without polling task_get
on every id.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from aura.core.tasks.store import TasksStore
from aura.core.tasks.types import TaskKind, TaskRecord, TaskStatus
from aura.schemas.tool import tool_metadata

_StatusFilter = Literal["all", "running", "completed", "failed", "cancelled"]
_KindFilter = Literal["all", "subagent", "shell"]


class TaskListParams(BaseModel):
    status: _StatusFilter = Field(
        default="all",
        description=(
            "Filter returned tasks to one lifecycle state, or 'all' for "
            "every record."
        ),
    )
    kind: _KindFilter = Field(
        default="all",
        description=(
            "Filter by task kind: 'subagent' (task_create), 'shell' "
            "(bash_background), or 'all' for both."
        ),
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
    }


_ALL_STATUSES: tuple[TaskStatus, ...] = (
    "running", "completed", "failed", "cancelled",
)


class TaskList(BaseTool):
    """Return a newest-first window of TaskRecords plus status counts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "task_list"
    description: str = (
        "List recent subagent tasks with per-status counts. Pass "
        "status='running' (or 'completed'/'failed'/'cancelled') to filter; "
        "default 'all' returns everything. limit caps the window (default 20)."
    )
    args_schema: type[BaseModel] = TaskListParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_read_only=True,
        is_concurrency_safe=True,
        max_result_size_chars=8000,
        args_preview=_preview,
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
        # Counts are over the FULL fleet regardless of filter — the user
        # asked "show me the failed ones", and also wants the running
        # count to know if anything is still in flight.
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
