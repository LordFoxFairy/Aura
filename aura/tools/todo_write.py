"""todo_write — replace the session's todo list."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aura.schemas.state import LoopState
from aura.schemas.todos import TodoItem
from aura.schemas.tool import ToolMetadata


class TodoWriteParams(BaseModel):
    todos: list[TodoItem] = Field(
        ..., description="Complete new list; replaces prior state."
    )

    @model_validator(mode="after")
    def _ensure_single_in_progress(self) -> TodoWriteParams:
        # At most one in_progress; zero is fine.
        n = sum(1 for t in self.todos if t.status == "in_progress")
        if n > 1:
            raise ValueError(
                f"only one item may have status='in_progress' at a time; found {n}"
            )
        return self


def _preview(args: dict[str, Any]) -> str:
    return f"todos: {len(args.get('todos', []))} items"


class TodoWrite(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "todo_write"
    description: str = (
        "Update the todo list for the current session. Use this proactively to "
        "track multi-step work — write a list upfront and update it as you "
        "complete items. Keep exactly one item in_progress when actively "
        "working; mark completed the moment an item is done."
    )
    args_schema: type[BaseModel] = TodoWriteParams
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=False,
        is_concurrency_safe=False,
        rule_matcher=None,
        args_preview=_preview,
        timeout_sec=None,
    )
    state: LoopState

    def _run(self, todos: list[TodoItem]) -> dict[str, Any]:
        # Preserve list identity (LoopSlots is frozen but the list is mutable).
        self.state.slots.todos.clear()
        self.state.slots.todos.extend(todos)
        return {"message": "Todos updated."}
