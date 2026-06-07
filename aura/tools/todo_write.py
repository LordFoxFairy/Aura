"""todo_write — replace the session's todo list."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aura.application.loop_state import LoopState
from aura.domain.todos import TodoItem
from aura.domain.tool import ToolMetadata, coerce_json_collection


class TodoWriteParams(BaseModel):
    todos: list[TodoItem] = Field(
        ..., description="Complete new list; replaces prior state."
    )

    @field_validator("todos", mode="before")
    @classmethod
    def _coerce_todos(cls, v: object) -> object:
        return coerce_json_collection(v)

    @model_validator(mode="after")
    def _ensure_single_in_progress(self) -> TodoWriteParams:
        n = sum(1 for t in self.todos if t.status == "in_progress")
        if n > 1:
            raise ValueError(
                f"only one item may have status='in_progress' at a time; found {n}"
            )
        return self


class TodoWriteResult(TypedDict):
    message: str


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
    args_schema: type[BaseModel] = TodoWriteParams  # pyright: ignore[reportIncompatibleVariableOverride]  # langchain BaseTool declares args_schema as mutable ArgsSchema|None; subclass narrows widely on purpose.
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=False,
        is_concurrency_safe=False,
        rule_matcher=None,
        args_preview=_preview,
        timeout_sec=None,
    )
    state: LoopState

    def _run(self, todos: list[TodoItem]) -> TodoWriteResult:
        # In-place mutation: LoopSlots is frozen but the todos list is shared by reference.
        self.state.slots.todos.clear()
        self.state.slots.todos.extend(todos)
        return {"message": "Todos updated."}
