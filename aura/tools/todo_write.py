"""todo_write tool — factory-bound per Agent, mutates ``LoopState.custom['todos']``.

Auto-clears the list when every item is ``completed``. Rendering lives in the
Context layer to keep the core → tools dependency direction one-way.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.core.state import LoopState
from aura.tools.base import build_tool


class TodoItem(BaseModel):
    content: str = Field(..., min_length=1, description="Imperative form of the task")
    status: Literal["pending", "in_progress", "completed"]
    activeForm: str = Field(
        ..., min_length=1, description="Present continuous form shown during execution"
    )


class TodoWriteParams(BaseModel):
    todos: list[TodoItem] = Field(
        ..., description="Complete new list; replaces prior state"
    )


def make_todo_write_tool(state: LoopState) -> BaseTool:
    def _run(todos: list[TodoItem]) -> dict[str, Any]:
        if todos and all(t.status == "completed" for t in todos):
            state.custom["todos"] = []
        else:
            state.custom["todos"] = [t.model_dump() for t in todos]
        return {"message": "Todos updated."}

    return build_tool(
        name="todo_write",
        description=(
            "Update the todo list for the current session. Use this proactively to "
            "track multi-step work — write a todo list upfront and update it as you "
            "complete items. Keep exactly one item in_progress when actively working; "
            "mark completed the moment an item is done."
        ),
        args_schema=TodoWriteParams,
        func=_run,
    )
