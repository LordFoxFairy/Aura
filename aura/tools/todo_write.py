"""todo_write tool — stateful, factory-bound per Agent instance.

Mutates LoopState.custom["todos"]. Auto-clears when every item is completed.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.core.state import LoopState
from aura.tools.base import build_tool

_DESCRIPTION = (
    "Update the todo list for the current session. Use this proactively to "
    "track multi-step work — write a todo list upfront and update it as you "
    "complete items. Keep exactly one item in_progress when actively working; "
    "mark completed the moment an item is done."
)


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


def render_todos(todos: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for t in todos:
        if t["status"] == "completed":
            lines.append(f"- [completed] {t['content']}")
        else:
            lines.append(
                f"- [{t['status']}] {t['content']} (active: {t['activeForm']})"
            )
    return "\n".join(lines)


def make_todo_write_tool(state: LoopState) -> BaseTool:
    def _run(todos: list[TodoItem]) -> dict[str, Any]:
        if todos and all(t.status == "completed" for t in todos):
            state.custom["todos"] = []
        else:
            state.custom["todos"] = [t.model_dump() for t in todos]
        return {"message": "Todos updated."}

    return build_tool(
        name="todo_write",
        description=_DESCRIPTION,
        args_schema=TodoWriteParams,
        func=_run,
        is_read_only=False,
        is_destructive=False,
        is_concurrency_safe=False,
    )
