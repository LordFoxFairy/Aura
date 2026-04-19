"""todo_write — replace the session's todo list.

Schema (``TodoItem``) and ``<todos>`` rendering live in ``aura.core.todos``;
this file is just the write path: pydantic-validated input → ``LoopState``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from aura.schemas.state import LoopState
from aura.schemas.todos import TodoItem
from aura.schemas.tool import tool_metadata


class TodoWriteParams(BaseModel):
    todos: list[TodoItem] = Field(
        ..., description="Complete new list; replaces prior state."
    )


def _preview(args: dict[str, Any]) -> str:
    return f"todos: {len(args.get('todos', []))} items"


class TodoWrite(BaseTool):
    # LoopState is a stdlib @dataclass, not a pydantic model; this lets pydantic
    # accept it as a field type without trying to validate its internals.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "todo_write"
    description: str = (
        "Update the todo list for the current session. Use this proactively to "
        "track multi-step work — write a list upfront and update it as you "
        "complete items. Keep exactly one item in_progress when actively "
        "working; mark completed the moment an item is done."
    )
    args_schema: type[BaseModel] = TodoWriteParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_concurrency_safe=False,  # mutates shared LoopState — cannot parallelize
        args_preview=_preview,
    )
    state: LoopState

    def _run(self, todos: list[TodoItem]) -> dict[str, Any]:
        self.state.custom["todos"] = list(todos)
        return {"message": "Todos updated."}
