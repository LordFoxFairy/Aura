"""Typed todo schema and rendering — single source of truth.

``TodoItem`` is the shared pydantic model that both ``aura.tools.todo_write``
(for its ``args_schema``) and ``aura.core.memory.context`` (for ``<todos>``
prompt rendering) depend on. Keeping the schema in ``core`` respects the
``core → tools`` dependency direction: the tool imports from core, not the
other way around.

``state.custom["todos"]`` is the in-memory session state — by convention it
holds ``list[TodoItem]`` (pydantic instances, not dumped dicts), so both the
tool and the renderer see typed attribute access (``t.content``, ``t.status``,
``t.active_form``). If a future change adds a field, both sites get the new
attribute at once; there is no silent dict-shape drift between writer and
reader.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(BaseModel):
    content: str = Field(
        ...,
        min_length=1,
        description="Imperative form of the task (e.g. 'Add retry logic').",
    )
    status: TodoStatus = Field(
        ...,
        description=(
            "Current state of this item. Keep EXACTLY ONE item in_progress "
            "while actively working; flip to completed the moment the item's "
            "work is finished. Use pending for queued items not yet started."
        ),
    )
    active_form: str = Field(
        ...,
        min_length=1,
        description=(
            "Present-continuous phrasing shown while this item is in_progress "
            "(e.g. 'Adding retry logic'). Matches content but in -ing form."
        ),
    )


def render_todos_body(todos: list[TodoItem]) -> str:
    """Render a list of TodoItem for the ``<todos>`` prompt block.

    Format is intentionally informal (the model rereads its own output):
    completed items get only their content; active items also show the
    present-continuous form so the model can track what "now" is.
    """
    lines: list[str] = []
    for t in todos:
        if t.status == "completed":
            lines.append(f"- [completed] {t.content}")
        else:
            lines.append(f"- [{t.status}] {t.content} (active: {t.active_form})")
    return "\n".join(lines)
