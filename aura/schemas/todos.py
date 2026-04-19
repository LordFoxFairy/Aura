"""Typed todo schema — ``TodoItem``.

Shared data type for both ``aura.tools.todo_write`` (as its ``args_schema``)
and ``aura.core.memory.context`` (for rendering the ``<todos>`` prompt
block). ``state.custom["todos"]`` holds ``list[TodoItem]`` — pydantic
instances, not dumped dicts — so both the writer and the reader get typed
attribute access without silent dict-shape drift.

The rendering helper lives with the consumer (``aura.core.memory.context``),
not here — schemas are data only.
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
