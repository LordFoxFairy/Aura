"""Typed todo schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(BaseModel):
    content: str = Field(
        ...,
        min_length=1,
        description="Imperative form of the task.",
    )
    status: TodoStatus = Field(
        ...,
        description="Current state; keep exactly one in_progress while working.",
    )
    active_form: str = Field(
        ...,
        min_length=1,
        description="Present-continuous phrasing shown while in_progress.",
    )
