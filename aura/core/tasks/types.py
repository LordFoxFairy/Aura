"""TaskRecord — the one dataclass persisted per subagent invocation.

Mutation rules (enforced by :class:`TasksStore`, not by the dataclass
itself):

- ``status``, ``final_result``, ``error``, ``finished_at`` are set exactly
  once, when a terminal transition is taken (completed / failed / cancelled).
- ``messages`` is append-only during the ``running`` window — the subagent's
  transcript accretes, never rewrites.
- Anything else (``description``, ``prompt``, ``parent_id``, ``started_at``,
  ``id``) is write-at-create and read-only thereafter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import BaseMessage

TaskStatus = Literal["running", "completed", "failed", "cancelled"]


@dataclass
class TaskRecord:
    id: str
    parent_id: str | None
    description: str
    prompt: str
    status: TaskStatus = "running"
    messages: list[BaseMessage] = field(default_factory=list)
    final_result: str | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
