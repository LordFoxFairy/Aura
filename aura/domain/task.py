"""TaskRecord — one dataclass per subagent / shell / teammate task."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langchain_core.messages import BaseMessage

TaskStatus = Literal["running", "completed", "failed", "cancelled"]
TaskKind = Literal["subagent", "shell", "teammate"]

# Tight cap: serialised into task_get; a long subagent could fire hundreds.
_RECENT_ACTIVITIES_CAP = 5
# Shell output is line-by-line and much chattier than tool names.
SHELL_RECENT_ACTIVITIES_CAP = 20


@dataclass
class TaskProgress:
    tool_count: int = 0
    token_count: int = 0
    line_count: int = 0
    last_activity_at: float | None = None
    recent_activities: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TaskRecord:
    id: str
    description: str
    prompt: str
    status: TaskStatus = "running"
    kind: TaskKind = "subagent"
    # Free-form so the store layer needn't re-import the registry; None for non-subagent kinds.
    agent_type: str | None = None
    messages: list[BaseMessage] = field(default_factory=list)
    final_result: str | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    observed_at: float | None = None
    progress: TaskProgress = field(default_factory=TaskProgress)
    metadata: dict[str, object] = field(default_factory=dict)
    transcript_path: Path | None = None
    model_spec: str = ""


@dataclass
class TaskNotification:
    # Terminal-transition push drained by Context.build into <task-notification>.

    task_id: str
    status: TaskStatus
    summary: str | None
    description: str = ""
    exit_code: int | None = None


def append_recent(
    progress: TaskProgress,
    activity: str,
    *,
    cap: int = _RECENT_ACTIVITIES_CAP,
) -> None:
    progress.recent_activities.append(activity)
    if len(progress.recent_activities) > cap:
        del progress.recent_activities[:-cap]
