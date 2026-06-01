"""Stateful-tool factory protocol + per-tool factories."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.tools import BaseTool

from aura.application.loop_state import LoopState
from aura.application.tasks.spawn_port import SpawnPort
from aura.application.tasks.store import TasksStore
from aura.infrastructure.persistence.storage import SessionStorage
from aura.tools.ask_user import AskUserQuestion, UserAsker
from aura.tools.send_message import SendMessage
from aura.tools.task_create import TaskCreate
from aura.tools.task_get import TaskGet
from aura.tools.task_list import TaskList
from aura.tools.task_stop import TaskStop
from aura.tools.todo_write import TodoWrite


@dataclass(frozen=True)
class ToolRuntime:
    """Frozen DI container; factories pick the fields they need."""

    state: LoopState
    asker: UserAsker | None = None
    tasks_store: TasksStore | None = None
    spawner: SpawnPort | None = None
    running_tasks: dict[str, asyncio.Task[None]] | None = None
    running_shells: dict[str, asyncio.subprocess.Process] | None = None
    transcript_storage: SessionStorage | None = None
    # Providers (not snapshots): ToolRuntime is built once; team binding is set later by join_team.
    team_provider: Callable[[], Any] | None = None
    member_name_provider: Callable[[], str | None] | None = None


@runtime_checkable
class StatefulToolFactory(Protocol):
    """Each factory carries a ``name`` and a :meth:`build` returning a wired tool."""

    name: str

    def build(self, runtime: ToolRuntime) -> BaseTool: ...


class TodoWriteFactory:
    name: str = "todo_write"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        return TodoWrite(state=runtime.state)


class AskUserQuestionFactory:
    name: str = "ask_user_question"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        if runtime.asker is None:  # pragma: no cover  # Agent always wires a fallback
            raise RuntimeError(
                "AskUserQuestionFactory.build requires runtime.asker."
            )
        return AskUserQuestion(asker=runtime.asker)


class TaskCreateFactory:
    name: str = "task_create"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        if (
            runtime.tasks_store is None
            or runtime.spawner is None
            or runtime.running_tasks is None
        ):
            raise RuntimeError(
                "TaskCreateFactory.build requires tasks_store, "
                "spawner, and running_tasks on the runtime."
            )
        return TaskCreate(
            store=runtime.tasks_store,
            spawner=runtime.spawner,
            running=runtime.running_tasks,
            transcript_storage=runtime.transcript_storage,
        )


class TaskGetFactory:
    name: str = "task_get"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        if runtime.tasks_store is None:
            raise RuntimeError(
                "TaskGetFactory.build requires runtime.tasks_store."
            )
        return TaskGet(store=runtime.tasks_store)


class TaskListFactory:
    name: str = "task_list"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        if runtime.tasks_store is None:
            raise RuntimeError(
                "TaskListFactory.build requires runtime.tasks_store."
            )
        return TaskList(store=runtime.tasks_store)


class TaskStopFactory:
    name: str = "task_stop"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        if (
            runtime.tasks_store is None
            or runtime.running_tasks is None
            or runtime.running_shells is None
        ):
            raise RuntimeError(
                "TaskStopFactory.build requires tasks_store, running_tasks, "
                "and running_shells on the runtime."
            )
        return TaskStop(
            store=runtime.tasks_store,
            running=runtime.running_tasks,
            running_shells=runtime.running_shells,
        )


class SendMessageFactory:
    """Outside a team the tool surfaces a clean ToolError; gating lives on :class:`Agent`."""

    name: str = "send_message"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        if runtime.team_provider is None or runtime.member_name_provider is None:
            raise RuntimeError(
                "SendMessageFactory.build requires team_provider + member_name_provider.",
            )
        return SendMessage(
            team_provider=runtime.team_provider,
            member_name_provider=runtime.member_name_provider,
        )


STATEFUL_TOOL_FACTORIES: list[StatefulToolFactory] = [
    TodoWriteFactory(),
    AskUserQuestionFactory(),
    TaskCreateFactory(),
    TaskGetFactory(),
    TaskListFactory(),
    TaskStopFactory(),
    SendMessageFactory(),
]
