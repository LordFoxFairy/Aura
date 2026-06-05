"""Builders mapping a stable tool name to its runtime-wired stateful tool."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from aura.application.runtime.tool_runtime import ToolRuntime
from aura.tools.ask_user import AskUserQuestion
from aura.tools.send_message import SendMessage
from aura.tools.task_create import TaskCreate
from aura.tools.task_get import TaskGet
from aura.tools.task_list import TaskList
from aura.tools.task_stop import TaskStop
from aura.tools.todo_write import TodoWrite


@dataclass(frozen=True)
class StatefulToolFactory:
    name: str
    build: Callable[[ToolRuntime], BaseTool]


def _todo_write(runtime: ToolRuntime) -> BaseTool:
    return TodoWrite(state=runtime.state)


def _ask_user_question(runtime: ToolRuntime) -> BaseTool:
    if runtime.asker is None:
        raise RuntimeError("ask_user_question requires runtime.asker")
    return AskUserQuestion(asker=runtime.asker)


def _task_create(runtime: ToolRuntime) -> BaseTool:
    if runtime.tasks_store is None or runtime.spawner is None or runtime.running_tasks is None:
        raise RuntimeError("task_create requires tasks_store, spawner, running_tasks")
    return TaskCreate(
        store=runtime.tasks_store,
        spawner=runtime.spawner,
        running=runtime.running_tasks,
        transcript_storage=runtime.transcript_storage,
    )


def _task_get(runtime: ToolRuntime) -> BaseTool:
    if runtime.tasks_store is None:
        raise RuntimeError("task_get requires tasks_store")
    return TaskGet(store=runtime.tasks_store)


def _task_list(runtime: ToolRuntime) -> BaseTool:
    if runtime.tasks_store is None:
        raise RuntimeError("task_list requires tasks_store")
    return TaskList(store=runtime.tasks_store)


def _task_stop(runtime: ToolRuntime) -> BaseTool:
    if (
        runtime.tasks_store is None
        or runtime.running_tasks is None
        or runtime.running_shells is None
    ):
        raise RuntimeError("task_stop requires tasks_store, running_tasks, running_shells")
    return TaskStop(
        store=runtime.tasks_store,
        running=runtime.running_tasks,
        running_shells=runtime.running_shells,
    )


def _send_message(runtime: ToolRuntime) -> BaseTool:
    # Gating lives on AgentSession; outside a team the tool surfaces a clean ToolError.
    if runtime.team_provider is None or runtime.member_name_provider is None:
        raise RuntimeError("send_message requires team_provider, member_name_provider")
    return SendMessage(
        team_provider=runtime.team_provider,
        member_name_provider=runtime.member_name_provider,
    )


STATEFUL_TOOL_FACTORIES: list[StatefulToolFactory] = [
    StatefulToolFactory("todo_write", _todo_write),
    StatefulToolFactory("ask_user_question", _ask_user_question),
    StatefulToolFactory("task_create", _task_create),
    StatefulToolFactory("task_get", _task_get),
    StatefulToolFactory("task_list", _task_list),
    StatefulToolFactory("task_stop", _task_stop),
    StatefulToolFactory("send_message", _send_message),
]
