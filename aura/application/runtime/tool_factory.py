"""Stateful-tool factory protocol — Phase 2 §4.

Today (`Agent.__init__:414-478`) wires each stateful builtin tool with a
hand-written if/elif clause: ``todo_write`` gets ``state``,
``ask_user_question`` gets ``asker``, ``task_create`` gets a fistful of
dependencies, etc. Adding a new stateful tool means editing
:class:`Agent` itself — a god-object pull no architecture wants.

Phase 2 Task 5 swaps that pattern for a registry-of-factories: each
stateful tool ships a small :class:`StatefulToolFactory` class that
encapsulates "which dependencies do I need, and how do I build my
tool". :class:`Agent` becomes a dumb consumer — iterate the registry,
build, register. Adding a 22nd stateful tool no longer touches Agent.

Task 5 (this module) lands the protocol + the runtime container + a
PILOT factory for ``todo_write`` so the shape is exercised end-to-end
before Task 6 migrates the remaining six stateful tools and deletes
the if/elif.

Design notes:

- :class:`ToolRuntime` is a frozen dataclass with optional fields so
  factories that don't need a particular dependency can ignore it.
  Adding a new dependency = adding an Optional field; existing
  factories keep compiling.
- :class:`StatefulToolFactory` is ``@runtime_checkable`` so
  ``isinstance(obj, StatefulToolFactory)`` works for the registration
  guard and tests can assert protocol conformance directly.
- The pilot :class:`TodoWriteFactory` mirrors the current wiring
  (``TodoWrite(state=runtime.state)``) so behaviour is unchanged. Task
  6 lands the remaining factories and the registration loop.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from langchain_core.tools import BaseTool

from aura.schemas.state import LoopState
from aura.tools.ask_user import AskUserQuestion
from aura.tools.send_message import SendMessage
from aura.tools.task_create import TaskCreate
from aura.tools.task_get import TaskGet
from aura.tools.task_list import TaskList
from aura.tools.task_stop import TaskStop
from aura.tools.todo_write import TodoWrite

if TYPE_CHECKING:
    # Type-only imports — keep the runtime module light. ``Agent`` is
    # threaded through ``ToolRuntime.agent`` purely as a back-reference
    # for ``send_message`` (Phase A teams); typed as ``Any`` at runtime
    # to avoid a circular import (agent imports runtime, runtime would
    # otherwise import agent).
    from aura.application.tasks.factory import SubagentFactory
    from aura.application.tasks.store import TasksStore
    from aura.application.teams.manager import TeamManager
    from aura.infrastructure.persistence.storage import SessionStorage
    from aura.tools.ask_user import UserAsker


@dataclass(frozen=True)
class ToolRuntime:
    """Frozen bag of dependencies stateful tools may consume.

    All fields are optional except :attr:`state` (every factory the
    Phase 2 plan migrates either uses ``state`` or ignores it harmlessly
    — keeping the LoopState reference mandatory matches today's
    invariant that every Agent has exactly one ``LoopState``).

    Factories pick the fields they need; unused fields stay ``None``
    for that factory's lifetime. Frozen so a factory cannot mutate the
    runtime out from under :class:`Agent` — dependency injection is a
    one-way arrow.
    """

    # Mandatory: every Agent owns exactly one LoopState; some factories
    # (todo_write) use it directly, others ignore it. Kept non-Optional
    # so tests don't have to invent a sentinel.
    state: LoopState
    # Optional: only present when a CLI-backed asker was injected at
    # Agent construction. ``ask_user_question`` and ``exit_plan_mode``
    # consume this; other factories leave it None.
    asker: UserAsker | None = None
    # Optional: backing store for subagent task records. ``task_create``,
    # ``task_get``, ``task_list``, and ``task_stop`` factories rely on
    # it. Always present in real Agents (the Agent constructs one
    # internally), but kept Optional so tests can build a runtime
    # without spinning up TasksStore for factories that don't need it.
    tasks_store: TasksStore | None = None
    # Optional: factory used by ``task_create`` to spawn a subagent
    # backing each task. Wired by :class:`Agent` from its own
    # ``_subagent_factory``.
    subagent_factory: SubagentFactory | None = None
    # Optional: live registry of asyncio.Tasks driving subagent runs.
    # ``task_create`` registers, ``task_stop`` cancels.
    running_tasks: dict[str, asyncio.Task[None]] | None = None
    # Optional: live registry of bash background subprocesses.
    # ``task_stop`` and ``bash_background`` consume.
    running_shells: dict[str, asyncio.subprocess.Process] | None = None
    # Optional: transcript persistence target for spawned subagents
    # (``task_create`` hands it to the spawn so subagent history lands
    # under a per-task SessionStorage view).
    transcript_storage: SessionStorage | None = None
    # Optional: leader Agent back-reference for ``send_message``
    # (Phase A teams). Typed ``Any`` to avoid the agent → runtime →
    # agent import cycle; the only attribute the consumer touches is
    # ``.team`` / ``._team_member_name``.
    agent: Any = None
    # Optional: explicit TeamManager handle. Today ``send_message`` walks
    # ``agent.team`` at invocation time, but the spec calls out
    # ``team_manager`` as a first-class runtime field so future factories
    # (e.g. a teammate-listing tool) can bind to the manager directly
    # without going through the Agent back-reference.
    team_manager: TeamManager | None = None


@runtime_checkable
class StatefulToolFactory(Protocol):
    """Protocol every stateful-tool factory must satisfy.

    A factory carries the tool's NAME (so :class:`Agent` can match it
    against ``cfg.tools.enabled`` without instantiating the tool first)
    and a :meth:`build` method that returns a wired :class:`BaseTool`
    instance given the runtime container.

    ``runtime_checkable`` so ``isinstance(factory, StatefulToolFactory)``
    works at registration time for defensive guards and so tests can
    assert protocol conformance without ABC inheritance gymnastics.
    """

    name: str

    def build(self, runtime: ToolRuntime) -> BaseTool: ...


class TodoWriteFactory:
    """Factory for the ``todo_write`` tool.

    Mirrors the historical wiring ``TodoWrite(state=self._state)``.
    """

    name: str = "todo_write"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        return TodoWrite(state=runtime.state)


class AskUserQuestionFactory:
    """Factory for the ``ask_user_question`` tool.

    Mirrors the historical wiring
    ``AskUserQuestion(asker=question_asker or _unavailable_question_asker)``.
    The ``_unavailable_question_asker`` fallback lives in
    :mod:`aura.core.agent` so the Agent constructor still owns the
    "no CLI was injected" sentinel; the factory consumes whatever
    ``runtime.asker`` was wired with (the Agent fills in the fallback
    before constructing the runtime).
    """

    name: str = "ask_user_question"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        if runtime.asker is None:  # pragma: no cover — Agent always wires a fallback
            raise RuntimeError(
                "AskUserQuestionFactory.build requires runtime.asker; "
                "Agent.__init__ must inject either a CLI asker or the "
                "_unavailable_question_asker fallback."
            )
        return AskUserQuestion(asker=runtime.asker)


class TaskCreateFactory:
    """Factory for the ``task_create`` tool.

    Mirrors the historical wiring
    ``TaskCreate(store=..., factory=..., running=..., transcript_storage=...)``.
    All four dependencies are required; raise if a caller forgot to
    populate one (catches misconfiguration at construction time rather
    than at first invocation).
    """

    name: str = "task_create"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        if (
            runtime.tasks_store is None
            or runtime.subagent_factory is None
            or runtime.running_tasks is None
        ):
            raise RuntimeError(
                "TaskCreateFactory.build requires tasks_store, "
                "subagent_factory, and running_tasks on the runtime."
            )
        return TaskCreate(
            store=runtime.tasks_store,
            factory=runtime.subagent_factory,
            running=runtime.running_tasks,
            transcript_storage=runtime.transcript_storage,
        )


class TaskGetFactory:
    """Factory for the ``task_get`` tool.

    Mirrors the historical wiring ``TaskGet(store=self._tasks_store)``.
    """

    name: str = "task_get"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        if runtime.tasks_store is None:
            raise RuntimeError(
                "TaskGetFactory.build requires runtime.tasks_store."
            )
        return TaskGet(store=runtime.tasks_store)


class TaskListFactory:
    """Factory for the ``task_list`` tool.

    Mirrors the historical wiring ``TaskList(store=self._tasks_store)``.
    """

    name: str = "task_list"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        if runtime.tasks_store is None:
            raise RuntimeError(
                "TaskListFactory.build requires runtime.tasks_store."
            )
        return TaskList(store=runtime.tasks_store)


class TaskStopFactory:
    """Factory for the ``task_stop`` tool.

    Mirrors the historical wiring
    ``TaskStop(store=..., running=..., running_shells=...)``.
    """

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
    """Factory for the ``send_message`` tool (Phase A teams).

    Mirrors the historical wiring ``SendMessage(agent=self)``. The tool
    walks ``agent.team`` / ``agent._team_member_name`` at invoke time;
    outside a team the tool surfaces a clean ToolError. Teams feature
    gating (``cfg.teams.enabled``) is enforced by
    :meth:`Agent._auto_enable_send_message_for_team` /
    :meth:`Agent.join_team` — the factory itself just preserves the
    historical "always build" behaviour so user-pinned
    ``tools.enabled=["send_message"]`` configs keep working.
    """

    name: str = "send_message"

    def build(self, runtime: ToolRuntime) -> BaseTool:
        if runtime.agent is None:
            raise RuntimeError(
                "SendMessageFactory.build requires runtime.agent "
                "(the leader Agent back-reference)."
            )
        return SendMessage(agent=runtime.agent)


# Ordered list consumed by :meth:`Agent.__init__`. Order matches the
# historical if/elif chain so the registration sequence (and any
# downstream effects of registration order, e.g. tool-list logging)
# stay byte-identical to the pre-Task-6 code path.
STATEFUL_TOOL_FACTORIES: list[StatefulToolFactory] = [
    TodoWriteFactory(),
    AskUserQuestionFactory(),
    TaskCreateFactory(),
    TaskGetFactory(),
    TaskListFactory(),
    TaskStopFactory(),
    SendMessageFactory(),
]
