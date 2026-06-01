"""Phase 2 Tasks 5+6 — :class:`StatefulToolFactory` Protocol + 7 factories.

These tests pin down the contract for all 7 stateful-tool factories
landed by Phase 2: TodoWrite (pilot, Task 5), and the 6 added in Task 6
(AskUserQuestion, TaskCreate, TaskGet, TaskList, TaskStop,
SendMessage). They verify:

- the Protocol is ``runtime_checkable`` so ``isinstance`` works at
  registration time and in tests;
- each factory satisfies the Protocol shape (``name`` + ``build``);
- each factory wires the right dependencies through from
  :class:`ToolRuntime` (identity check, not equality, so we catch any
  accidental fresh-construct bugs);
- the registry constant ``STATEFUL_TOOL_FACTORIES`` lists every factory
  in a stable order so ``Agent.__init__``'s loop is deterministic.

Tests deliberately avoid building an :class:`Agent` — the whole point
of the factory pattern is that wiring is exercisable in isolation.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aura.application.loop_state import LoopState
from aura.application.runtime import (
    STATEFUL_TOOL_FACTORIES,
    AskUserQuestionFactory,
    SendMessageFactory,
    StatefulToolFactory,
    TaskCreateFactory,
    TaskGetFactory,
    TaskListFactory,
    TaskStopFactory,
    TodoWriteFactory,
    ToolRuntime,
)
from aura.application.session import AgentSession
from aura.application.tasks.spawn import SubagentSpawner
from aura.application.tasks.store import TasksStore
from aura.config.schema import AuraConfig
from aura.domain.todos import TodoItem
from aura.infrastructure.persistence.storage import SessionStorage
from aura.tools.ask_user import AskUserQuestion, FormQuestionDict
from aura.tools.send_message import SendMessage
from aura.tools.task_create import TaskCreate
from aura.tools.task_get import TaskGet
from aura.tools.task_list import TaskList
from aura.tools.task_stop import TaskStop
from aura.tools.todo_write import TodoWrite
from tests.conftest import FakeChatModel


def test_stateful_tool_factory_is_runtime_checkable() -> None:
    """Protocol must be ``@runtime_checkable`` so ``isinstance`` works.

    The eventual registration loop in :class:`Agent` will guard
    ``factory.build(...)`` calls behind an ``isinstance`` check; that
    guard relies on this property of the Protocol.
    """
    factory = TodoWriteFactory()
    assert isinstance(factory, StatefulToolFactory)


def test_todo_write_factory_satisfies_protocol_shape() -> None:
    """Pilot factory must expose ``name`` (str) and a callable ``build``.

    These are the two members the registration loop reads — pinning
    the names + types here catches refactor drift before it bleeds
    into Agent wiring.
    """
    factory = TodoWriteFactory()
    assert factory.name == "todo_write"
    assert callable(factory.build)


def test_todo_write_factory_builds_tool_wired_to_runtime_state() -> None:
    """``build(runtime)`` returns a TodoWrite bound to runtime.state.

    The whole reason ``ToolRuntime`` exists is to thread the live
    :class:`LoopState` through to the tool. This test confirms the
    factory does NOT construct a fresh state — it forwards the one
    handed in. Equivalent to today's ``TodoWrite(state=self._state)``
    line in ``Agent.__init__``.
    """
    state = LoopState()
    runtime = ToolRuntime(state=state)
    factory = TodoWriteFactory()

    tool = factory.build(runtime)

    assert isinstance(tool, TodoWrite)
    # Identity check — not equality. The whole point is that mutations
    # the tool makes to ``state.slots.todos`` must show up on the
    # caller's LoopState.
    assert tool.state is state


async def test_built_todo_write_tool_writes_to_runtime_state() -> None:
    """End-to-end: build via factory, invoke, observe state mutation.

    Mirrors ``test_todo_write.test_single_pending_todo_sets_state_and_returns_message``
    but reaches the tool via the factory path. Confirms behaviour is
    byte-identical to today's hand-wired construction.
    """
    state = LoopState()
    runtime = ToolRuntime(state=state)
    tool = TodoWriteFactory().build(runtime)

    out = await tool.ainvoke(
        {"todos": [{"content": "a", "status": "pending", "active_form": "Doing a"}]}
    )

    assert state.slots.todos == [
        TodoItem(content="a", status="pending", active_form="Doing a")
    ]
    assert out == {"message": "Todos updated."}


def test_tool_runtime_optional_fields_default_to_none() -> None:
    """ToolRuntime is frozen with optional dependencies defaulting to None.

    The whole point of the optional fields is that a factory which
    only needs ``state`` (like the pilot) doesn't have to invent
    placeholder objects for the other dependencies. Confirms the
    shape so future factories can rely on the defaults.
    """
    state = LoopState()
    runtime = ToolRuntime(state=state)

    assert runtime.state is state
    assert runtime.asker is None
    assert runtime.tasks_store is None
    assert runtime.spawner is None
    assert runtime.running_tasks is None
    assert runtime.running_shells is None
    assert runtime.transcript_storage is None
    assert runtime.team_provider is None
    assert runtime.member_name_provider is None


# --- Task 6 factories ---------------------------------------------------


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate(
        {
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }
    )


def _stub_subagent_factory() -> SubagentSpawner[AgentSession]:
    """Minimal SubagentSpawner just for identity checks on the wiring.

    The factory is not actually invoked in these tests; we only need a
    real instance so :meth:`TaskCreateFactory.build` can hand it to
    :class:`TaskCreate`.
    """
    return SubagentSpawner(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        build_child=AgentSession,
        model_factory=lambda: FakeChatModel(),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )


async def _stub_asker(
    questions: list[FormQuestionDict],
) -> dict[str, str]:
    return {q.get("question", ""): "" for q in questions}


def test_ask_user_question_factory_wires_runtime_asker() -> None:
    """AskUserQuestionFactory hands runtime.asker straight to the tool.

    Mirrors the historical
    ``AskUserQuestion(asker=question_asker or _unavailable_question_asker)``.
    Identity check confirms no copy/wrap layer was inserted.
    """
    runtime = ToolRuntime(state=LoopState(), asker=_stub_asker)
    factory = AskUserQuestionFactory()

    tool = factory.build(runtime)

    assert isinstance(tool, AskUserQuestion)
    assert factory.name == "ask_user_question"
    assert tool.asker is _stub_asker


def test_ask_user_question_factory_rejects_missing_asker() -> None:
    """Factory raises if Agent.__init__ forgot the fallback asker.

    Defensive guard — ``_unavailable_question_asker`` is meant to be
    threaded in even when no CLI was injected, so a None asker is a
    misconfiguration not a runtime condition.
    """
    runtime = ToolRuntime(state=LoopState())
    with pytest.raises(RuntimeError, match="asker"):
        AskUserQuestionFactory().build(runtime)


def test_task_create_factory_wires_store_factory_running_storage() -> None:
    """TaskCreateFactory threads tasks_store, subagent_factory,
    running_tasks, transcript_storage straight through.

    Mirrors the historical
    ``TaskCreate(store=..., spawner=..., running=..., transcript_storage=...)``.
    """
    store = TasksStore()
    sub_factory = _stub_subagent_factory()
    running: dict[str, asyncio.Task[None]] = {}
    storage = SessionStorage(Path(":memory:"))
    runtime = ToolRuntime(
        state=LoopState(),
        tasks_store=store,
        spawner=sub_factory,
        running_tasks=running,
        transcript_storage=storage,
    )

    tool = TaskCreateFactory().build(runtime)

    assert isinstance(tool, TaskCreate)
    assert TaskCreateFactory().name == "task_create"
    assert tool.store is store
    assert tool.spawner is sub_factory
    # ``running`` is a PrivateAttr forwarded via the .running property.
    assert tool.running is running


def test_task_create_factory_rejects_missing_deps() -> None:
    """Factory rejects construction when any required dep is absent."""
    runtime = ToolRuntime(state=LoopState())  # nothing wired
    with pytest.raises(RuntimeError, match="tasks_store"):
        TaskCreateFactory().build(runtime)


def test_task_get_factory_wires_store() -> None:
    """TaskGetFactory hands runtime.tasks_store to TaskGet.

    Identity check — TaskGet must read from the SAME store the Agent
    writes to, otherwise task lookups silently miss.
    """
    store = TasksStore()
    runtime = ToolRuntime(state=LoopState(), tasks_store=store)
    factory = TaskGetFactory()

    tool = factory.build(runtime)

    assert isinstance(tool, TaskGet)
    assert factory.name == "task_get"
    assert tool.store is store


def test_task_list_factory_wires_store() -> None:
    """TaskListFactory hands runtime.tasks_store to TaskList."""
    store = TasksStore()
    runtime = ToolRuntime(state=LoopState(), tasks_store=store)
    factory = TaskListFactory()

    tool = factory.build(runtime)

    assert isinstance(tool, TaskList)
    assert factory.name == "task_list"
    assert tool.store is store


def test_task_stop_factory_wires_store_running_running_shells() -> None:
    """TaskStopFactory threads tasks_store + running_tasks + running_shells.

    Mirrors the historical
    ``TaskStop(store=..., running=..., running_shells=...)``.
    """
    store = TasksStore()
    running: dict[str, asyncio.Task[None]] = {}
    running_shells: dict[str, asyncio.subprocess.Process] = {}
    runtime = ToolRuntime(
        state=LoopState(),
        tasks_store=store,
        running_tasks=running,
        running_shells=running_shells,
    )

    tool = TaskStopFactory().build(runtime)

    assert isinstance(tool, TaskStop)
    assert TaskStopFactory().name == "task_stop"
    assert tool.store is store
    assert tool.running is running
    assert tool.running_shells is running_shells


def test_task_stop_factory_rejects_missing_running_shells() -> None:
    """Factory rejects when ``running_shells`` is absent — Agent always
    constructs an empty dict, so None signals a misconfig not "no
    shells running"."""
    store = TasksStore()
    runtime = ToolRuntime(
        state=LoopState(),
        tasks_store=store,
        running_tasks={},
    )  # running_shells missing
    with pytest.raises(RuntimeError, match="running_shells"):
        TaskStopFactory().build(runtime)


def test_send_message_factory_wires_runtime_providers() -> None:
    """SendMessageFactory wires live team/member providers into SendMessage.

    Providers (not snapshots) so a join_team after tool construction is
    reflected on the next invoke.
    """
    sentinel_team = object()
    runtime = ToolRuntime(
        state=LoopState(),
        team_provider=lambda: sentinel_team,
        member_name_provider=lambda: "alice",
    )

    tool = SendMessageFactory().build(runtime)

    assert isinstance(tool, SendMessage)
    assert SendMessageFactory().name == "send_message"
    assert tool._team_provider() is sentinel_team
    assert tool._member_name_provider() == "alice"


def test_send_message_factory_rejects_missing_providers() -> None:
    """Factory rejects when team/member providers are not wired."""
    runtime = ToolRuntime(state=LoopState())
    with pytest.raises(RuntimeError, match="provider"):
        SendMessageFactory().build(runtime)


def test_stateful_tool_factories_registry_lists_all_seven_in_order() -> None:
    """``STATEFUL_TOOL_FACTORIES`` exports all 7 factories in the order
    Agent.__init__ depends on (matches the historical if/elif sequence).

    Order matters because tool registration order can be observed via
    ``ToolRegistry.tools()`` iteration; pinning it here catches any
    accidental reorder that would change downstream logging / tool
    schemas the LLM sees.
    """
    names = [factory.name for factory in STATEFUL_TOOL_FACTORIES]

    assert names == [
        "todo_write",
        "ask_user_question",
        "task_create",
        "task_get",
        "task_list",
        "task_stop",
        "send_message",
    ]
    # All entries satisfy the Protocol — the @runtime_checkable Protocol
    # check guards the registration loop in Agent.__init__.
    for factory in STATEFUL_TOOL_FACTORIES:
        assert isinstance(factory, StatefulToolFactory)
