"""Phase 2 Task 5 — :class:`StatefulToolFactory` Protocol + pilot.

These tests pin down the contract before Task 6 migrates the rest of
the stateful tools off the if/elif chain in ``Agent.__init__``:

- the Protocol is ``runtime_checkable`` so ``isinstance`` works at
  registration time and in tests;
- the pilot :class:`TodoWriteFactory` satisfies the Protocol;
- ``TodoWriteFactory.build(runtime)`` produces a :class:`TodoWrite`
  whose ``state`` is the same instance handed in via
  :class:`ToolRuntime` — i.e. the factory wires the dependency rather
  than constructing a fresh state.

Tests deliberately avoid building an :class:`Agent` — the whole point
of the factory pattern is that wiring is exercisable in isolation.
"""
from __future__ import annotations

from aura.core.runtime import (
    StatefulToolFactory,
    TodoWriteFactory,
    ToolRuntime,
)
from aura.schemas.state import LoopState
from aura.schemas.todos import TodoItem
from aura.tools.todo_write import TodoWrite


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
    assert runtime.subagent_factory is None
    assert runtime.running_tasks is None
    assert runtime.running_shells is None
    assert runtime.transcript_storage is None
    assert runtime.agent is None
    assert runtime.team_manager is None
