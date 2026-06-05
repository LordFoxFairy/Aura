"""StatefulToolFactory registry — each name builds its runtime-wired tool, guarding missing deps."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aura.application.loop_state import LoopState
from aura.application.runtime import (
    STATEFUL_TOOL_FACTORIES,
    StatefulToolFactory,
    ToolRuntime,
)
from aura.application.session import AgentSession
from aura.application.tasks.spawn import SpawnContext, SubagentSpawner
from aura.application.tasks.store import TasksStore
from aura.application.teams.team_port import TeamPort
from aura.config.schema import AuraConfig
from aura.domain.team import TeamMessage, TeamMessageKind, TeamRecord
from aura.domain.todos import TodoItem
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.wire.events import CoordinationEvent
from aura.tools.ask_user import AskUserQuestion, FormQuestionDict
from aura.tools.send_message import SendMessage
from aura.tools.task_create import TaskCreate
from aura.tools.task_get import TaskGet
from aura.tools.task_list import TaskList
from aura.tools.task_stop import TaskStop
from aura.tools.todo_write import TodoWrite
from tests.conftest import FakeChatModel


def _factory(name: str) -> StatefulToolFactory:
    return next(f for f in STATEFUL_TOOL_FACTORIES if f.name == name)


def test_registry_lists_all_seven_in_stable_order() -> None:
    """Registration order is observable in tool schemas the LLM sees — pin it."""
    assert [f.name for f in STATEFUL_TOOL_FACTORIES] == [
        "todo_write",
        "ask_user_question",
        "task_create",
        "task_get",
        "task_list",
        "task_stop",
        "send_message",
    ]
    for f in STATEFUL_TOOL_FACTORIES:
        assert isinstance(f, StatefulToolFactory)
        assert callable(f.build)


def test_todo_write_builds_tool_wired_to_runtime_state() -> None:
    """build forwards the live LoopState (identity) so tool mutations reach the caller."""
    state = LoopState()
    tool = _factory("todo_write").build(ToolRuntime(state=state))
    assert isinstance(tool, TodoWrite)
    assert tool.state is state


async def test_built_todo_write_tool_writes_to_runtime_state() -> None:
    """End-to-end via the registry: invoking the built tool mutates the shared state."""
    state = LoopState()
    tool = _factory("todo_write").build(ToolRuntime(state=state))
    out = await tool.ainvoke(
        {"todos": [{"content": "a", "status": "pending", "active_form": "Doing a"}]}
    )
    assert state.slots.todos == [TodoItem(content="a", status="pending", active_form="Doing a")]
    assert out == {"message": "Todos updated."}


def test_tool_runtime_optional_fields_default_to_none() -> None:
    """Optional deps default to None so a state-only tool needs no placeholders."""
    runtime = ToolRuntime(state=LoopState())
    assert runtime.asker is None
    assert runtime.tasks_store is None
    assert runtime.spawner is None
    assert runtime.running_tasks is None
    assert runtime.running_shells is None
    assert runtime.transcript_storage is None
    assert runtime.team_provider is None
    assert runtime.member_name_provider is None


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate(
        {
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }
    )


def _stub_subagent_factory() -> SubagentSpawner[AgentSession]:
    return SubagentSpawner(
        SpawnContext(
            parent_config=_cfg(),
            parent_model_spec="openai:gpt-4o-mini",
            build_child=AgentSession,
            model_factory=lambda: FakeChatModel(),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )


async def _stub_asker(questions: list[FormQuestionDict]) -> dict[str, str]:
    return {q.get("question", ""): "" for q in questions}


def test_ask_user_question_wires_runtime_asker() -> None:
    """The asker is handed straight through (identity) — no copy/wrap layer."""
    tool = _factory("ask_user_question").build(ToolRuntime(state=LoopState(), asker=_stub_asker))
    assert isinstance(tool, AskUserQuestion)
    assert tool.asker is _stub_asker


def test_ask_user_question_rejects_missing_asker() -> None:
    """A None asker is a misconfig (the fallback is always threaded), not a runtime state."""
    with pytest.raises(RuntimeError, match="asker"):
        _factory("ask_user_question").build(ToolRuntime(state=LoopState()))


def test_task_create_wires_store_spawner_running_storage() -> None:
    """All four deps thread through by identity so lookups hit the live store."""
    store = TasksStore()
    sub_factory = _stub_subagent_factory()
    running: dict[str, asyncio.Task[None]] = {}
    storage = SessionStorage(Path(":memory:"))
    tool = _factory("task_create").build(
        ToolRuntime(
            state=LoopState(),
            tasks_store=store,
            spawner=sub_factory,
            running_tasks=running,
            transcript_storage=storage,
        )
    )
    assert isinstance(tool, TaskCreate)
    assert tool.store is store
    assert tool.spawner is sub_factory
    assert tool.running is running


def test_task_create_rejects_missing_deps() -> None:
    with pytest.raises(RuntimeError, match="tasks_store"):
        _factory("task_create").build(ToolRuntime(state=LoopState()))


def test_task_get_wires_store() -> None:
    """TaskGet must read the SAME store AgentSession writes, else lookups silently miss."""
    store = TasksStore()
    tool = _factory("task_get").build(ToolRuntime(state=LoopState(), tasks_store=store))
    assert isinstance(tool, TaskGet)
    assert tool.store is store


def test_task_list_wires_store() -> None:
    store = TasksStore()
    tool = _factory("task_list").build(ToolRuntime(state=LoopState(), tasks_store=store))
    assert isinstance(tool, TaskList)
    assert tool.store is store


def test_task_stop_wires_store_running_running_shells() -> None:
    store = TasksStore()
    running: dict[str, asyncio.Task[None]] = {}
    running_shells: dict[str, asyncio.subprocess.Process] = {}
    tool = _factory("task_stop").build(
        ToolRuntime(
            state=LoopState(),
            tasks_store=store,
            running_tasks=running,
            running_shells=running_shells,
        )
    )
    assert isinstance(tool, TaskStop)
    assert tool.store is store
    assert tool.running is running
    assert tool.running_shells is running_shells


def test_task_stop_rejects_missing_running_shells() -> None:
    """AgentSession always builds an empty dict, so None signals misconfig not "no shells"."""
    with pytest.raises(RuntimeError, match="running_shells"):
        _factory("task_stop").build(
            ToolRuntime(state=LoopState(), tasks_store=TasksStore(), running_tasks={})
        )


class _StubTeam:
    is_active: bool = False
    team: TeamRecord | None = None
    pending_protocol_events: tuple[CoordinationEvent, ...] = ()

    @property
    def storage(self) -> SessionStorage:
        raise NotImplementedError

    def post_message(self, msg: TeamMessage) -> None: ...
    def send(
        self,
        *,
        sender: str,
        recipient: str,
        body: str,
        kind: TeamMessageKind = "text",
    ) -> list[TeamMessage]:
        return []

    def confirm_shutdown(self, member_name: str, *, body: str = "") -> None: ...
    def drain_protocol_events(self) -> list[CoordinationEvent]:
        return []

    async def cleanup_session_teams(self) -> None: ...


def test_send_message_wires_runtime_providers() -> None:
    """Live providers (not snapshots) so a join_team after build reflects on next invoke."""
    sentinel_team: TeamPort = _StubTeam()
    tool = _factory("send_message").build(
        ToolRuntime(
            state=LoopState(),
            team_provider=lambda: sentinel_team,
            member_name_provider=lambda: "alice",
        )
    )
    assert isinstance(tool, SendMessage)
    assert tool._team_provider() is sentinel_team
    assert tool._member_name_provider() == "alice"


def test_send_message_rejects_missing_providers() -> None:
    with pytest.raises(RuntimeError, match="provider"):
        _factory("send_message").build(ToolRuntime(state=LoopState()))
