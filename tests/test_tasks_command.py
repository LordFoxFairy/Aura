"""/tasks — slash command listing subagent tasks.

Renders a tiny fixed-width table; exercised here via the AgentSession-bound handle
directly. Sorting by ``-started_at`` gives newest-first which is what a
user scanning "what did I kick off recently" expects.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aura.application.commands.tasks import (
    TaskGetCommand,
    TasksCommand,
    TaskStopCommand,
)
from aura.application.commands.tasks import TasksCommand as CapabilityTasksCommand
from aura.application.session import AgentSession
from aura.config.schema import AuraConfig
from aura.infrastructure.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel


def test_tasks_command_core_facade_points_at_capabilities_module() -> None:
    assert TasksCommand is CapabilityTasksCommand
    assert TasksCommand.__module__ == "aura.application.commands.tasks"


def _agent(tmp_path: Path) -> AgentSession:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    return AgentSession(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "db"),
    )


@pytest.mark.asyncio
async def test_tasks_command_prints_placeholder_when_empty(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    cmd = TasksCommand()
    result = await cmd.handle("", agent)
    assert result.handled is True
    assert result.kind == "print"
    assert result.text == "(no tasks)"
    await agent.aclose()


@pytest.mark.asyncio
async def test_tasks_command_lists_running_tasks(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent._tasks_store.create(description="scan repo", prompt="x")
    cmd = TasksCommand()
    result = await cmd.handle("", agent)
    assert result.handled is True
    assert "scan repo" in result.text
    assert "running" in result.text
    await agent.aclose()


@pytest.mark.asyncio
async def test_tasks_command_sorts_newest_first(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    older = agent._tasks_store.create(description="older", prompt="p1")
    newer = agent._tasks_store.create(description="newer", prompt="p2")
    # Force deterministic ordering regardless of monotonic-clock granularity.
    older.started_at = 100.0
    newer.started_at = 200.0
    cmd = TasksCommand()
    result = await cmd.handle("", agent)
    lines = result.text.splitlines()
    assert lines[0].endswith("newer")
    assert lines[1].endswith("older")
    await agent.aclose()


# --- /task-get + /task-stop: arg-matrix shared by both id-taking commands ---


@pytest.mark.parametrize("cmd_cls", [TaskGetCommand, TaskStopCommand])
@pytest.mark.parametrize("arg", ["", "   "])
async def test_id_command_blank_arg_prints_usage(
    tmp_path: Path, cmd_cls: type, arg: str,
) -> None:
    """Blank arg must surface usage, never silently resolve to an arbitrary task."""
    agent = _agent(tmp_path)
    result = await cmd_cls().handle(arg, agent)
    assert result.kind == "print"
    assert "usage:" in result.text and cmd_cls().name in result.text
    await agent.aclose()


@pytest.mark.parametrize("cmd_cls", [TaskGetCommand, TaskStopCommand])
async def test_id_command_unknown_prefix_reports_no_match(
    tmp_path: Path, cmd_cls: type,
) -> None:
    agent = _agent(tmp_path)
    result = await cmd_cls().handle("deadbeef", agent)
    assert result.kind == "print"
    assert "no task matches 'deadbeef'" in result.text
    await agent.aclose()


async def test_task_get_renders_every_optional_field(tmp_path: Path) -> None:
    """Each optional field appears only when set — duration/tool/result/error branches."""
    agent = _agent(tmp_path)
    rec = agent._tasks_store.create(description="scan repo", prompt="x")
    rec.agent_type = "explore"
    rec.status = "completed"
    rec.started_at = 100.0
    rec.finished_at = 103.5
    rec.progress.tool_count = 4
    rec.progress.recent_activities = ["bash: ls", "read: foo.py"]
    rec.final_result = "found 3 bugs"
    rec.error = "partial timeout"

    text = (await TaskGetCommand().handle(rec.id, agent)).text

    assert "agent_type  explore" in text
    assert "status      completed" in text
    assert "duration    3.50s" in text
    assert "tool_count  4" in text
    assert "recent      bash: ls, read: foo.py" in text
    assert "result      found 3 bugs" in text
    assert "error       partial timeout" in text
    await agent.aclose()


async def test_task_get_omits_unset_optionals_and_resolves_by_prefix(
    tmp_path: Path,
) -> None:
    """A still-running task: no duration/result/error lines; 8-hex prefix resolves it."""
    agent = _agent(tmp_path)
    rec = agent._tasks_store.create(description="live", prompt="x")
    text = (await TaskGetCommand().handle(rec.id[:8], agent)).text
    assert "status      running" in text
    assert "duration" not in text
    assert "result" not in text
    assert "error" not in text
    await agent.aclose()


async def test_task_get_ambiguous_prefix_refuses_to_guess(tmp_path: Path) -> None:
    """Two ids sharing a prefix must NOT silently resolve to one — data-loss risk."""
    agent = _agent(tmp_path)
    a = agent._tasks_store.create(description="a", prompt="x")
    b = agent._tasks_store.create(description="b", prompt="y")
    a.id = "feed0001"
    b.id = "feed0002"
    result = await TaskGetCommand().handle("feed", agent)
    assert "no task matches 'feed'" in result.text
    await agent.aclose()


async def test_task_stop_refuses_already_finished_task(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    rec = agent._tasks_store.create(description="done", prompt="x")
    rec.status = "completed"
    result = await TaskStopCommand().handle(rec.id, agent)
    assert "already completed" in result.text and "nothing to stop" in result.text
    await agent.aclose()


async def test_task_stop_cancels_running_task_with_no_live_handle(
    tmp_path: Path,
) -> None:
    """No in-flight asyncio handle (e.g. after resume) still marks the record cancelled."""
    agent = _agent(tmp_path)
    rec = agent._tasks_store.create(description="orphan", prompt="x")
    result = await TaskStopCommand().handle(rec.id, agent)
    assert "cancelled" in result.text
    stored = agent._tasks_store.get(rec.id)
    assert stored is not None and stored.status == "cancelled"
    await agent.aclose()


async def test_task_stop_cancels_live_handle(tmp_path: Path) -> None:
    """A live handle is cancelled cooperatively and reported cancelled."""
    agent = _agent(tmp_path)
    rec = agent._tasks_store.create(description="busy", prompt="x")

    async def _work() -> None:
        await asyncio.sleep(30)

    handle = asyncio.create_task(_work())
    agent.running_tasks[rec.id] = handle
    try:
        result = await TaskStopCommand().handle(rec.id, agent)
        assert "cancelled" in result.text
        assert handle.cancelled()
    finally:
        await agent.aclose()
