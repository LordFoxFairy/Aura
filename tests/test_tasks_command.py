"""/tasks — slash command listing subagent tasks.

Renders a tiny fixed-width table; exercised here via the Agent-bound handle
directly. Sorting by ``-started_at`` gives newest-first which is what a
user scanning "what did I kick off recently" expects.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.commands.tasks import TasksCommand
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel


def _agent(tmp_path: Path) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    return Agent(
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
    agent.close()


@pytest.mark.asyncio
async def test_tasks_command_lists_running_tasks(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent._tasks_store.create(description="scan repo", prompt="x")
    cmd = TasksCommand()
    result = await cmd.handle("", agent)
    assert result.handled is True
    assert "scan repo" in result.text
    assert "running" in result.text
    agent.close()


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
    agent.close()
