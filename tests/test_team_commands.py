"""``/team`` slash command — dispatch verbs, error mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.cli.commands import build_default_registry, dispatch
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.commands.team import TeamCommand
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
        storage=SessionStorage(tmp_path / "sessions.db"),
    )


@pytest.mark.asyncio
async def test_team_command_help_when_no_args(tmp_path: Path) -> None:
    cmd = TeamCommand()
    result = await cmd.handle("", _agent(tmp_path))
    assert result.handled is True
    assert "/team" in result.text
    assert "create" in result.text


@pytest.mark.asyncio
async def test_team_create_sets_active_team(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    result = await cmd.handle("create demo", agent)
    assert "team 'demo' created" in result.text
    assert agent.team is not None


@pytest.mark.asyncio
async def test_team_send_to_unknown_member_returns_error(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("send ghost hello", agent)
    assert "team error" in result.text
    assert "ghost" in result.text


@pytest.mark.asyncio
async def test_team_list_includes_active_marker(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("list", agent)
    assert "demo" in result.text
    # Active team is prefixed with '* '
    assert "* demo" in result.text


@pytest.mark.asyncio
async def test_team_help_subcommand(tmp_path: Path) -> None:
    cmd = TeamCommand()
    result = await cmd.handle("help", _agent(tmp_path))
    assert "subcommand" in result.text


@pytest.mark.asyncio
async def test_team_unknown_subcommand(tmp_path: Path) -> None:
    cmd = TeamCommand()
    result = await cmd.handle("zonk", _agent(tmp_path))
    assert "unknown subcommand" in result.text


@pytest.mark.asyncio
async def test_dispatch_team_through_registry(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    r = build_default_registry()
    result = await dispatch("/team", agent, r)
    assert result.handled is True
    assert "subcommand" in result.text


@pytest.mark.asyncio
async def test_team_delete_clears_team(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    assert agent.team is not None
    result = await cmd.handle("delete", agent)
    assert "deleted" in result.text
    assert agent.team is None
