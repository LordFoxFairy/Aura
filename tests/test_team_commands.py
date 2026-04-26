"""``/team`` slash command — dispatch verbs, error mapping."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from aura.cli.commands import build_default_registry, dispatch
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.commands.team import TeamCommand
from aura.core.persistence.storage import SessionStorage
from aura.core.teams.manager import TeamManager
from aura.core.teams.types import TeammateMember
from tests.conftest import FakeChatModel


def _members(agent: Agent) -> list[TeammateMember]:
    """Read the live team's members through the leader's TeamManager.

    The manager is stored on a private attribute by ``_ensure_manager``;
    this helper centralizes the cast so the test bodies stay readable.
    """
    mgr = cast(TeamManager, agent._team_manager)  # type: ignore[attr-defined]
    return mgr.list_members()


def _agent(tmp_path: Path, *, teams_enabled: bool = True) -> Agent:
    # ``teams.enabled=True`` is required from v0.18 onwards — the gate
    # (claude-code parity with isAgentSwarmsEnabled()) defaults to False
    # and would make ``Agent.join_team`` raise. These slash-dispatch tests
    # need the gate open so /team verbs reach their handlers.
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "teams": {"enabled": teams_enabled},
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
    # v0.18+ teams gate (claude-code parity with isAgentSwarmsEnabled()):
    # ``TeamCommand`` is only registered with the default registry when
    # the agent's config has ``teams.enabled=True``. Pass the agent so
    # the gate sees an opt-in config.
    r = build_default_registry(agent)
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


# ---------------------------------------------------------------------------
# /team add --backend <kind> CLI flag (v0.18.x)
# ---------------------------------------------------------------------------


@pytest.fixture
def _fake_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub OPENAI_API_KEY so the factory can build a model in tests.

    The ``/team add`` path runs the SubagentFactory, which validates a
    provider's credential env var even when the resulting model never
    fires (these tests don't pump events). A literal placeholder is
    enough — no network is hit.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-fake-key")


@pytest.mark.asyncio
async def test_team_add_default_backend_is_in_process(
    tmp_path: Path, _fake_openai_key: None,
) -> None:
    """``/team add alice`` defaults to in_process — backend not surfaced."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("add alice", agent)
    assert "added" in result.text
    # backend suffix is suppressed for the default value (terse output).
    assert "backend=" not in result.text
    members = _members(agent)
    assert any(m.name == "alice" and m.backend_type == "in_process"
               for m in members)


@pytest.mark.asyncio
async def test_team_add_explicit_in_process_backend(
    tmp_path: Path, _fake_openai_key: None,
) -> None:
    """``--backend in_process`` is accepted explicitly + persisted."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("add bob --backend in_process", agent)
    assert "added" in result.text
    member = next(m for m in _members(agent)
                  if m.name == "bob")
    assert member.backend_type == "in_process"


@pytest.mark.asyncio
async def test_team_add_pane_backend_outside_tmux_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--backend pane`` without ``$TMUX`` surfaces a friendly error.

    The registry's ``pane_backend_available()`` walks ``$TMUX`` + the
    PATH; we strip ``$TMUX`` so the gate fires regardless of host env.
    """
    monkeypatch.delenv("TMUX", raising=False)
    # Reset registry singletons so the env-gate check actually runs
    # (it's bypassed if a singleton was cached by a prior test).
    from aura.core.teams.backends.registry import _reset_for_tests
    _reset_for_tests()
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("add carol --backend pane", agent)
    assert "error" in result.text.lower()
    assert "tmux" in result.text.lower()
    # No half-spawned member: the registry check fires before we touch
    # team state.
    assert all(m.name != "carol" for m in _members(agent))


@pytest.mark.asyncio
async def test_team_add_unknown_backend_rejected(tmp_path: Path) -> None:
    """Typoed backend value gives a usage hint, not a stack trace."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("add dan --backend zomg", agent)
    assert "unknown backend" in result.text
    assert "in_process" in result.text  # hint lists valid values
    assert all(m.name != "dan" for m in _members(agent))


@pytest.mark.asyncio
async def test_team_add_backend_flag_anywhere(
    tmp_path: Path, _fake_openai_key: None,
) -> None:
    """``--backend`` can precede positional args — flag-position-agnostic."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    # Flag interleaved between positional args.
    result = await cmd.handle(
        "add eve general-purpose --backend in_process", agent,
    )
    assert "added" in result.text
    member = next(m for m in _members(agent)
                  if m.name == "eve")
    assert member.agent_type == "general-purpose"
    assert member.backend_type == "in_process"


@pytest.mark.asyncio
async def test_team_add_backend_missing_value(tmp_path: Path) -> None:
    """Trailing ``--backend`` with no value gives a usage hint."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("add frank --backend", agent)
    assert "usage" in result.text.lower()
    assert "backend" in result.text.lower()
