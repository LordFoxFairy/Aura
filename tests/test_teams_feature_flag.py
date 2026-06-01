"""Teams feature gate: ``/team``, ``join_team``, and ``send_message`` opt-in semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from aura.application.commands.factory import build_default_registry
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.infrastructure.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel


def _cfg(*, teams_enabled: bool) -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "teams": {"enabled": teams_enabled},
    })


def _agent(tmp_path: Path, *, teams_enabled: bool) -> Agent:
    return Agent(
        config=_cfg(teams_enabled=teams_enabled),
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "sessions.db"),
    )


def _stub_manager() -> Any:
    mgr = MagicMock()
    mgr.is_active = True
    return mgr


def test_teams_disabled_by_default() -> None:
    """Sanity: shipped default ``AuraConfig`` has the gate off."""
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
    })
    assert cfg.teams.enabled is False


def test_teams_disabled_by_default_no_team_command_registered(
    tmp_path: Path,
) -> None:
    """Default-off config: ``/team`` is absent from the default registry."""
    agent = _agent(tmp_path, teams_enabled=False)
    registry = build_default_registry(agent)
    assert "/team" not in {c.name for c in registry.list()}


def test_teams_enabled_registers_team_command(tmp_path: Path) -> None:
    """Opt-in config: ``/team`` IS registered with the default registry."""
    agent = _agent(tmp_path, teams_enabled=True)
    registry = build_default_registry(agent)
    assert "/team" in {c.name for c in registry.list()}


def test_teams_disabled_join_team_raises(tmp_path: Path) -> None:
    """Programmatic ``Agent.join_team`` rejects with a config-pointer error."""
    agent = _agent(tmp_path, teams_enabled=False)
    with pytest.raises(RuntimeError) as excinfo:
        agent.join_team(manager=_stub_manager())
    msg = str(excinfo.value)
    assert "teams" in msg.lower()
    assert "teams.enabled" in msg


def test_teams_disabled_send_message_not_in_default_registry(
    tmp_path: Path,
) -> None:
    """Disabled gate: no ``send_message`` in the tool registry post-construction.

    The default ``ToolsConfig.enabled`` allowlist deliberately omits
    ``send_message`` (it is gated to inside-team usage), and the gate
    suppresses the auto-enable path that would otherwise add it. Net
    effect: the LLM never sees the tool.
    """
    agent = _agent(tmp_path, teams_enabled=False)
    assert "send_message" not in agent._registry


def test_teams_enabled_send_message_auto_enabled_on_join(
    tmp_path: Path,
) -> None:
    """With the gate open, ``join_team`` registers ``send_message`` end-to-end."""
    agent = _agent(tmp_path, teams_enabled=True)
    assert "send_message" not in agent._registry
    agent.join_team(manager=_stub_manager())
    assert "send_message" in agent._registry
    bound_names = {t.name for t in agent._registry.tools()}
    assert "send_message" in bound_names


def test_teams_disabled_build_default_registry_without_agent_omits_team_command() -> None:
    """No-agent registry build can't read config; safe default omits ``/team``."""
    registry = build_default_registry()
    assert "/team" not in {c.name for c in registry.list()}
