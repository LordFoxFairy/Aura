"""TEAMS feature gate — claude-code parity with ``isAgentSwarmsEnabled()``.

The teams (multi-agent swarm) subsystem is dormant by default. This
module pins the four behavioural guarantees of the gate:

1. With the default config (``teams.enabled=False``), the
   ``/team`` slash command is NOT registered with the default REPL
   command registry — invisible to ``/help`` and tab completion.
2. With ``teams.enabled=True`` the same registry call DOES register
   the ``TeamCommand`` — opt-in symmetry.
3. With the gate off, ``Agent.join_team`` raises ``RuntimeError``
   pointing at the config flag — programmatic API mirrors the
   slash-command surface.
4. The default ``ToolsConfig.enabled`` allowlist never carries
   ``send_message`` — that tool is auto-enabled inside ``join_team``,
   not ``__init__``. With the gate off this auto-enable is a no-op,
   so a default-built registry has no ``send_message``.
5. With the gate on, the auto-enable round-trip works end-to-end:
   join_team registers ``send_message``, the registry sees it, and
   the loop's bound model carries the new schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from aura.cli.commands import build_default_registry
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
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
    """With the gate open, ``join_team`` registers ``send_message``.

    Pins the Phase A.1 contract end-to-end under the gate: leader is
    not yet in a team → tool absent; leader joins → tool present
    (registry + loop's bound model in sync).
    """
    agent = _agent(tmp_path, teams_enabled=True)
    assert "send_message" not in agent._registry
    agent.join_team(manager=_stub_manager())
    assert "send_message" in agent._registry
    bound_names = {t.name for t in agent._registry.tools()}
    assert "send_message" in bound_names


def test_teams_disabled_build_default_registry_without_agent_omits_team_command() -> None:
    """No-agent registry build: gate defaults to off (claude-code-aligned).

    ``build_default_registry()`` with ``agent=None`` cannot consult a
    config, so the safe default (matching the ``teams.enabled=False``
    shipping default) is to skip ``TeamCommand``. Callers that need
    the gated command must pass an Agent built with the gate open.
    """
    registry = build_default_registry()
    assert "/team" not in {c.name for c in registry.list()}
