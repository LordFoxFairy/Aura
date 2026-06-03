"""``send_message`` auto-enables on ``join_team`` and unwinds on ``leave_team``."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from aura.application.session import AgentSession
from aura.config.schema import AuraConfig, ToolsConfig
from aura.infrastructure.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel


def _default_cfg() -> AuraConfig:
    """Default ``tools.enabled`` with ``teams.enabled=True`` for auto-enable path."""
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "teams": {"enabled": True},
    })


def _custom_allowlist_cfg(tools: list[str]) -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": tools},
        "teams": {"enabled": True},
    })


def _agent(tmp_path: Path, cfg: AuraConfig | None = None) -> AgentSession:
    return AgentSession(
        config=cfg or _default_cfg(),
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "sessions.db"),
    )


def _stub_manager() -> Any:
    """Minimal TeamManager surrogate — only ``is_active`` is touched here."""
    mgr = MagicMock()
    mgr.is_active = True
    return mgr


def test_send_message_appears_in_registry_after_join_team(tmp_path: Path) -> None:
    """Default config: join_team registers send_message + rebinds the loop."""
    agent = _agent(tmp_path)
    # Sanity: shipped default does NOT include send_message.
    assert "send_message" not in ToolsConfig().enabled
    assert "send_message" not in agent._registry
    agent.join_team(manager=_stub_manager())
    assert "send_message" in agent._registry
    # Bound model carries the new schema — the loop's ``_bound``
    # reference was replaced via ``_rebind_tools``.
    bound_names = {
        t.name for t in agent._registry.tools()
    }
    assert "send_message" in bound_names


def test_send_message_removed_from_registry_after_leave_team(
    tmp_path: Path,
) -> None:
    """leave_team unregisters + rebinds — LLM stops seeing send_message."""
    agent = _agent(tmp_path)
    agent.join_team(manager=_stub_manager())
    assert "send_message" in agent._registry
    agent.leave_team()
    assert "send_message" not in agent._registry


def test_user_explicit_allowlist_overrides_auto_enable(tmp_path: Path) -> None:
    """A user-pinned allowlist (without send_message) is respected.

    The auto-enable path is suppressed so the user's choice wins even
    when they pin an allowlist that excludes ``send_message``. The
    tool's outside-team error path is the user's safety net — they
    explicitly opted out, so an LLM call attempt would fail clean.
    """
    cfg = _custom_allowlist_cfg(["read_file", "bash"])
    agent = _agent(tmp_path, cfg=cfg)
    assert "send_message" not in agent._registry
    agent.join_team(manager=_stub_manager())
    # Still NOT in the registry — user's pin wins.
    assert "send_message" not in agent._registry


def test_user_allowlist_includes_send_message_kept_on_leave(
    tmp_path: Path,
) -> None:
    """If the user OPTED IN via tools.enabled, leave_team keeps the tool.

    This is the symmetric case to the override test: when the user
    pinned ``send_message`` themselves, we did NOT auto-add it (it
    was added at construction time), so we must NOT auto-remove it
    on leave either. Otherwise a /team delete in a session that
    deliberately wired send_message would silently strip the tool.
    """
    cfg = _custom_allowlist_cfg(["read_file", "bash", "send_message"])
    agent = _agent(tmp_path, cfg=cfg)
    assert "send_message" in agent._registry
    agent.join_team(manager=_stub_manager())
    assert "send_message" in agent._registry
    agent.leave_team()
    # User-opted-in tool is preserved.
    assert "send_message" in agent._registry


def test_double_join_is_idempotent(tmp_path: Path) -> None:
    """Joining the same team twice does not re-register send_message.

    Re-registration would raise (registry rejects duplicate names);
    the auto-enable code must short-circuit when the tool is already
    bound.
    """
    agent = _agent(tmp_path)
    mgr = _stub_manager()
    agent.join_team(manager=mgr)
    # Second join — idempotent (same manager).
    agent.join_team(manager=mgr)
    assert "send_message" in agent._registry
