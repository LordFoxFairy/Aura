"""Tests for the ``/mcp`` slash command — in-REPL MCP control surface.

Uses a lightweight ``_FakeAgent`` holding only the ``mcp_manager`` slot,
which is all :class:`MCPCommand` touches. This avoids spinning up the full
AgentSession rig (storage, LLM fake, context builder) for what is pure command-
dispatch + plain-text formatting logic.

The manager itself is real (``MCPManager`` with mocked subprocess calls
via monkeypatch) where the test asserts on state transitions, and a hand-
rolled spy for the simpler cases where we only need to verify the command
dispatcher called the right method on the manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from aura.application.commands.mcp import MCPCommand
from aura.application.session import AgentSession
from aura.infrastructure.mcp.types import MCPServerConfig, MCPServerStatus


@dataclass
class _FakeAgent:
    """Minimal stand-in — MCPCommand only reads ``mcp_manager``."""

    mcp_manager: Any


def _as_agent(fake: _FakeAgent) -> AgentSession:
    """Cast helper — MCPCommand's handle() is typed as ``AgentSession`` but only
    touches ``mcp_manager``, so duck-typing is safe at runtime.
    """
    return cast(AgentSession, fake)


class _SpyManager:
    """Records method calls + returns canned status list / text results."""

    def __init__(
        self,
        statuses: list[MCPServerStatus] | None = None,
        *,
        enable_result: str = "",
        disable_result: str = "",
        reconnect_result: str = "",
        approve_result: str = "",
        revoke_result: str = "",
        known_names: list[str] | None = None,
    ) -> None:
        self._statuses = statuses or []
        self.enable_calls: list[str] = []
        self.disable_calls: list[str] = []
        self.reconnect_calls: list[str] = []
        self.approve_calls: list[str] = []
        self.revoke_calls: list[str] = []
        self._enable_result = enable_result
        self._disable_result = disable_result
        self._reconnect_result = reconnect_result
        self._approve_result = approve_result
        self._revoke_result = revoke_result
        self._known = known_names or [s.name for s in self._statuses]

    def status(self) -> list[MCPServerStatus]:
        return list(self._statuses)

    def known_server_names(self) -> list[str]:
        return list(self._known)

    async def enable(self, name: str) -> str:
        self.enable_calls.append(name)
        return self._enable_result or f"MCP server {name!r} enabled and connected"

    async def disable(self, name: str) -> str:
        self.disable_calls.append(name)
        return self._disable_result or f"MCP server {name!r} disabled"

    async def reconnect(self, name: str) -> str:
        self.reconnect_calls.append(name)
        return self._reconnect_result or f"MCP server {name!r} reconnected"

    async def approve(self, name: str) -> str:
        self.approve_calls.append(name)
        return self._approve_result or f"MCP server {name!r} approved"

    async def revoke(self, name: str) -> str:
        self.revoke_calls.append(name)
        return self._revoke_result or f"MCP server {name!r} revoked"


@pytest.mark.asyncio
async def test_mcp_empty_manager_prints_placeholder() -> None:
    agent = _FakeAgent(mcp_manager=_SpyManager(statuses=[]))
    result = await MCPCommand().handle("", _as_agent(agent))
    assert result.handled is True
    assert result.kind == "print"
    assert result.text == "(no MCP servers configured)"


@pytest.mark.asyncio
async def test_mcp_list_with_no_manager_attached() -> None:
    # When no MCP is configured the agent has no manager at all — the
    # /mcp list view must still render, with the same placeholder as an
    # empty-configured manager.
    agent = _FakeAgent(mcp_manager=None)
    result = await MCPCommand().handle("list", _as_agent(agent))
    assert result.handled is True
    assert "(no MCP servers configured)" in result.text


@pytest.mark.asyncio
async def test_mcp_list_renders_table_with_connected_and_disabled_rows() -> None:
    statuses = [
        MCPServerStatus(
            name="github",
            transport="stdio",
            state="connected",
            error_message=None,
            tool_count=5,
            resource_count=2,
            prompt_count=1,
        ),
        MCPServerStatus(
            name="disabled-srv",
            transport="sse",
            state="disabled",
            error_message=None,
            tool_count=0,
            resource_count=0,
            prompt_count=0,
        ),
    ]
    agent = _FakeAgent(mcp_manager=_SpyManager(statuses=statuses))
    result = await MCPCommand().handle("", _as_agent(agent))
    lines = result.text.splitlines()
    # Header row present.
    assert lines[0].startswith("NAME")
    assert "TRANSPORT" in lines[0]
    assert "STATUS" in lines[0]
    # Body rows contain per-server data + correct state.
    body = "\n".join(lines[1:])
    assert "github" in body
    assert "stdio" in body
    assert "connected" in body
    assert "5" in body  # tool count
    assert "disabled-srv" in body
    assert "sse" in body
    assert "disabled" in body


@pytest.mark.asyncio
async def test_mcp_list_error_row_surfaces_error_message() -> None:
    statuses = [
        MCPServerStatus(
            name="broken",
            transport="stdio",
            state="error",
            error_message="RuntimeError: cannot spawn child",
            tool_count=0,
            resource_count=0,
            prompt_count=0,
        ),
    ]
    agent = _FakeAgent(mcp_manager=_SpyManager(statuses=statuses))
    result = await MCPCommand().handle("list", _as_agent(agent))
    assert "broken" in result.text
    assert "error: RuntimeError: cannot spawn child" in result.text


@pytest.mark.asyncio
async def test_mcp_enable_delegates_to_manager_and_surfaces_result() -> None:
    spy = _SpyManager(
        statuses=[
            MCPServerStatus(
                name="github", transport="stdio", state="disabled",
                error_message=None, tool_count=0,
                resource_count=0, prompt_count=0,
            ),
        ],
        enable_result="MCP server 'github' enabled and connected",
    )
    agent = _FakeAgent(mcp_manager=spy)
    result = await MCPCommand().handle("enable github", _as_agent(agent))
    assert spy.enable_calls == ["github"]
    assert "connected" in result.text


@pytest.mark.asyncio
async def test_mcp_disable_delegates_to_manager() -> None:
    spy = _SpyManager(
        statuses=[
            MCPServerStatus(
                name="github", transport="stdio", state="connected",
                error_message=None, tool_count=3,
                resource_count=0, prompt_count=0,
            ),
        ],
    )
    agent = _FakeAgent(mcp_manager=spy)
    result = await MCPCommand().handle("disable github", _as_agent(agent))
    assert spy.disable_calls == ["github"]
    assert "disabled" in result.text


@pytest.mark.asyncio
async def test_mcp_reconnect_delegates_to_manager() -> None:
    spy = _SpyManager(
        statuses=[
            MCPServerStatus(
                name="github", transport="stdio", state="connected",
                error_message=None, tool_count=3,
                resource_count=0, prompt_count=0,
            ),
        ],
    )
    agent = _FakeAgent(mcp_manager=spy)
    result = await MCPCommand().handle("reconnect github", _as_agent(agent))
    assert spy.reconnect_calls == ["github"]
    assert "reconnected" in result.text


@pytest.mark.asyncio
async def test_mcp_enable_unknown_surfaces_manager_error() -> None:
    spy = _SpyManager(
        statuses=[
            MCPServerStatus(
                name="github", transport="stdio", state="connected",
                error_message=None, tool_count=0,
                resource_count=0, prompt_count=0,
            ),
        ],
        enable_result="no MCP server named 'nonexistent'; known: ['github']",
    )
    agent = _FakeAgent(mcp_manager=spy)
    result = await MCPCommand().handle("enable nonexistent", _as_agent(agent))
    assert "no MCP server named" in result.text
    assert "nonexistent" in result.text
    assert "github" in result.text  # known-names hint


@pytest.mark.asyncio
async def test_mcp_enable_without_target_is_usage_error() -> None:
    agent = _FakeAgent(mcp_manager=_SpyManager(statuses=[]))
    result = await MCPCommand().handle("enable", _as_agent(agent))
    assert result.kind == "print"
    assert "usage:" in result.text.lower()
    assert "enable" in result.text


@pytest.mark.asyncio
async def test_mcp_disable_without_target_is_usage_error() -> None:
    agent = _FakeAgent(mcp_manager=_SpyManager(statuses=[]))
    result = await MCPCommand().handle("disable", _as_agent(agent))
    assert "usage:" in result.text.lower()


@pytest.mark.asyncio
async def test_mcp_reconnect_without_target_is_usage_error() -> None:
    agent = _FakeAgent(mcp_manager=_SpyManager(statuses=[]))
    result = await MCPCommand().handle("reconnect", _as_agent(agent))
    assert "usage:" in result.text.lower()


@pytest.mark.asyncio
async def test_mcp_approve_delegates_to_manager() -> None:
    spy = _SpyManager(
        statuses=[
            MCPServerStatus(
                name="proj_srv", transport="stdio", state="unapproved",
                error_message=None, tool_count=0,
                resource_count=0, prompt_count=0,
            ),
        ],
        approve_result="MCP server 'proj_srv' approved",
    )
    agent = _FakeAgent(mcp_manager=spy)
    result = await MCPCommand().handle("approve proj_srv", _as_agent(agent))
    assert spy.approve_calls == ["proj_srv"]
    assert "approved" in result.text


@pytest.mark.asyncio
async def test_mcp_revoke_delegates_to_manager() -> None:
    spy = _SpyManager(
        statuses=[
            MCPServerStatus(
                name="proj_srv", transport="stdio", state="connected",
                error_message=None, tool_count=2,
                resource_count=0, prompt_count=0,
            ),
        ],
        revoke_result="MCP server 'proj_srv' revoked",
    )
    agent = _FakeAgent(mcp_manager=spy)
    result = await MCPCommand().handle("revoke proj_srv", _as_agent(agent))
    assert spy.revoke_calls == ["proj_srv"]
    assert "revoked" in result.text


@pytest.mark.asyncio
async def test_mcp_approve_without_target_is_usage_error() -> None:
    agent = _FakeAgent(mcp_manager=_SpyManager(statuses=[]))
    result = await MCPCommand().handle("approve", _as_agent(agent))
    assert "usage:" in result.text.lower()
    assert "approve" in result.text


@pytest.mark.asyncio
async def test_mcp_revoke_without_target_is_usage_error() -> None:
    agent = _FakeAgent(mcp_manager=_SpyManager(statuses=[]))
    result = await MCPCommand().handle("revoke", _as_agent(agent))
    assert "usage:" in result.text.lower()
    assert "revoke" in result.text


@pytest.mark.asyncio
async def test_mcp_help_lists_subcommands() -> None:
    agent = _FakeAgent(mcp_manager=None)
    result = await MCPCommand().handle("help", _as_agent(agent))
    for sub in ("list", "enable", "disable", "reconnect", "help"):
        assert sub in result.text, f"missing subcommand {sub!r} in /mcp help"


@pytest.mark.asyncio
async def test_mcp_unknown_subcommand_lists_valid_options() -> None:
    agent = _FakeAgent(mcp_manager=None)
    result = await MCPCommand().handle("nope", _as_agent(agent))
    assert "unknown" in result.text.lower()
    assert "'nope'" in result.text
    # Must list the valid subcommands so the operator can correct the typo.
    for sub in ("list", "enable", "disable", "reconnect", "help"):
        assert sub in result.text


@pytest.mark.asyncio
async def test_mcp_toggle_without_manager_returns_friendly_error() -> None:
    # enable/disable/reconnect require a manager — without one they must
    # print a friendly message (no AttributeError or traceback).
    agent = _FakeAgent(mcp_manager=None)
    result = await MCPCommand().handle("enable foo", _as_agent(agent))
    assert result.handled is True
    assert "no mcp manager" in result.text.lower()


def test_mcp_command_registered_in_default_registry() -> None:
    """``build_default_registry`` must include ``/mcp``."""
    from aura.application.commands.factory import build_default_registry

    reg = build_default_registry()
    names = [c.name for c in reg.list()]
    assert "/mcp" in names


def test_mcp_command_owned_by_capabilities_module() -> None:
    assert MCPCommand.__module__ == "aura.application.commands.mcp"


def test_mcp_command_has_expected_surface() -> None:
    cmd = MCPCommand()
    assert cmd.name == "/mcp"
    assert cmd.source == "builtin"
    assert cmd.description


class _ReloadSpyManager(_SpyManager):
    """Adds a ``reload`` seam so we can drive the ``/mcp reload`` branch."""

    def __init__(
        self,
        statuses: list[MCPServerStatus] | None = None,
        *,
        reload_result: str = "+1 -0",
    ) -> None:
        super().__init__(statuses=statuses)
        self.reload_calls: list[list[MCPServerConfig]] = []
        self._reload_result = reload_result

    async def reload(self, configs: list[MCPServerConfig]) -> str:
        self.reload_calls.append(list(configs))
        return self._reload_result


@pytest.mark.asyncio
async def test_mcp_reload_without_manager_returns_friendly_error() -> None:
    """``/mcp reload`` with no manager must degrade to a message, not crash."""
    agent = _FakeAgent(mcp_manager=None)
    result = await MCPCommand().handle("reload", _as_agent(agent))
    assert result.handled is True
    assert result.kind == "print"
    assert "no MCP manager" in result.text


@pytest.mark.asyncio
async def test_mcp_reload_passes_loaded_configs_and_surfaces_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/mcp reload`` must re-read the store and hand the configs to the manager."""
    loaded: list[MCPServerConfig] = []
    monkeypatch.setattr(
        "aura.application.commands.mcp.mcp_store.load", lambda: loaded
    )
    spy = _ReloadSpyManager(reload_result="+2 -1")
    agent = _FakeAgent(mcp_manager=spy)
    result = await MCPCommand().handle("reload", _as_agent(agent))
    assert spy.reload_calls == [loaded]
    assert result.text == "+2 -1"
    assert result.kind == "print"


@pytest.mark.asyncio
async def test_mcp_reload_store_failure_is_swallowed_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store read blowing up must surface as a print, never propagate out."""

    def _boom() -> list[MCPServerConfig]:
        raise RuntimeError("corrupt mcp_store.json")

    monkeypatch.setattr("aura.application.commands.mcp.mcp_store.load", _boom)
    spy = _ReloadSpyManager()
    agent = _FakeAgent(mcp_manager=spy)
    result = await MCPCommand().handle("reload", _as_agent(agent))
    assert result.handled is True
    assert "reload failed" in result.text
    assert "corrupt mcp_store.json" in result.text
    # Manager.reload must NOT run when the config load fails.
    assert spy.reload_calls == []


@pytest.mark.asyncio
async def test_mcp_reload_is_idempotent_across_repeated_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated reloads stay deterministic — each forwards a fresh config read."""
    monkeypatch.setattr(
        "aura.application.commands.mcp.mcp_store.load", lambda: []
    )
    spy = _ReloadSpyManager(reload_result="+0 -0")
    agent = _FakeAgent(mcp_manager=spy)
    first = await MCPCommand().handle("reload", _as_agent(agent))
    second = await MCPCommand().handle("reload", _as_agent(agent))
    assert first.text == second.text == "+0 -0"
    assert len(spy.reload_calls) == 2


@pytest.mark.asyncio
async def test_mcp_list_appends_unapproved_approval_hint() -> None:
    """Unapproved project-layer servers must get a call-to-action footer so the
    operator knows they exist and how to load them."""
    statuses = [
        MCPServerStatus(
            name="connected-srv", transport="stdio", state="connected",
            error_message=None, tool_count=1, resource_count=0, prompt_count=0,
        ),
        MCPServerStatus(
            name="pending-a", transport="stdio", state="unapproved",
            error_message=None, tool_count=0, resource_count=0, prompt_count=0,
        ),
        MCPServerStatus(
            name="pending-b", transport="sse", state="unapproved",
            error_message=None, tool_count=0, resource_count=0, prompt_count=0,
        ),
    ]
    agent = _FakeAgent(mcp_manager=_SpyManager(statuses=statuses))
    result = await MCPCommand().handle("list", _as_agent(agent))
    assert result.kind == "view"
    assert "unapproved project-layer servers: pending-a, pending-b" in result.text
    assert "/mcp approve <name>" in result.text


@pytest.mark.asyncio
async def test_mcp_list_no_hint_when_nothing_unapproved() -> None:
    """A clean fleet must not show the approval footer — no false prompts."""
    statuses = [
        MCPServerStatus(
            name="ok", transport="stdio", state="connected",
            error_message=None, tool_count=0, resource_count=0, prompt_count=0,
        ),
    ]
    agent = _FakeAgent(mcp_manager=_SpyManager(statuses=statuses))
    result = await MCPCommand().handle("list", _as_agent(agent))
    assert "unapproved" not in result.text
    assert "/mcp approve" not in result.text


@pytest.mark.asyncio
async def test_mcp_list_truncates_overlong_error_message() -> None:
    """A pathologically long error must be clipped with an ellipsis so one bad
    server can't blow out the status table width."""
    long_msg = "x" * 200
    statuses = [
        MCPServerStatus(
            name="broken", transport="stdio", state="error",
            error_message=long_msg, tool_count=0,
            resource_count=0, prompt_count=0,
        ),
    ]
    agent = _FakeAgent(mcp_manager=_SpyManager(statuses=statuses))
    result = await MCPCommand().handle("list", _as_agent(agent))
    assert "…" in result.text
    assert long_msg not in result.text
    # Clipped body stays well under the raw length.
    assert len("x" * 200) > max(len(line) for line in result.text.splitlines())


@pytest.mark.asyncio
async def test_mcp_list_error_row_without_message_shows_unknown() -> None:
    """An errored server with no message must still render a non-empty cell."""
    statuses = [
        MCPServerStatus(
            name="broken", transport="stdio", state="error",
            error_message=None, tool_count=0,
            resource_count=0, prompt_count=0,
        ),
    ]
    agent = _FakeAgent(mcp_manager=_SpyManager(statuses=statuses))
    result = await MCPCommand().handle("list", _as_agent(agent))
    assert "error: unknown error" in result.text


@pytest.mark.asyncio
async def test_mcp_toggle_joins_multiword_target() -> None:
    """A server name with spaces must be re-joined intact before dispatch."""
    spy = _SpyManager(
        statuses=[
            MCPServerStatus(
                name="my server", transport="stdio", state="disabled",
                error_message=None, tool_count=0,
                resource_count=0, prompt_count=0,
            ),
        ],
    )
    agent = _FakeAgent(mcp_manager=spy)
    result = await MCPCommand().handle("enable my server", _as_agent(agent))
    assert spy.enable_calls == ["my server"]
    assert result.kind == "print"


@pytest.mark.asyncio
async def test_mcp_toggle_strips_surrounding_whitespace_in_target() -> None:
    """Extra inter-token spaces collapse to a single clean target name."""
    spy = _SpyManager(
        statuses=[
            MCPServerStatus(
                name="github", transport="stdio", state="disabled",
                error_message=None, tool_count=0,
                resource_count=0, prompt_count=0,
            ),
        ],
    )
    agent = _FakeAgent(mcp_manager=spy)
    result = await MCPCommand().handle("enable   github   ", _as_agent(agent))
    assert spy.enable_calls == ["github"]
    assert result.handled is True


@pytest.mark.asyncio
async def test_mcp_toggle_unknown_action_reports_unknown_subcommand() -> None:
    """The ``_toggle`` defensive fall-through maps an unexpected action to the
    same unknown-subcommand error the dispatcher emits."""
    spy = _SpyManager(statuses=[])
    agent = _FakeAgent(mcp_manager=spy)
    result = await MCPCommand()._toggle(_as_agent(agent), "bogus", "x")
    assert "unknown" in result.text.lower()
    assert "'bogus'" in result.text
    assert spy.enable_calls == []
