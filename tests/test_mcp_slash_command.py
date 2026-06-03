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
from aura.infrastructure.mcp.manager import MCPServerStatus


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
