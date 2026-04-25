"""Tests for F-06-003: ``needs_auth`` MCP server state.

The classifier (:func:`aura.core.mcp.manager._is_needs_auth_error`) routes
auth-failure exceptions to ``needs_auth`` instead of plain ``error``;
the manager's connect path then:

- sets ``state="needs_auth"``;
- journals ``mcp_connect_needs_auth`` (distinct from ``mcp_connect_failed``);
- skips auto-reconnect (retrying a 401 burns rate budget without
  learning anything new).

Plain network/server errors keep their original ``error`` semantics
including the remote-transport exponential-backoff reconnect ladder.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aura.config.schema import MCPServerConfig
from aura.core.mcp.manager import (
    MCPManager,
    _is_needs_auth_error,
)

# ---------------------------------------------------------------------------
# Pure classifier
# ---------------------------------------------------------------------------


class _CodedError(Exception):
    """Mirror MCP SDK's coded error shape: ``exc.code`` is the JSON-RPC
    error code; ``str(exc)`` is the message."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def test_classifier_matches_jsonrpc_minus_32001() -> None:
    """The MCP spec'd code for "Authentication required"."""
    assert _is_needs_auth_error(_CodedError(-32001, "anything")) is True


def test_classifier_matches_401_substring() -> None:
    assert _is_needs_auth_error(RuntimeError("HTTP 401 Unauthorized")) is True


def test_classifier_matches_403_substring() -> None:
    assert _is_needs_auth_error(RuntimeError("403 forbidden")) is True


def test_classifier_matches_oauth_substring() -> None:
    assert _is_needs_auth_error(RuntimeError("OAuth flow incomplete")) is True


def test_classifier_matches_unauthorized_text() -> None:
    assert _is_needs_auth_error(RuntimeError("unauthorized request")) is True


def test_classifier_rejects_generic_runtime_error() -> None:
    """Connection refused / DNS failure / etc must NOT promote to
    ``needs_auth`` — they should hit the auto-reconnect ladder."""
    assert _is_needs_auth_error(RuntimeError("connection refused")) is False
    assert _is_needs_auth_error(RuntimeError("dns lookup failed")) is False


def test_classifier_other_codes_not_needs_auth() -> None:
    """Non-auth JSON-RPC codes don't trigger the route."""
    assert _is_needs_auth_error(_CodedError(-32600, "Invalid Request")) is False
    assert _is_needs_auth_error(_CodedError(-32601, "Method not found")) is False


# ---------------------------------------------------------------------------
# Manager connect path → state transition
# ---------------------------------------------------------------------------


async def _empty_list(*_args: Any, **_kwargs: Any) -> list[Any]:
    return []


def _wire_fake_client(
    monkeypatch: pytest.MonkeyPatch, fake_client: Any,
) -> None:
    from aura.core.mcp import manager as manager_mod

    monkeypatch.setattr(
        manager_mod, "MultiServerMCPClient", lambda _cfg: fake_client,
    )
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_empty_list),
    )
    monkeypatch.setattr(
        MCPManager, "_list_resources", staticmethod(_empty_list),
    )


@pytest.mark.asyncio
async def test_auth_error_promotes_to_needs_auth_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """A 401 from get_tools sets state=needs_auth + journals
    ``mcp_connect_needs_auth`` instead of ``mcp_connect_failed``."""
    from aura.core.persistence import journal

    log = tmp_path / "j.jsonl"
    journal.configure(log)

    fake_client = MagicMock()
    fake_client.get_tools = AsyncMock(
        side_effect=_CodedError(-32001, "Authentication required"),
    )
    _wire_fake_client(monkeypatch, fake_client)

    mgr = MCPManager([
        MCPServerConfig(
            name="gh", transport="streamable_http", url="https://api.example.com/mcp",
        ),
    ])
    await mgr.start_all()

    statuses = mgr.status()
    assert len(statuses) == 1
    assert statuses[0].state == "needs_auth"
    # error_message is surfaced for needs_auth so the operator sees the
    # underlying provider text.
    assert statuses[0].error_message is not None
    assert "Authentication required" in statuses[0].error_message

    events = [
        json.loads(line)
        for line in log.read_text().splitlines()
        if line.strip()
    ]
    by_event = {e["event"] for e in events}
    assert "mcp_connect_needs_auth" in by_event
    assert "mcp_connect_failed" not in by_event
    journal.reset()


@pytest.mark.asyncio
async def test_generic_error_stays_error_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """A generic (non-auth) failure remains the plain ``error`` state."""
    from aura.core.persistence import journal

    journal.configure(tmp_path / "j.jsonl")

    fake_client = MagicMock()
    fake_client.get_tools = AsyncMock(
        side_effect=RuntimeError("connection refused"),
    )
    _wire_fake_client(monkeypatch, fake_client)

    mgr = MCPManager([
        MCPServerConfig(name="gh", command="npx", args=[]),
    ])
    await mgr.start_all()

    statuses = mgr.status()
    assert statuses[0].state == "error"
    assert statuses[0].error_message is not None
    assert "connection refused" in statuses[0].error_message
    journal.reset()


@pytest.mark.asyncio
async def test_needs_auth_skips_auto_reconnect_for_remote_transports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """A 401 on an SSE/streamable_http server must NOT schedule the
    exponential-backoff reconnect loop — retrying a 401 won't fix it."""
    from aura.core.persistence import journal

    journal.configure(tmp_path / "j.jsonl")

    fake_client = MagicMock()
    fake_client.get_tools = AsyncMock(
        side_effect=RuntimeError("HTTP 401 Unauthorized"),
    )
    _wire_fake_client(monkeypatch, fake_client)

    mgr = MCPManager([
        MCPServerConfig(
            name="api", transport="sse", url="https://api.example.com/mcp",
        ),
    ])
    await mgr.start_all()

    # No reconnect task spawned for needs_auth.
    assert "api" not in mgr._reconnect_tasks
    journal.reset()


@pytest.mark.asyncio
async def test_generic_error_on_remote_DOES_reconnect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """Regression: a non-auth error on remote transport still triggers
    the auto-reconnect ladder (the original behaviour)."""
    from aura.core.persistence import journal

    journal.configure(tmp_path / "j.jsonl")

    fake_client = MagicMock()
    fake_client.get_tools = AsyncMock(side_effect=RuntimeError("502 bad gateway"))
    _wire_fake_client(monkeypatch, fake_client)

    mgr = MCPManager([
        MCPServerConfig(
            name="api", transport="sse", url="https://api.example.com/mcp",
        ),
    ])
    await mgr.start_all()

    # The reconnect task IS scheduled (502 isn't an auth signal).
    assert "api" in mgr._reconnect_tasks
    # Cancel the spawned task so the test exits cleanly.
    mgr._reconnect_tasks["api"].cancel()
    journal.reset()


def test_status_dataclass_carries_needs_auth_state() -> None:
    """The ``MCPServerStatus.state`` field accepts ``"needs_auth"``
    (the literal union covers it). Pin so a future literal narrowing
    doesn't drop the value."""
    from aura.core.mcp.manager import MCPServerStatus

    s = MCPServerStatus(
        name="x",
        transport="sse",
        state="needs_auth",
        error_message="401",
        tool_count=0,
        resource_count=0,
        prompt_count=0,
    )
    assert s.state == "needs_auth"
