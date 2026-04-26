"""Tests for MCPServerConfig schema + manager wiring across transports.

Covers the three transports the library supports — stdio, sse, and
streamable_http — and the cross-field validation that keeps each transport's
required fields honest (command XOR url).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from aura.config.schema import AuraConfigError, MCPServerConfig
from aura.core.mcp.manager import MCPManager

# --- schema validation -----------------------------------------------------


def test_stdio_config_with_command_validates() -> None:
    cfg = MCPServerConfig(name="gh", command="npx", args=["-y", "pkg"])
    assert cfg.transport == "stdio"
    assert cfg.command == "npx"
    assert cfg.url is None


def test_sse_config_with_url_validates() -> None:
    cfg = MCPServerConfig(
        name="remote", transport="sse", url="https://example.com/sse",
    )
    assert cfg.transport == "sse"
    assert cfg.url == "https://example.com/sse"
    assert cfg.command is None
    assert cfg.headers == {}


def test_streamable_http_config_with_url_and_headers_validates() -> None:
    cfg = MCPServerConfig(
        name="cloud",
        transport="streamable_http",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer abc"},
    )
    assert cfg.transport == "streamable_http"
    assert cfg.url == "https://example.com/mcp"
    assert cfg.headers == {"Authorization": "Bearer abc"}


def test_sse_missing_url_raises() -> None:
    with pytest.raises(ValidationError, match="'url' is required"):
        MCPServerConfig(name="bad", transport="sse")


def test_streamable_http_missing_url_raises() -> None:
    with pytest.raises(ValidationError, match="'url' is required"):
        MCPServerConfig(name="bad", transport="streamable_http")


def test_stdio_missing_command_raises() -> None:
    with pytest.raises(ValidationError, match="'command' is required"):
        MCPServerConfig(name="bad", transport="stdio")


def test_stdio_default_missing_command_raises() -> None:
    # transport defaults to stdio — still requires command.
    with pytest.raises(ValidationError, match="'command' is required"):
        MCPServerConfig(name="bad")


def test_sse_with_command_is_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="'command' is not valid"):
        MCPServerConfig(
            name="bad", transport="sse", url="https://x", command="npx",
        )


def test_stdio_with_url_is_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="'url' is not valid"):
        MCPServerConfig(
            name="bad", transport="stdio", command="npx", url="https://x",
        )


def test_unknown_transport_rejected_by_literal() -> None:
    # Runtime guard: even though the Literal forbids this statically, a
    # JSON config file on disk can carry any string; pydantic must reject.
    with pytest.raises(ValidationError):
        MCPServerConfig.model_validate(
            {"name": "bad", "transport": "carrier-pigeon", "command": "x"},
        )


# --- wiring: MCPManager._build_connections ---------------------------------


def test_build_connections_stdio() -> None:
    mgr = MCPManager([
        MCPServerConfig(name="gh", command="npx", args=["-y", "pkg"],
                        env={"TOKEN": "t"}),
    ])
    conns = mgr._build_connections()
    # F-06-004 — every connection now carries ``session_kwargs.message_handler``
    # for list-changed observability; assert on the transport-shape fields,
    # not strict equality.
    entry = conns["gh"]
    assert entry["transport"] == "stdio"
    assert entry["command"] == "npx"
    assert entry["args"] == ["-y", "pkg"]
    assert entry["env"] == {"TOKEN": "t"}
    assert callable(entry["session_kwargs"]["message_handler"])


def test_build_connections_sse_no_headers() -> None:
    mgr = MCPManager([
        MCPServerConfig(name="r", transport="sse", url="https://x/sse"),
    ])
    conns = mgr._build_connections()
    entry = conns["r"]
    assert entry["transport"] == "sse"
    assert entry["url"] == "https://x/sse"
    assert callable(entry["session_kwargs"]["message_handler"])


def test_build_connections_streamable_http_with_headers() -> None:
    mgr = MCPManager([
        MCPServerConfig(
            name="c",
            transport="streamable_http",
            url="https://x/mcp",
            headers={"Authorization": "Bearer t"},
        ),
    ])
    conns = mgr._build_connections()
    entry = conns["c"]
    assert entry["transport"] == "streamable_http"
    assert entry["url"] == "https://x/mcp"
    assert entry["headers"] == {"Authorization": "Bearer t"}
    assert callable(entry["session_kwargs"]["message_handler"])


def test_build_connections_mixed_transports() -> None:
    mgr = MCPManager([
        MCPServerConfig(name="local", command="npx", args=["x"]),
        MCPServerConfig(name="remote", transport="sse", url="https://a"),
        MCPServerConfig(
            name="cloud", transport="streamable_http", url="https://b",
            headers={"X-Key": "k"},
        ),
    ])
    conns = mgr._build_connections()
    assert set(conns) == {"local", "remote", "cloud"}
    assert conns["local"]["transport"] == "stdio"
    assert conns["remote"]["transport"] == "sse"
    assert conns["cloud"]["transport"] == "streamable_http"
    assert conns["cloud"]["headers"] == {"X-Key": "k"}


# --- library-capability gate -----------------------------------------------


def test_unsupported_transport_raises_aura_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.core.mcp import manager as manager_mod

    monkeypatch.setattr(
        manager_mod, "_supported_transports", lambda: {"stdio"},
    )
    cfg = MCPServerConfig(
        name="remote", transport="sse", url="https://example.com/sse",
    )
    with pytest.raises(AuraConfigError, match="doesn't support"):
        MCPManager([cfg])


def test_disabled_server_bypasses_transport_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Disabled servers are filtered first, so even an unsupported transport
    # entry doesn't break startup as long as it's off.
    from aura.core.mcp import manager as manager_mod

    monkeypatch.setattr(
        manager_mod, "_supported_transports", lambda: {"stdio"},
    )
    cfg = MCPServerConfig(
        name="remote", transport="sse", url="https://x", enabled=False,
    )
    # Must not raise.
    MCPManager([cfg])


# --- end-to-end: manager passes transport dicts through to the library ----


@pytest.mark.asyncio
async def test_start_all_passes_sse_connection_dict_to_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.core.mcp import manager as manager_mod

    captured: dict[str, Any] = {}

    def _capture(connections: dict[str, Any]) -> MagicMock:
        captured["connections"] = connections
        fake = MagicMock()
        fake.get_tools = AsyncMock(return_value=[])
        return fake

    async def _fake_list_prompts(client: Any, server_name: str) -> list[Any]:
        return []

    monkeypatch.setattr(manager_mod, "MultiServerMCPClient", _capture)
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_fake_list_prompts),
    )

    mgr = MCPManager([
        MCPServerConfig(
            name="remote",
            transport="sse",
            url="https://example.com/sse",
            headers={"Authorization": "Bearer abc"},
        ),
    ])
    await mgr.start_all()

    entry = captured["connections"]["remote"]
    assert entry["transport"] == "sse"
    assert entry["url"] == "https://example.com/sse"
    assert entry["headers"] == {"Authorization": "Bearer abc"}
    # F-06-004 — list-changed message handler attached to the connection.
    assert callable(entry["session_kwargs"]["message_handler"])
