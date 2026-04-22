"""Tests for aura.core.mcp.manager — MCPManager wraps MultiServerMCPClient."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from aura.config.schema import MCPServerConfig
from aura.core.mcp.manager import MCPManager


class _P(BaseModel):
    q: str = ""


def _fake_tool(name: str) -> StructuredTool:
    async def _coro(q: str = "") -> dict[str, Any]:
        return {}
    return StructuredTool(
        name=name,
        description="fake",
        args_schema=_P,
        coroutine=_coro,
    )


@pytest.mark.asyncio
async def test_start_all_empty_config_returns_empty() -> None:
    mgr = MCPManager([])
    tools, commands = await mgr.start_all()
    assert tools == []
    assert commands == []


@pytest.mark.asyncio
async def test_start_all_single_server_wraps_tools_with_aura_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.get_tools = AsyncMock(return_value=[_fake_tool("search")])

    # No prompts for this minimal test — simulate an empty-list response.
    async def _fake_list_prompts(client: Any, server_name: str) -> list[Any]:
        return []

    from aura.core.mcp import manager as manager_mod

    monkeypatch.setattr(
        manager_mod, "MultiServerMCPClient", lambda cfg: fake_client,
    )
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_fake_list_prompts),
    )

    mgr = MCPManager([
        MCPServerConfig(name="gh", command="npx", args=["-y", "x"]),
    ])
    tools, commands = await mgr.start_all()

    assert len(tools) == 1
    t = tools[0]
    assert t.name == "mcp__gh__search"
    assert (t.metadata or {}).get("is_destructive") is True
    assert (t.metadata or {}).get("max_result_size_chars") == 30_000
    fake_client.get_tools.assert_awaited_once_with(server_name="gh")


@pytest.mark.asyncio
async def test_start_all_broken_server_graceful_degrade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    from aura.core import journal
    from aura.core.mcp import manager as manager_mod

    log_path = tmp_path / "journal.jsonl"
    journal.configure(log_path)

    fake_client = MagicMock()

    async def _get_tools(*, server_name: str) -> list[StructuredTool]:
        if server_name == "broken":
            raise RuntimeError("cannot connect")
        return [_fake_tool("ok_tool")]

    fake_client.get_tools = AsyncMock(side_effect=_get_tools)

    async def _fake_list_prompts(client: Any, server_name: str) -> list[Any]:
        return []

    monkeypatch.setattr(
        manager_mod, "MultiServerMCPClient", lambda cfg: fake_client,
    )
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_fake_list_prompts),
    )

    mgr = MCPManager([
        MCPServerConfig(name="broken", command="npx", args=[]),
        MCPServerConfig(name="good", command="npx", args=[]),
    ])
    tools, commands = await mgr.start_all()

    # "good" server's tool survives; "broken" is dropped.
    names = [t.name for t in tools]
    assert "mcp__good__ok_tool" in names
    assert not any(n.startswith("mcp__broken__") for n in names)

    # Journal records the connection failure.
    journal_text = log_path.read_text(encoding="utf-8")
    assert "mcp_connect_failed" in journal_text
    assert "broken" in journal_text
    journal.reset()


@pytest.mark.asyncio
async def test_stop_all_suppresses_teardown_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The library currently has no close() method; stop_all is defensively
    # written to call any close/__aexit__ if the library adds one, and must
    # never propagate exceptions from teardown.
    from aura.core.mcp import manager as manager_mod

    fake_client = MagicMock()
    fake_client.get_tools = AsyncMock(return_value=[])

    async def _fake_list_prompts(client: Any, server_name: str) -> list[Any]:
        return []

    monkeypatch.setattr(
        manager_mod, "MultiServerMCPClient", lambda cfg: fake_client,
    )
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_fake_list_prompts),
    )

    mgr = MCPManager([MCPServerConfig(name="x", command="npx", args=[])])
    await mgr.start_all()
    # Must not raise even if the client does something unexpected on teardown.
    await mgr.stop_all()


@pytest.mark.asyncio
async def test_start_all_skips_disabled_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.core.mcp import manager as manager_mod

    fake_client = MagicMock()
    fake_client.get_tools = AsyncMock(return_value=[_fake_tool("t")])

    async def _fake_list_prompts(client: Any, server_name: str) -> list[Any]:
        return []

    monkeypatch.setattr(
        manager_mod, "MultiServerMCPClient", lambda cfg: fake_client,
    )
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_fake_list_prompts),
    )

    mgr = MCPManager([
        MCPServerConfig(name="off", command="npx", args=[], enabled=False),
    ])
    tools, commands = await mgr.start_all()
    assert tools == []
    assert commands == []
    # Never asked the client about a disabled server.
    fake_client.get_tools.assert_not_awaited()
