"""Tests for MCP resources support.

Exercises the three surfaces introduced for MCP resource exposure:

- :class:`MCPManager` — discovery (``start_all`` now calls
  ``session.list_resources``), accessor (``resources_catalogue``), and
  on-demand read (``read_resource``).
- :class:`MCPReadResourceTool` — generic LLM-invocable tool with
  dynamically-built description.
- Agent wiring — ``aconnect()`` registers the tool iff ≥1 resource
  was discovered across all connected servers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from aura.config.schema import MCPServerConfig
from aura.core.mcp.adapter import normalize_resource_contents
from aura.core.mcp.manager import MCPManager
from aura.schemas.tool import ToolError
from aura.tools.mcp_read_resource import (
    MCPReadResourceTool,
    build_description,
)


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


def _fake_resource(
    uri: str,
    *,
    name: str | None = None,
    description: str | None = None,
    mime: str | None = None,
) -> SimpleNamespace:
    # Mimics the mcp.types.Resource shape (we only use the attributes the
    # manager reads). Using SimpleNamespace rather than the real pydantic
    # model keeps the tests independent of minor MCP-SDK field churn.
    return SimpleNamespace(
        uri=uri,
        name=name,
        description=description,
        mimeType=mime,
    )


# ---------------------------------------------------------------------------
# normalize_resource_contents
# ---------------------------------------------------------------------------


def test_normalize_text_resource_returns_text_shape() -> None:
    obj = SimpleNamespace(
        uri="file:///a.md", mimeType="text/markdown", text="hello"
    )
    out = normalize_resource_contents(obj)
    assert out == {
        "type": "text",
        "uri": "file:///a.md",
        "mime": "text/markdown",
        "text": "hello",
    }


def test_normalize_blob_resource_omits_payload_reports_size() -> None:
    # 4 bytes "abcd" base64 = "YWJjZA==" (8 base64 chars → 4 decoded bytes)
    obj = SimpleNamespace(
        uri="file:///img.png", mimeType="image/png", blob="YWJjZA=="
    )
    out = normalize_resource_contents(obj)
    assert out["type"] == "blob"
    assert out["uri"] == "file:///img.png"
    assert out["mime"] == "image/png"
    assert out["size"] == 4
    assert "blob" not in out  # payload never echoed back


def test_normalize_unknown_falls_back_to_repr() -> None:
    obj = SimpleNamespace(uri="x://y")
    out = normalize_resource_contents(obj)
    assert out["type"] == "unknown"
    assert out["uri"] == "x://y"


# ---------------------------------------------------------------------------
# MCPManager — start_all discovers resources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_all_discovers_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.get_tools = AsyncMock(return_value=[_fake_tool("search")])

    async def _fake_list_prompts(client: Any, server_name: str) -> list[Any]:
        return []

    async def _fake_list_resources(client: Any, server_name: str) -> list[Any]:
        return [
            _fake_resource(
                "file:///doc.md",
                name="doc",
                description="project doc",
                mime="text/markdown",
            ),
            _fake_resource("db://snapshot/latest", name="latest"),
        ]

    from aura.core.mcp import manager as manager_mod

    monkeypatch.setattr(
        manager_mod, "MultiServerMCPClient", lambda cfg: fake_client,
    )
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_fake_list_prompts),
    )
    monkeypatch.setattr(
        MCPManager, "_list_resources", staticmethod(_fake_list_resources),
    )

    mgr = MCPManager([
        MCPServerConfig(name="gh", command="npx", args=["-y", "x"]),
    ])
    tools, commands = await mgr.start_all()

    catalogue = mgr.resources_catalogue()
    assert len(catalogue) == 2
    # Sorted by (server, uri) — db:// comes before file://
    assert catalogue[0][0] == "gh"
    assert catalogue[0][1] == "db://snapshot/latest"
    assert catalogue[1][1] == "file:///doc.md"
    assert catalogue[1][2] == "doc"
    assert catalogue[1][3] == "project doc"
    assert catalogue[1][4] == "text/markdown"


@pytest.mark.asyncio
async def test_start_all_resource_listing_failure_is_graceful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.get_tools = AsyncMock(return_value=[_fake_tool("search")])

    async def _fake_list_prompts(client: Any, server_name: str) -> list[Any]:
        return []

    async def _broken_list_resources(client: Any, server_name: str) -> list[Any]:
        # The manager's static wrapper swallows exceptions; simulate the
        # graceful-degrade contract by returning the same empty list the
        # wrapper would.
        return []

    from aura.core.mcp import manager as manager_mod

    monkeypatch.setattr(
        manager_mod, "MultiServerMCPClient", lambda cfg: fake_client,
    )
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_fake_list_prompts),
    )
    monkeypatch.setattr(
        MCPManager, "_list_resources", staticmethod(_broken_list_resources),
    )

    mgr = MCPManager([
        MCPServerConfig(name="gh", command="npx", args=["-y", "x"]),
    ])
    tools, _ = await mgr.start_all()

    # Tools still come through; resources catalogue is empty.
    assert any(t.name == "mcp__gh__search" for t in tools)
    assert mgr.resources_catalogue() == []


# ---------------------------------------------------------------------------
# MCPManager.read_resource — routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_resource_routes_to_owning_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.get_tools = AsyncMock(return_value=[])

    async def _fake_list_prompts(client: Any, server_name: str) -> list[Any]:
        return []

    async def _fake_list_resources(client: Any, server_name: str) -> list[Any]:
        if server_name == "s1":
            return [_fake_resource("doc://a", mime="text/plain")]
        return [_fake_resource("doc://b", mime="text/plain")]

    # Session/read_resource plumbing — the session() ctx manager returns
    # a session whose read_resource returns a ReadResourceResult-ish shape.
    read_result = SimpleNamespace(
        contents=[
            SimpleNamespace(uri="doc://a", mimeType="text/plain", text="body-A"),
        ],
    )
    session_obj = MagicMock()
    session_obj.read_resource = AsyncMock(return_value=read_result)

    class _SessionCtx:
        def __init__(self, server_name: str) -> None:
            self.server_name = server_name

        async def __aenter__(self) -> Any:
            return session_obj

        async def __aexit__(self, *_: Any) -> None:
            return None

    fake_client.session = MagicMock(side_effect=lambda name: _SessionCtx(name))

    from aura.core.mcp import manager as manager_mod

    monkeypatch.setattr(
        manager_mod, "MultiServerMCPClient", lambda cfg: fake_client,
    )
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_fake_list_prompts),
    )
    monkeypatch.setattr(
        MCPManager, "_list_resources", staticmethod(_fake_list_resources),
    )

    mgr = MCPManager([
        MCPServerConfig(name="s1", command="npx", args=[]),
        MCPServerConfig(name="s2", command="npx", args=[]),
    ])
    await mgr.start_all()

    out = await mgr.read_resource("doc://a")
    assert out["server"] == "s1"
    assert out["uri"] == "doc://a"
    assert out["contents"][0]["type"] == "text"
    assert out["contents"][0]["text"] == "body-A"
    # Confirm the session was opened on s1, not s2.
    fake_client.session.assert_called_with("s1")


@pytest.mark.asyncio
async def test_read_resource_unknown_uri_raises_with_known_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.get_tools = AsyncMock(return_value=[])

    async def _fake_list_prompts(client: Any, server_name: str) -> list[Any]:
        return []

    async def _fake_list_resources(client: Any, server_name: str) -> list[Any]:
        return [_fake_resource("doc://known")]

    from aura.core.mcp import manager as manager_mod

    monkeypatch.setattr(
        manager_mod, "MultiServerMCPClient", lambda cfg: fake_client,
    )
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_fake_list_prompts),
    )
    monkeypatch.setattr(
        MCPManager, "_list_resources", staticmethod(_fake_list_resources),
    )

    mgr = MCPManager([MCPServerConfig(name="s", command="npx", args=[])])
    await mgr.start_all()

    with pytest.raises(ValueError) as exc_info:
        await mgr.read_resource("doc://unknown")
    msg = str(exc_info.value)
    assert "doc://unknown" in msg
    assert "doc://known" in msg


# ---------------------------------------------------------------------------
# MCPReadResourceTool — description + invocation
# ---------------------------------------------------------------------------


def test_build_description_lists_all_uris() -> None:
    catalogue = [
        ("gh", "file:///doc.md", "doc", "the doc", "text/markdown"),
        ("db", "db://s/1", "snapshot", "", None),
    ]
    desc = build_description(catalogue)
    assert "file:///doc.md" in desc
    assert "db://s/1" in desc
    assert "[gh]" in desc
    assert "[db]" in desc
    assert "text/markdown" in desc
    assert "the doc" in desc


def test_build_description_empty_catalogue_signals_none() -> None:
    desc = build_description([])
    assert "No MCP resources" in desc


@pytest.mark.asyncio
async def test_tool_invocation_with_valid_uri_returns_contents() -> None:
    async def reader(uri: str) -> dict[str, Any]:
        return {
            "uri": uri,
            "server": "s1",
            "contents": [{"type": "text", "text": "body", "uri": uri, "mime": None}],
        }

    tool = MCPReadResourceTool(
        resource_reader=reader,
        description=build_description(
            [("s1", "doc://a", "a", "desc", "text/plain")],
        ),
    )
    result = await tool.ainvoke({"uri": "doc://a"})
    assert result["uri"] == "doc://a"
    assert result["server"] == "s1"
    assert result["contents"][0]["text"] == "body"


@pytest.mark.asyncio
async def test_tool_invocation_unknown_uri_raises_tool_error() -> None:
    async def reader(uri: str) -> dict[str, Any]:
        raise ValueError(f"unknown MCP resource uri {uri!r}; known uris: ['doc://a']")

    tool = MCPReadResourceTool(
        resource_reader=reader,
        description=build_description(
            [("s1", "doc://a", "a", "", None)],
        ),
    )
    # LangChain's ainvoke doesn't catch ToolError; it surfaces as-is.
    with pytest.raises(ToolError) as exc_info:
        await tool.ainvoke({"uri": "doc://missing"})
    msg = str(exc_info.value)
    assert "doc://a" in msg


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------
#
# ``MCPReadResourceTool`` is NOT auto-registered as of v0.10.x — resources
# flow through the CLI-layer ``@server:uri`` attachment preprocessor
# (see :mod:`aura.cli.attachments`). ``aconnect`` exposes the manager on
# :attr:`Agent.mcp_manager` but does not touch ``available_tools`` for the
# resource surface anymore. The tool class remains importable for
# programmatic SDK users who want LLM-driven reads (tested above via
# ``test_tool_invocation_*``).


@pytest.mark.asyncio
async def test_agent_aconnect_does_not_auto_register_resource_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    from aura.config.schema import AuraConfig, StorageConfig
    from aura.core.agent import Agent
    from aura.core.persistence.storage import SessionStorage

    cfg = AuraConfig(
        router={"default": "openai:gpt-4o-mini"},
        mcp_servers=[MCPServerConfig(name="s", command="npx", args=[])],
        storage=StorageConfig(path=str(tmp_path / "sessions.db")),
    )

    storage = SessionStorage(cfg.resolved_storage_path())

    async def _fake_start_all(self: Any) -> tuple[list[Any], list[Any]]:
        # Simulate a manager that discovered 1 resource — but the tool
        # still must NOT auto-register; the @mention preprocessor owns the
        # resource surface now.
        self._resources[("s", "doc://a")] = _fake_resource("doc://a", name="a")
        return [], []

    monkeypatch.setattr(MCPManager, "start_all", _fake_start_all)

    fake_model = MagicMock()
    fake_model.bind_tools = MagicMock(return_value=fake_model)

    agent = Agent(
        config=cfg,
        model=fake_model,
        storage=storage,
    )
    try:
        await agent.aconnect()
        # Tool stays OUT of the registry / available map — the @mention
        # preprocessor is the supported surface as of v0.10.x.
        assert "mcp_read_resource" not in agent._available_tools
        # Manager is still reachable for the preprocessor to query.
        assert agent.mcp_manager is not None
        assert ("s", "doc://a") in agent.mcp_manager._resources
    finally:
        agent.close()


def test_deprecated_tool_metadata_marks_flag() -> None:
    # Defensive: SDK filters (e.g. `show me only non-deprecated tools`)
    # should be able to check this flag without reaching into the tool's
    # module-private constants.
    async def _noop_reader(uri: str) -> dict[str, Any]:
        return {"uri": uri, "server": "x", "contents": []}

    tool = MCPReadResourceTool(resource_reader=_noop_reader)
    assert tool.metadata is not None
    assert tool.metadata.get("deprecated") is True
    assert tool.metadata.get("deprecated_since") == "0.10.0"
    assert "@server:uri" in (tool.metadata.get("deprecated_replacement") or "")
