"""Tests for aura.core.mcp.manager — MCPManager wraps MultiServerMCPClient."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from aura.config.schema import MCPServerConfig
from aura.core.mcp.manager import MCPManager, MCPServerStatus


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

    async def _fake_list_resources(client: Any, server_name: str) -> list[Any]:
        return []

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

    assert len(tools) == 1
    t = tools[0]
    assert t.name == "mcp__gh__search"
    assert (t.metadata or {}).get("is_destructive") is True
    assert (t.metadata or {}).get("max_result_size_chars") == 30_000
    fake_client.get_tools.assert_awaited_once_with(server_name="gh")
    # Resources catalogue is empty when the server exposes no resources.
    assert mgr.resources_catalogue() == []


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

    async def _fake_list_resources(client: Any, server_name: str) -> list[Any]:
        return []

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

    async def _fake_list_resources(client: Any, server_name: str) -> list[Any]:
        return []

    monkeypatch.setattr(
        manager_mod, "MultiServerMCPClient", lambda cfg: fake_client,
    )
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_fake_list_prompts),
    )
    monkeypatch.setattr(
        MCPManager, "_list_resources", staticmethod(_fake_list_resources),
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

    async def _fake_list_resources(client: Any, server_name: str) -> list[Any]:
        return []

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
        MCPServerConfig(name="off", command="npx", args=[], enabled=False),
    ])
    tools, commands = await mgr.start_all()
    assert tools == []
    assert commands == []
    # Never asked the client about a disabled server.
    fake_client.get_tools.assert_not_awaited()


# ---------------------------------------------------------------------------
# In-REPL control surface — /mcp enable / disable / reconnect / status
# ---------------------------------------------------------------------------


def _patch_manager_internals(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tools_by_server: dict[str, list[StructuredTool]] | None = None,
    errors_by_server: dict[str, Exception] | None = None,
) -> MagicMock:
    """Wire up a fake MultiServerMCPClient that returns per-server data.

    Returns the fake-client MagicMock so individual tests can inspect
    call counts. Both ``_list_prompts`` and ``_list_resources`` are
    stubbed to empty lists — we only exercise tool discovery here.
    """
    tools_by_server = tools_by_server or {}
    errors_by_server = errors_by_server or {}

    fake_client = MagicMock()
    # ``.connections`` is a real dict — ``enable``/``disable`` mutate it.
    fake_client.connections = {}

    async def _get_tools(*, server_name: str) -> list[StructuredTool]:
        if server_name in errors_by_server:
            raise errors_by_server[server_name]
        return list(tools_by_server.get(server_name, []))

    fake_client.get_tools = AsyncMock(side_effect=_get_tools)

    async def _fake_list_prompts(client: Any, server_name: str) -> list[Any]:
        return []

    async def _fake_list_resources(client: Any, server_name: str) -> list[Any]:
        return []

    from aura.core.mcp import manager as manager_mod

    def _make_client(connections: dict[str, Any]) -> MagicMock:
        # Library populates ``.connections`` from the ctor arg; mimic that
        # so ``session(name)`` lookups behave like the real library would.
        fake_client.connections = dict(connections)
        return fake_client

    monkeypatch.setattr(manager_mod, "MultiServerMCPClient", _make_client)
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_fake_list_prompts),
    )
    monkeypatch.setattr(
        MCPManager, "_list_resources", staticmethod(_fake_list_resources),
    )
    return fake_client


def test_status_before_start_all_is_never_started() -> None:
    """Sanity: no connect attempt yet → every known server is never_started."""
    mgr = MCPManager([
        MCPServerConfig(name="a", command="npx", args=[]),
        MCPServerConfig(name="b", command="npx", args=[]),
    ])
    rows = mgr.status()
    assert [r.name for r in rows] == ["a", "b"]
    assert all(r.state == "never_started" for r in rows)
    assert all(r.error_message is None for r in rows)
    assert all(r.tool_count == 0 for r in rows)


def test_status_includes_disabled_by_config() -> None:
    """enabled=False at construction → disabled state in status()."""
    mgr = MCPManager([
        MCPServerConfig(name="on", command="npx", args=[]),
        MCPServerConfig(name="off", command="npx", args=[], enabled=False),
    ])
    rows = {r.name: r for r in mgr.status()}
    assert rows["on"].state == "never_started"
    assert rows["off"].state == "disabled"


def test_status_never_raises_with_no_servers() -> None:
    # A zero-config manager must still return a (possibly empty) list —
    # the /mcp list view renders this as the "no MCP servers" placeholder.
    mgr = MCPManager([])
    assert mgr.status() == []


def test_known_server_names_returns_all_configured() -> None:
    mgr = MCPManager([
        MCPServerConfig(name="alpha", command="npx", args=[]),
        MCPServerConfig(name="beta", command="npx", args=[], enabled=False),
    ])
    # Preserves config order (disabled included).
    assert mgr.known_server_names() == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_status_after_start_all_reflects_connected_and_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_manager_internals(
        monkeypatch,
        tools_by_server={"good": [_fake_tool("t1"), _fake_tool("t2")]},
        errors_by_server={"bad": RuntimeError("boom")},
    )
    mgr = MCPManager([
        MCPServerConfig(name="good", command="npx", args=[]),
        MCPServerConfig(name="bad", command="npx", args=[]),
    ])
    await mgr.start_all()
    rows = {r.name: r for r in mgr.status()}
    assert rows["good"].state == "connected"
    assert rows["good"].tool_count == 2
    assert rows["bad"].state == "error"
    assert rows["bad"].error_message is not None
    assert "boom" in rows["bad"].error_message


@pytest.mark.asyncio
async def test_disable_flips_status_to_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_manager_internals(
        monkeypatch,
        tools_by_server={"srv": [_fake_tool("only")]},
    )
    mgr = MCPManager([MCPServerConfig(name="srv", command="npx", args=[])])
    await mgr.start_all()
    assert next(r for r in mgr.status() if r.name == "srv").state == "connected"

    result = await mgr.disable("srv")
    assert "disabled" in result
    row = next(r for r in mgr.status() if r.name == "srv")
    assert row.state == "disabled"
    assert row.tool_count == 0
    assert row.error_message is None


@pytest.mark.asyncio
async def test_disable_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_manager_internals(monkeypatch)
    mgr = MCPManager([MCPServerConfig(name="srv", command="npx", args=[])])
    # No start_all; immediately disable — state was never_started.
    first = await mgr.disable("srv")
    assert "disabled" in first
    # Second disable must not raise and must state already-disabled.
    second = await mgr.disable("srv")
    assert "already disabled" in second


@pytest.mark.asyncio
async def test_disable_unknown_name_returns_error_text() -> None:
    # No start_all, no subprocess path touched — manager built from empty
    # configs, unknown-name disable returns a textual error.
    mgr = MCPManager([MCPServerConfig(name="known", command="npx", args=[])])
    text = await mgr.disable("nonexistent")
    assert "no MCP server named" in text
    assert "nonexistent" in text
    assert "known" in text  # known-names hint


@pytest.mark.asyncio
async def test_enable_unknown_name_returns_error_text() -> None:
    mgr = MCPManager([MCPServerConfig(name="known", command="npx", args=[])])
    text = await mgr.enable("nonexistent")
    assert "no MCP server named" in text
    assert "nonexistent" in text


@pytest.mark.asyncio
async def test_reconnect_unknown_name_returns_error_text() -> None:
    mgr = MCPManager([MCPServerConfig(name="known", command="npx", args=[])])
    text = await mgr.reconnect("nonexistent")
    assert "no MCP server named" in text


@pytest.mark.asyncio
async def test_enable_after_disable_reconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_manager_internals(
        monkeypatch,
        tools_by_server={"srv": [_fake_tool("x")]},
    )
    mgr = MCPManager([MCPServerConfig(name="srv", command="npx", args=[])])
    await mgr.start_all()
    await mgr.disable("srv")
    row = next(r for r in mgr.status() if r.name == "srv")
    assert row.state == "disabled"

    text = await mgr.enable("srv")
    assert "connected" in text
    row = next(r for r in mgr.status() if r.name == "srv")
    assert row.state == "connected"
    assert row.tool_count == 1


@pytest.mark.asyncio
async def test_enable_already_connected_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_manager_internals(
        monkeypatch,
        tools_by_server={"srv": [_fake_tool("x")]},
    )
    mgr = MCPManager([MCPServerConfig(name="srv", command="npx", args=[])])
    await mgr.start_all()
    text = await mgr.enable("srv")
    assert "already connected" in text


@pytest.mark.asyncio
async def test_reconnect_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running reconnect twice in a row must not crash / must succeed both."""
    _patch_manager_internals(
        monkeypatch,
        tools_by_server={"srv": [_fake_tool("x")]},
    )
    mgr = MCPManager([MCPServerConfig(name="srv", command="npx", args=[])])
    await mgr.start_all()
    first = await mgr.reconnect("srv")
    second = await mgr.reconnect("srv")
    assert "reconnected" in first
    assert "reconnected" in second
    row = next(r for r in mgr.status() if r.name == "srv")
    assert row.state == "connected"


@pytest.mark.asyncio
async def test_reconnect_surfaces_error_on_failed_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconnect that fails must return an error string AND set error state."""
    _patch_manager_internals(
        monkeypatch,
        errors_by_server={"srv": RuntimeError("broken pipe")},
    )
    mgr = MCPManager([MCPServerConfig(name="srv", command="npx", args=[])])
    text = await mgr.reconnect("srv")
    assert "failed to reconnect" in text
    assert "broken pipe" in text
    row = next(r for r in mgr.status() if r.name == "srv")
    assert row.state == "error"
    assert row.error_message is not None


def test_mcp_server_status_is_frozen_dataclass() -> None:
    """``MCPServerStatus`` should be immutable so callers can't mutate state."""
    import dataclasses as _dc

    s = MCPServerStatus(
        name="x", transport="stdio", state="connected",
        error_message=None, tool_count=0, resource_count=0, prompt_count=0,
    )
    with pytest.raises(_dc.FrozenInstanceError):
        s.tool_count = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# F-06-002 — env-var expansion in ``_build_one_connection``.
# Belt-and-suspenders defence for configs constructed in code that bypass
# ``mcp_store.load()`` (the loader's own expansion is covered in
# ``tests/test_mcp_env_expand.py``).
# ---------------------------------------------------------------------------


def test_build_one_connection_expands_stdio_command_args_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_BIN", "/usr/bin/python")
    monkeypatch.setenv("MY_TOK", "deadbeef")
    cfg = MCPServerConfig(
        name="s1",
        transport="stdio",
        command="${MY_BIN}",
        args=["--flag", "${MY_BIN}"],
        env={"TOKEN": "${MY_TOK}"},
    )
    out = MCPManager._build_one_connection(cfg)
    entry = out["s1"]
    assert entry["command"] == "/usr/bin/python"
    assert entry["args"] == ["--flag", "/usr/bin/python"]
    assert entry["env"] == {"TOKEN": "deadbeef"}


def test_build_one_connection_missing_var_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_BIN", raising=False)
    cfg = MCPServerConfig(
        name="s1",
        transport="stdio",
        command="${MISSING_BIN}",
    )
    with pytest.raises(RuntimeError) as exc_info:
        MCPManager._build_one_connection(cfg)
    msg = str(exc_info.value)
    assert "MISSING_BIN" in msg
    assert "s1" in msg


def test_build_one_connection_default_used_when_var_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``${VAR:-default}`` does NOT raise — default substitutes silently."""
    monkeypatch.delenv("OPT_BIN", raising=False)
    cfg = MCPServerConfig(
        name="s1",
        transport="stdio",
        command="${OPT_BIN:-/bin/true}",
    )
    out = MCPManager._build_one_connection(cfg)
    assert out["s1"]["command"] == "/bin/true"


def test_build_one_connection_expands_url_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_HOST", "https://api.example.com")
    monkeypatch.setenv("BEARER", "token-xyz")
    cfg = MCPServerConfig(
        name="s1",
        transport="sse",
        url="${API_HOST}/mcp",
        headers={"Authorization": "Bearer ${BEARER}"},
    )
    out = MCPManager._build_one_connection(cfg)
    entry = out["s1"]
    assert entry["url"] == "https://api.example.com/mcp"
    assert entry["headers"] == {"Authorization": "Bearer token-xyz"}


def test_build_one_connection_missing_var_in_headers_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ABSENT_TOKEN", raising=False)
    cfg = MCPServerConfig(
        name="s1",
        transport="sse",
        url="https://api.example.com/mcp",
        headers={"Authorization": "Bearer ${ABSENT_TOKEN}"},
    )
    with pytest.raises(RuntimeError) as exc_info:
        MCPManager._build_one_connection(cfg)
    msg = str(exc_info.value)
    assert "ABSENT_TOKEN" in msg
    assert "s1" in msg


# ---------------------------------------------------------------------------
# F-06-004 — list-changed message handler attached to each connection
# ---------------------------------------------------------------------------


def test_build_one_connection_attaches_message_handler_stdio() -> None:
    """Every stdio connection carries a session_kwargs.message_handler."""
    cfg = MCPServerConfig(name="alpha", transport="stdio", command="echo")
    out = MCPManager._build_one_connection(cfg)
    entry = out["alpha"]
    assert "session_kwargs" in entry
    assert callable(entry["session_kwargs"]["message_handler"])


def test_build_one_connection_attaches_message_handler_remote() -> None:
    """SSE connections carry the same session_kwargs.message_handler hook."""
    cfg = MCPServerConfig(
        name="beta", transport="sse", url="https://x.example/mcp",
    )
    out = MCPManager._build_one_connection(cfg)
    entry = out["beta"]
    assert "session_kwargs" in entry
    assert callable(entry["session_kwargs"]["message_handler"])


@pytest.mark.asyncio
async def test_list_changed_handler_journals_only_relevant_methods(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The handler journals tools/prompts/resources list_changed; others ignored."""
    from aura.core.mcp.manager import _make_list_changed_logger
    from aura.core.persistence import journal as journal_module
    log_path = tmp_path / "audit.jsonl"
    journal_module.configure(log_path)
    try:
        handler = _make_list_changed_logger("server-x")

        class _Notification:
            def __init__(self, method: str) -> None:
                self.root = type("Root", (), {"method": method})()

        # Three relevant notifications + one irrelevant one.
        await handler(_Notification("notifications/tools/list_changed"))
        await handler(_Notification("notifications/prompts/list_changed"))
        await handler(_Notification("notifications/resources/list_changed"))
        await handler(_Notification("notifications/logging/setLevel"))

        import json
        events = [
            json.loads(line) for line in log_path.read_text().splitlines() if line
        ]
        list_changed = [e for e in events if e["event"] == "mcp_list_changed"]
        assert len(list_changed) == 3
        assert all(e["server"] == "server-x" for e in list_changed)
        methods = {e["method"] for e in list_changed}
        assert methods == {
            "notifications/tools/list_changed",
            "notifications/prompts/list_changed",
            "notifications/resources/list_changed",
        }
    finally:
        journal_module.reset()
