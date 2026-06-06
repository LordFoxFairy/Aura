"""Tests for aura.infrastructure.mcp.manager — MCPManager wraps MultiServerMCPClient."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import Awaitable
from typing import Any, get_type_hints
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from aura.config.schema import AuraConfigError, MCPServerConfig
from aura.domain.tool_meta_access import meta_dict
from aura.infrastructure.mcp import manager as _manager_mod
from aura.infrastructure.mcp.manager import MCPManager
from aura.infrastructure.mcp.types import MCPServerStatus
from aura.infrastructure.persistence import journal as journal_module


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

    from aura.infrastructure.mcp import manager as manager_mod

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
    assert meta_dict(t).get("is_destructive") is True
    assert meta_dict(t).get("max_result_size_chars") == 30_000
    fake_client.get_tools.assert_awaited_once_with(server_name="gh")
    # Resources catalogue is empty when the server exposes no resources.
    assert mgr.resources_catalogue() == []


@pytest.mark.asyncio
async def test_start_all_broken_server_graceful_degrade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    from aura.core import journal
    from aura.infrastructure.mcp import manager as manager_mod

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
    from aura.infrastructure.mcp import manager as manager_mod

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
    from aura.infrastructure.mcp import manager as manager_mod

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

    from aura.infrastructure.mcp import manager as manager_mod

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
        s.__setattr__("tool_count", 5)


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
    from aura.infrastructure.mcp.manager import _make_list_changed_logger
    from aura.infrastructure.persistence import journal as journal_module
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


def test_manager_start_all_type_hints_resolve_without_type_checking_imports() -> None:
    """Public annotations stay resolvable after dropping TYPE_CHECKING helpers."""
    hints = get_type_hints(MCPManager.start_all)
    tools_t, commands_t = hints["return"].__args__
    assert tools_t == list[BaseTool]
    assert commands_t.__origin__ is list


# --------------------------------------------------------------------------- #
# Boundary tests appended to push aura.infrastructure.mcp.manager toward ~95%. #
# All seams are mocked: no real MCP server / subprocess / network / sleep.     #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class _FakePrompt:
    """Minimal MCP prompt object matching the manager's _PromptLike protocol."""

    name: object
    description: object
    arguments: object


@dataclasses.dataclass(frozen=True)
class _FakeResource:
    """Minimal MCP resource object matching the _ResourceLike protocol."""

    uri: object
    name: object
    description: object
    mimeType: object  # noqa: N815  # mirrors the MCP wire field name verbatim


@dataclasses.dataclass(frozen=True)
class _CodedError(Exception):
    """Exception exposing a JSON-RPC ``code`` like a real MCP error envelope."""

    code: int

    def __str__(self) -> str:
        return f"coded-error({self.code})"


class _FakeContents:
    """Resource-contents row carrying a text payload for normalization."""

    def __init__(self, uri: str, text: str) -> None:
        self.uri = uri
        self.mimeType = "text/plain"
        self.text = text


class _FakeSession:
    """Async-context session returning canned list/read responses."""

    def __init__(
        self,
        *,
        prompts: list[Any] | None = None,
        resources: list[Any] | None = None,
        read_contents: list[Any] | None = None,
    ) -> None:
        self._prompts = prompts or []
        self._resources = resources or []
        self._read_contents = read_contents or []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def list_prompts(self) -> Any:
        return type("R", (), {"prompts": self._prompts})()

    async def list_resources(self) -> Any:
        return type("R", (), {"resources": self._resources})()

    async def read_resource(self, uri: Any) -> Any:
        return type("R", (), {"contents": self._read_contents})()


def _cfg(name: str, **kw: Any) -> MCPServerConfig:
    """Build an enabled stdio MCPServerConfig with sensible test defaults."""
    params: dict[str, Any] = {"command": "npx", "args": []}
    params.update(kw)
    return MCPServerConfig(name=name, **params)


# ----------------------------- pure helpers -------------------------------- #


def test_resolve_op_timeout_rejects_non_positive_explicit() -> None:
    """A non-positive explicit timeout is a config error, not a silent clamp."""
    with pytest.raises(ValueError, match="must be positive"):
        MCPManager([], op_timeout_sec=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        MCPManager([], op_timeout_sec=-3.5)


def test_resolve_op_timeout_explicit_positive_wins() -> None:
    """Explicit kwarg overrides env + default so callers stay deterministic."""
    mgr = MCPManager([], op_timeout_sec=12.5)
    assert mgr.op_timeout_sec == 12.5


@pytest.mark.parametrize(
    ("env_val", "expected"),
    [
        ("45.5", 45.5),
        ("not-a-number", 30.0),
        ("0", 30.0),
        ("-1", 30.0),
        ("", 30.0),
    ],
)
def test_resolve_op_timeout_env_matrix(
    monkeypatch: pytest.MonkeyPatch, env_val: str, expected: float
) -> None:
    """Env override: valid>0 used; garbage / zero / negative fall back to default."""
    monkeypatch.setenv("AURA_MCP_TIMEOUT_SEC", env_val)
    mgr = MCPManager([])
    assert mgr.op_timeout_sec == expected


def test_resolve_op_timeout_unset_env_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no kwarg and no env var, the manager pins the 30s default."""
    monkeypatch.delenv("AURA_MCP_TIMEOUT_SEC", raising=False)
    mgr = MCPManager([])
    assert mgr.op_timeout_sec == 30.0


@pytest.mark.parametrize(
    "text",
    ["oauth required", "Unauthorized", "got 401 back", "403 forbidden"],
)
def test_is_needs_auth_error_text_hints(text: str) -> None:
    """Auth failures are detected from message hints, not just the -32001 code."""
    assert _manager_mod._is_needs_auth_error(RuntimeError(text)) is True


def test_is_needs_auth_error_code_takes_priority() -> None:
    """A -32001 JSON-RPC code marks needs-auth even with an opaque message."""
    assert _manager_mod._is_needs_auth_error(_CodedError(code=-32001)) is True


def test_is_needs_auth_error_plain_failure_is_not_auth() -> None:
    """A generic failure must NOT be misclassified as an auth problem."""
    assert _manager_mod._is_needs_auth_error(RuntimeError("disk full")) is False


def test_unsupported_transport_raises_aura_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport the installed adapter lacks must hard-fail at construction."""
    monkeypatch.setattr(_manager_mod, "_supported_transports", lambda: {"stdio"})
    with pytest.raises(AuraConfigError):
        MCPManager([_cfg("s", transport="sse", url="https://x/mcp", command=None)])


# ------------------------- list_prompts / list_resources ------------------- #


@pytest.mark.asyncio
async def test_list_prompts_returns_session_prompts() -> None:
    """The real _list_prompts body forwards the session's prompt list."""
    prompt = _FakePrompt(name="p", description="d", arguments=[])
    client = MagicMock()
    client.session = lambda name: _FakeSession(prompts=[prompt])
    out = await MCPManager._list_prompts(client, "srv")
    assert out == [prompt]


@pytest.mark.asyncio
async def test_list_prompts_swallows_session_failure() -> None:
    """A server lacking the prompts capability yields [] without blocking tools."""

    def _boom(name: str) -> _FakeSession:
        raise RuntimeError("no prompts capability")

    client = MagicMock()
    client.session = _boom
    assert await MCPManager._list_prompts(client, "srv") == []


@pytest.mark.asyncio
async def test_list_resources_returns_session_resources() -> None:
    """The real _list_resources body forwards the session's resource list."""
    res = _FakeResource(uri="file:///a", name="a", description="", mimeType="text/plain")
    client = MagicMock()
    client.session = lambda name: _FakeSession(resources=[res])
    out = await MCPManager._list_resources(client, "srv")
    assert out == [res]


@pytest.mark.asyncio
async def test_list_resources_swallows_session_failure() -> None:
    """A server lacking the resources capability yields [] without blocking tools."""

    def _boom(name: str) -> _FakeSession:
        raise RuntimeError("no resources capability")

    client = MagicMock()
    client.session = _boom
    assert await MCPManager._list_resources(client, "srv") == []


# ------------------------------- timeout path ------------------------------ #


@pytest.mark.asyncio
async def test_run_with_timeout_wraps_timeout_as_runtime_error() -> None:
    """A stalled MCP op surfaces as a RuntimeError naming op + server, not a hang."""
    mgr = MCPManager([], op_timeout_sec=0.01)

    async def _never() -> None:
        await asyncio.Event().wait()

    with pytest.raises(RuntimeError, match="timed out"):
        await mgr._run_with_timeout(_never(), op_name="get_tools", server="srv")


# ------------------------- prompt-command discovery ------------------------ #


@pytest.mark.asyncio
async def test_connect_one_builds_commands_from_valid_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid prompts become commands; malformed prompt entries are filtered out."""
    _patch_manager_internals(monkeypatch, tools_by_server={"srv": [_fake_tool("t")]})

    good = _FakePrompt(name="hello", description="say hi", arguments=[])
    no_desc = _FakePrompt(name="bare", description=None, arguments="not-a-list")
    bad_name = _FakePrompt(name="", description="d", arguments=[])
    not_prompt = object()

    async def _prompts(client: Any, server_name: str) -> list[Any]:
        return [good, no_desc, bad_name, not_prompt]

    monkeypatch.setattr(MCPManager, "_list_prompts", staticmethod(_prompts))

    mgr = MCPManager([_cfg("srv")])
    _tools, commands = await mgr.start_all()
    assert len(commands) == 2  # good + no_desc survive; empty-name + non-prompt drop
    row = next(r for r in mgr.status() if r.name == "srv")
    assert row.prompt_count == 2


@pytest.mark.asyncio
async def test_connect_one_list_prompts_timeout_degrades(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """A prompts timeout (server up) journals + yields zero commands, not a crash."""
    _patch_manager_internals(monkeypatch, tools_by_server={"srv": [_fake_tool("t")]})

    async def _slow_prompts(client: Any, server_name: str) -> list[Any]:
        raise RuntimeError("unreachable")  # overridden below

    monkeypatch.setattr(MCPManager, "_list_prompts", staticmethod(_slow_prompts))
    log_path = tmp_path / "j.jsonl"
    journal_module.configure(log_path)
    try:
        mgr = MCPManager([_cfg("srv")])

        async def _raise_timeout(coro: Any, *, op_name: str, server: str) -> Any:
            with contextlib.suppress(Exception):
                coro.close()
            if op_name == "list_prompts":
                raise RuntimeError("timed out")
            if op_name == "list_resources":
                return []
            return [_fake_tool("t")]

        monkeypatch.setattr(mgr, "_run_with_timeout", _raise_timeout)
        _tools, commands = await mgr.start_all()
        assert commands == []
        assert "mcp_list_prompts_timeout" in log_path.read_text(encoding="utf-8")
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_connect_one_list_resources_timeout_degrades(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """A resources timeout journals + leaves the catalogue empty, server still up."""
    _patch_manager_internals(monkeypatch, tools_by_server={"srv": [_fake_tool("t")]})
    log_path = tmp_path / "j.jsonl"
    journal_module.configure(log_path)
    try:
        mgr = MCPManager([_cfg("srv")])

        async def _raise_timeout(coro: Any, *, op_name: str, server: str) -> Any:
            with contextlib.suppress(Exception):
                coro.close()
            if op_name == "list_resources":
                raise RuntimeError("timed out")
            if op_name == "list_prompts":
                return []
            return [_fake_tool("t")]

        monkeypatch.setattr(mgr, "_run_with_timeout", _raise_timeout)
        await mgr.start_all()
        assert mgr.resources_catalogue() == []
        assert "mcp_list_resources_timeout" in log_path.read_text(encoding="utf-8")
    finally:
        journal_module.reset()


# ----------------------------- needs-auth path ----------------------------- #


@pytest.mark.asyncio
async def test_connect_one_auth_failure_skips_reconnect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """A 401 sets needs_auth and schedules NO reconnect (it won't self-heal)."""
    _patch_manager_internals(
        monkeypatch,
        errors_by_server={"srv": RuntimeError("401 Unauthorized")},
    )
    log_path = tmp_path / "j.jsonl"
    journal_module.configure(log_path)
    try:
        mgr = MCPManager(
            [_cfg("srv", transport="sse", url="https://x/mcp", command=None)]
        )
        await mgr.start_all()
        row = next(r for r in mgr.status() if r.name == "srv")
        assert row.state == "needs_auth"
        assert row.error_message is not None and "401" in row.error_message
        assert mgr._reconnect_tasks == {}  # auth error must not arm a retry loop
        assert "mcp_connect_needs_auth" in log_path.read_text(encoding="utf-8")
    finally:
        journal_module.reset()


# --------------------------- resource catalogue ---------------------------- #


@pytest.mark.asyncio
async def test_resources_catalogue_sorted_and_fallback_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catalogue is (server,uri)-sorted; missing name falls back to URI tail."""
    res_named = _FakeResource(
        uri="file:///b.txt", name="Beta", description="d", mimeType="text/plain"
    )
    res_unnamed = _FakeResource(
        uri="file:///dir/a.txt", name="", description="", mimeType=None
    )
    bad = _FakeResource(uri=None, name="x", description="", mimeType="text/plain")

    _patch_manager_internals(monkeypatch, tools_by_server={"srv": [_fake_tool("t")]})

    async def _resources(client: Any, server_name: str) -> list[Any]:
        return [res_named, res_unnamed, bad]

    monkeypatch.setattr(MCPManager, "_list_resources", staticmethod(_resources))

    mgr = MCPManager([_cfg("srv")])
    await mgr.start_all()
    cat = mgr.resources_catalogue()
    assert [c[1] for c in cat] == ["file:///b.txt", "file:///dir/a.txt"]
    # Unnamed resource derives its display name from the URI tail.
    assert cat[1][2] == "a.txt"
    assert cat[1][4] is None  # non-str mimeType normalizes to None
    assert cat[0][2] == "Beta"


@pytest.mark.asyncio
async def test_read_resource_before_start_all_raises() -> None:
    """Reading before start_all is a programming error, surfaced as ValueError."""
    mgr = MCPManager([_cfg("srv")])
    with pytest.raises(ValueError, match="not started"):
        await mgr.read_resource("file:///a")


@pytest.mark.asyncio
async def test_read_resource_unknown_uri_lists_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown URI raises ValueError that enumerates the known URIs."""
    res = _FakeResource(
        uri="file:///known", name="k", description="", mimeType="text/plain"
    )
    _patch_manager_internals(monkeypatch, tools_by_server={"srv": [_fake_tool("t")]})

    async def _resources(client: Any, server_name: str) -> list[Any]:
        return [res]

    monkeypatch.setattr(MCPManager, "_list_resources", staticmethod(_resources))
    mgr = MCPManager([_cfg("srv")])
    await mgr.start_all()
    with pytest.raises(ValueError, match="unknown MCP resource") as exc:
        await mgr.read_resource("file:///missing")
    assert "file:///known" in str(exc.value)


@pytest.mark.asyncio
async def test_read_resource_returns_normalized_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A known URI reads + normalizes contents, tagged with its owning server."""
    res = _FakeResource(
        uri="file:///doc", name="doc", description="", mimeType="text/plain"
    )
    fake_client = _patch_manager_internals(
        monkeypatch, tools_by_server={"srv": [_fake_tool("t")]}
    )

    async def _resources(client: Any, server_name: str) -> list[Any]:
        return [res]

    monkeypatch.setattr(MCPManager, "_list_resources", staticmethod(_resources))
    fake_client.session = lambda name: _FakeSession(
        read_contents=[_FakeContents("file:///doc", "hello body")]
    )
    mgr = MCPManager([_cfg("srv")])
    await mgr.start_all()
    out = await mgr.read_resource("file:///doc")
    assert out["server"] == "srv"
    assert out["uri"] == "file:///doc"
    assert out["contents"][0]["text"] == "hello body"


# ----------------------------- reconnect loop ------------------------------ #


@pytest.mark.asyncio
async def test_reconnect_loop_succeeds_on_first_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """A remote drop arms a reconnect that succeeds once the server returns."""
    attempts: dict[str, int] = {"n": 0}

    async def _get_tools(*, server_name: str) -> list[StructuredTool]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("connection reset")
        return [_fake_tool("t")]

    fake_client = MagicMock()
    fake_client.connections = {}
    fake_client.get_tools = AsyncMock(side_effect=_get_tools)

    async def _empty(client: Any, server_name: str) -> list[Any]:
        return []

    def _make_client(connections: dict[str, Any]) -> MagicMock:
        fake_client.connections = dict(connections)
        return fake_client

    monkeypatch.setattr(_manager_mod, "MultiServerMCPClient", _make_client)
    monkeypatch.setattr(MCPManager, "_list_prompts", staticmethod(_empty))
    monkeypatch.setattr(MCPManager, "_list_resources", staticmethod(_empty))

    async def _no_sleep(_sec: float) -> None:
        return None

    monkeypatch.setattr("aura.infrastructure.mcp.manager.asyncio.sleep", _no_sleep)

    log_path = tmp_path / "j.jsonl"
    journal_module.configure(log_path)
    try:
        mgr = MCPManager(
            [_cfg("srv", transport="sse", url="https://x/mcp", command=None)]
        )
        await mgr.start_all()
        task = mgr._reconnect_tasks.get("srv")
        assert task is not None
        await task  # drive the loop to completion deterministically
        row = next(r for r in mgr.status() if r.name == "srv")
        assert row.state == "connected"
        text = log_path.read_text(encoding="utf-8")
        assert "mcp_reconnect_attempt" in text
        assert "mcp_reconnect_succeeded" in text
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_reconnect_loop_exhausts_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """A permanently-down remote retries 5x then journals exhaustion + gives up."""
    fake_client = MagicMock()
    fake_client.connections = {}
    fake_client.get_tools = AsyncMock(side_effect=RuntimeError("still down"))

    async def _empty(client: Any, server_name: str) -> list[Any]:
        return []

    def _make_client(connections: dict[str, Any]) -> MagicMock:
        fake_client.connections = dict(connections)
        return fake_client

    monkeypatch.setattr(_manager_mod, "MultiServerMCPClient", _make_client)
    monkeypatch.setattr(MCPManager, "_list_prompts", staticmethod(_empty))
    monkeypatch.setattr(MCPManager, "_list_resources", staticmethod(_empty))

    sleeps: list[float] = []

    async def _record_sleep(sec: float) -> None:
        sleeps.append(sec)

    monkeypatch.setattr(
        "aura.infrastructure.mcp.manager.asyncio.sleep", _record_sleep
    )

    log_path = tmp_path / "j.jsonl"
    journal_module.configure(log_path)
    try:
        mgr = MCPManager(
            [_cfg("srv", transport="sse", url="https://x/mcp", command=None)]
        )
        await mgr.start_all()
        task = mgr._reconnect_tasks.get("srv")
        assert task is not None
        await task
        # Backoff capped at 60s: 1,2,4,8,16 across 5 attempts.
        assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]
        assert "mcp_reconnect_exhausted" in log_path.read_text(encoding="utf-8")
        assert "srv" not in mgr._reconnect_tasks
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_reconnect_loop_aborts_when_disabled_mid_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the operator disables a server mid-backoff, the loop bails before retry."""
    fake_client = MagicMock()
    fake_client.connections = {}
    fake_client.get_tools = AsyncMock(side_effect=RuntimeError("down"))

    async def _empty(client: Any, server_name: str) -> list[Any]:
        return []

    def _make_client(connections: dict[str, Any]) -> MagicMock:
        fake_client.connections = dict(connections)
        return fake_client

    monkeypatch.setattr(_manager_mod, "MultiServerMCPClient", _make_client)
    monkeypatch.setattr(MCPManager, "_list_prompts", staticmethod(_empty))
    monkeypatch.setattr(MCPManager, "_list_resources", staticmethod(_empty))

    mgr = MCPManager(
        [_cfg("srv", transport="sse", url="https://x/mcp", command=None)]
    )
    cfg = mgr._configs_all[0]

    async def _flip_to_disabled(_sec: float) -> None:
        mgr._state["srv"] = "disabled"

    monkeypatch.setattr(
        "aura.infrastructure.mcp.manager.asyncio.sleep", _flip_to_disabled
    )
    # Drive the loop directly; the disabled-state guard returns before reconnect.
    await mgr._reconnect_loop(cfg)
    fake_client.get_tools.assert_not_awaited()


@pytest.mark.asyncio
async def test_schedule_reconnect_is_single_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second schedule while a reconnect is live must not spawn a duplicate task."""
    _patch_manager_internals(monkeypatch)
    mgr = MCPManager(
        [_cfg("srv", transport="sse", url="https://x/mcp", command=None)]
    )
    cfg = mgr._configs_all[0]

    started = asyncio.Event()
    release = asyncio.Event()

    async def _block(_cfg: MCPServerConfig) -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr(mgr, "_reconnect_loop", _block)
    mgr._schedule_reconnect(cfg)
    first = mgr._reconnect_tasks["srv"]
    await started.wait()
    mgr._schedule_reconnect(cfg)  # second call is a no-op while first runs
    assert mgr._reconnect_tasks["srv"] is first
    release.set()
    await first


# ------------------------------ stop_all teardown -------------------------- #


@pytest.mark.asyncio
async def test_stop_all_cancels_live_reconnect_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stop_all must cancel pending reconnect tasks so the loop can't leak."""
    _patch_manager_internals(monkeypatch)
    mgr = MCPManager(
        [_cfg("srv", transport="sse", url="https://x/mcp", command=None)]
    )
    cfg = mgr._configs_all[0]

    async def _forever(_cfg: MCPServerConfig) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(mgr, "_reconnect_loop", _forever)
    mgr._schedule_reconnect(cfg)
    task = mgr._reconnect_tasks["srv"]
    await mgr.stop_all()
    assert task.cancelled() or task.done()
    assert mgr._reconnect_tasks == {}


@pytest.mark.asyncio
async def test_stop_all_awaits_async_close_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a future library adds aclose(), stop_all must await it on teardown."""
    closed: dict[str, bool] = {"aclose": False}

    class _AsyncCloseClient:
        def __init__(self, connections: dict[str, Any]) -> None:
            self.connections = dict(connections)

        async def get_tools(self, *, server_name: str) -> list[StructuredTool]:
            return []

        async def aclose(self) -> None:
            closed["aclose"] = True

    async def _empty(client: Any, server_name: str) -> list[Any]:
        return []

    monkeypatch.setattr(_manager_mod, "MultiServerMCPClient", _AsyncCloseClient)
    monkeypatch.setattr(MCPManager, "_list_prompts", staticmethod(_empty))
    monkeypatch.setattr(MCPManager, "_list_resources", staticmethod(_empty))

    mgr = MCPManager([_cfg("srv")])
    await mgr.start_all()
    await mgr.stop_all()
    assert closed["aclose"] is True
    assert mgr._client is None


@pytest.mark.asyncio
async def test_stop_all_awaits_awaitable_sync_close_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A close() returning a coroutine must be awaited, not left dangling."""
    closed: dict[str, bool] = {"close": False}

    class _AwaitableCloseClient:
        def __init__(self, connections: dict[str, Any]) -> None:
            self.connections = dict(connections)

        async def get_tools(self, *, server_name: str) -> list[StructuredTool]:
            return []

        def close(self) -> Awaitable[None]:
            async def _do() -> None:
                closed["close"] = True

            return _do()

    async def _empty(client: Any, server_name: str) -> list[Any]:
        return []

    monkeypatch.setattr(_manager_mod, "MultiServerMCPClient", _AwaitableCloseClient)
    monkeypatch.setattr(MCPManager, "_list_prompts", staticmethod(_empty))
    monkeypatch.setattr(MCPManager, "_list_resources", staticmethod(_empty))

    mgr = MCPManager([_cfg("srv")])
    await mgr.start_all()
    await mgr.stop_all()
    assert closed["close"] is True


# ------------------------------ approval gate ------------------------------ #


def _patch_approvals(
    monkeypatch: pytest.MonkeyPatch,
    *,
    project_names: set[str],
    approved: set[str],
) -> dict[str, list[str]]:
    """Stub the approvals + store seams; return a log of approve/revoke calls."""
    calls: dict[str, list[str]] = {"approve": [], "revoke": []}

    def _is_approved(cfg: MCPServerConfig, **_kw: Any) -> bool:
        return cfg.name in approved

    def _approve(cfg: MCPServerConfig, **_kw: Any) -> object:
        calls["approve"].append(cfg.name)
        approved.add(cfg.name)
        return None

    def _revoke(name: str, **_kw: Any) -> bool:
        calls["revoke"].append(name)
        approved.discard(name)
        return True

    base = "aura.infrastructure.mcp.manager.mcp_approvals"
    store = "aura.infrastructure.mcp.manager.mcp_store.project_layer_names"
    monkeypatch.setattr(f"{base}.is_approved", _is_approved)
    monkeypatch.setattr(f"{base}.approve", _approve)
    monkeypatch.setattr(f"{base}.revoke", _revoke)
    monkeypatch.setattr(f"{base}.project_key", lambda: "proj")
    monkeypatch.setattr(store, lambda: set(project_names))
    return calls


def test_project_server_unapproved_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unapproved project-layer server starts in the 'unapproved' state."""
    _patch_approvals(monkeypatch, project_names={"proj_srv"}, approved=set())
    mgr = MCPManager([_cfg("proj_srv")])
    row = next(r for r in mgr.status() if r.name == "proj_srv")
    assert row.state == "unapproved"
    assert mgr.unapproved_server_names() == {"proj_srv"}


@pytest.mark.asyncio
async def test_start_all_skips_unapproved_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unapproved project servers are never connected by start_all."""
    fake_client = _patch_manager_internals(
        monkeypatch, tools_by_server={"proj_srv": [_fake_tool("t")]}
    )
    _patch_approvals(monkeypatch, project_names={"proj_srv"}, approved=set())
    mgr = MCPManager([_cfg("proj_srv")])
    tools, commands = await mgr.start_all()
    assert tools == []
    assert commands == []
    fake_client.get_tools.assert_not_awaited()


@pytest.mark.asyncio
async def test_approve_user_scope_server_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user-scope server needs no approval; approve() says so and skips persist."""
    calls = _patch_approvals(monkeypatch, project_names=set(), approved=set())
    _patch_manager_internals(monkeypatch, tools_by_server={"srv": [_fake_tool("t")]})
    mgr = MCPManager([_cfg("srv")])
    text = await mgr.approve("srv")
    assert "user-scope" in text
    assert calls["approve"] == []


@pytest.mark.asyncio
async def test_approve_project_server_persists_and_connects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approving a project server persists approval AND brings it online."""
    _patch_manager_internals(
        monkeypatch, tools_by_server={"proj_srv": [_fake_tool("t")]}
    )
    calls = _patch_approvals(monkeypatch, project_names={"proj_srv"}, approved=set())
    mgr = MCPManager([_cfg("proj_srv")])
    assert mgr.unapproved_server_names() == {"proj_srv"}
    text = await mgr.approve("proj_srv")
    assert "approved and connected" in text
    assert calls["approve"] == ["proj_srv"]
    row = next(r for r in mgr.status() if r.name == "proj_srv")
    assert row.state == "connected"
    assert mgr.unapproved_server_names() == set()


@pytest.mark.asyncio
async def test_approve_disabled_project_server_stays_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approving a DISABLED project server records approval but does not connect."""
    fake_client = _patch_manager_internals(monkeypatch)
    _patch_approvals(monkeypatch, project_names={"proj_srv"}, approved=set())
    mgr = MCPManager([_cfg("proj_srv", enabled=False)])
    text = await mgr.approve("proj_srv")
    assert "approved (state:" in text
    fake_client.get_tools.assert_not_awaited()


@pytest.mark.asyncio
async def test_approve_unknown_name_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """approve() on an unknown server name returns a textual error, not a raise."""
    _patch_approvals(monkeypatch, project_names=set(), approved=set())
    mgr = MCPManager([_cfg("known")])
    text = await mgr.approve("ghost")
    assert "no MCP server named" in text


@pytest.mark.asyncio
async def test_revoke_unknown_name_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """revoke() on an unknown server name returns a textual error, not a raise."""
    _patch_approvals(monkeypatch, project_names=set(), approved=set())
    mgr = MCPManager([_cfg("known")])
    text = await mgr.revoke("ghost")
    assert "no MCP server named" in text


@pytest.mark.asyncio
async def test_revoke_tears_down_and_flips_to_unapproved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revoking a connected project server disconnects it and persists the revoke."""
    _patch_manager_internals(
        monkeypatch, tools_by_server={"proj_srv": [_fake_tool("t")]}
    )
    calls = _patch_approvals(
        monkeypatch, project_names={"proj_srv"}, approved={"proj_srv"}
    )
    mgr = MCPManager([_cfg("proj_srv")])
    await mgr.start_all()
    assert next(r for r in mgr.status() if r.name == "proj_srv").state == "connected"
    text = await mgr.revoke("proj_srv")
    assert "revoked and disconnected" in text
    assert calls["revoke"] == ["proj_srv"]
    row = next(r for r in mgr.status() if r.name == "proj_srv")
    assert row.state == "unapproved"
    assert row.tool_count == 0
    assert "proj_srv" in mgr.unapproved_server_names()


@pytest.mark.asyncio
async def test_revoke_then_approve_is_idempotent_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approve→revoke→approve must converge to connected without leaking state."""
    _patch_manager_internals(
        monkeypatch, tools_by_server={"proj_srv": [_fake_tool("t")]}
    )
    _patch_approvals(monkeypatch, project_names={"proj_srv"}, approved=set())
    mgr = MCPManager([_cfg("proj_srv")])
    await mgr.approve("proj_srv")
    await mgr.revoke("proj_srv")
    assert "proj_srv" in mgr.unapproved_server_names()
    text = await mgr.approve("proj_srv")
    assert "approved and connected" in text
    assert mgr.unapproved_server_names() == set()


def test_construction_store_failure_degrades_to_empty_approvals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store blow-up at construction must NOT crash; approvals degrade to empty."""

    def _boom() -> set[str]:
        raise RuntimeError("store unreadable")

    monkeypatch.setattr(
        "aura.infrastructure.mcp.manager.mcp_store.project_layer_names", _boom
    )
    mgr = MCPManager([_cfg("srv")])  # project_server_names=None triggers the store
    assert mgr.unapproved_server_names() == set()


# -------------------------------- enable/reconnect errors ------------------ #


@pytest.mark.asyncio
async def test_enable_surfaces_error_on_failed_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enable() that can't connect returns a failure string with the cause."""
    _patch_manager_internals(
        monkeypatch, errors_by_server={"srv": RuntimeError("port closed")}
    )
    mgr = MCPManager([_cfg("srv")])
    text = await mgr.enable("srv")
    assert "failed to connect" in text
    assert "port closed" in text


# ----------------------------------- reload -------------------------------- #


@pytest.mark.asyncio
async def test_reload_reports_added_and_removed_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reload() returns a '+N -M' summary and forgets the removed server's state."""
    _patch_manager_internals(monkeypatch, tools_by_server={"a": [_fake_tool("t")]})
    _patch_approvals(monkeypatch, project_names=set(), approved=set())
    mgr = MCPManager([_cfg("a"), _cfg("b")])
    summary = await mgr.reload([_cfg("a"), _cfg("c")])
    assert summary == "+1 -1"
    names = {r.name for r in mgr.status()}
    assert names == {"a", "c"}
    assert "b" not in mgr.known_server_names()


@pytest.mark.asyncio
async def test_reload_marks_new_project_server_unapproved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newly-added unapproved project server lands in the 'unapproved' state."""
    _patch_manager_internals(monkeypatch)
    _patch_approvals(monkeypatch, project_names=set(), approved=set())
    mgr = MCPManager([_cfg("a")])
    monkeypatch.setattr(
        "aura.infrastructure.mcp.manager.mcp_store.project_layer_names",
        lambda: {"newproj"},
    )
    summary = await mgr.reload([_cfg("a"), _cfg("newproj")])
    assert summary == "+1 -0"
    row = next(r for r in mgr.status() if r.name == "newproj")
    assert row.state == "unapproved"


@pytest.mark.asyncio
async def test_reload_added_disabled_server_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newly-added disabled (non-project) server lands in 'disabled', not error."""
    _patch_manager_internals(monkeypatch)
    _patch_approvals(monkeypatch, project_names=set(), approved=set())
    mgr = MCPManager([_cfg("a")])
    summary = await mgr.reload([_cfg("a"), _cfg("off", enabled=False)])
    assert summary == "+1 -0"
    row = next(r for r in mgr.status() if r.name == "off")
    assert row.state == "disabled"


@pytest.mark.asyncio
async def test_reload_explicit_empty_project_names_no_store_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing project_server_names explicitly must bypass the store entirely."""
    _patch_manager_internals(monkeypatch)

    def _boom() -> set[str]:
        raise AssertionError("store must not be consulted")

    monkeypatch.setattr(
        "aura.infrastructure.mcp.manager.mcp_store.project_layer_names", _boom
    )
    mgr = MCPManager([_cfg("a")], project_server_names=set())
    summary = await mgr.reload([_cfg("a"), _cfg("b")], project_server_names=set())
    assert summary == "+1 -0"


def test_reload_store_failure_degrades_to_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store failure during reload must degrade to empty approvals, not raise."""

    async def _run() -> str:
        mgr = MCPManager([_cfg("a")], project_server_names=set())

        def _boom() -> set[str]:
            raise RuntimeError("store down")

        monkeypatch.setattr(
            "aura.infrastructure.mcp.manager.mcp_store.project_layer_names", _boom
        )
        return await mgr.reload([_cfg("a"), _cfg("b")])

    assert asyncio.run(_run()) == "+1 -0"


@pytest.mark.asyncio
async def test_reload_drops_live_client_connection_for_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing a started server via reload must purge its live client connection."""
    fake_client = _patch_manager_internals(
        monkeypatch, tools_by_server={"a": [_fake_tool("t")], "b": [_fake_tool("u")]}
    )
    _patch_approvals(monkeypatch, project_names=set(), approved=set())
    mgr = MCPManager([_cfg("a"), _cfg("b")])
    await mgr.start_all()
    assert "b" in fake_client.connections
    await mgr.reload([_cfg("a")])
    assert "b" not in fake_client.connections


# --------------------- reconnect-task cancellation seams ------------------- #


@pytest.mark.asyncio
async def test_reconnect_loop_sleep_cancelled_returns_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel landing during backoff sleep ends the loop without re-raising."""
    _patch_manager_internals(monkeypatch)
    mgr = MCPManager(
        [_cfg("srv", transport="sse", url="https://x/mcp", command=None)]
    )
    cfg = mgr._configs_all[0]

    async def _cancelling_sleep(_sec: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "aura.infrastructure.mcp.manager.asyncio.sleep", _cancelling_sleep
    )
    # CancelledError inside the sleep is caught and converted to a clean return.
    await mgr._reconnect_loop(cfg)


@pytest.mark.asyncio
async def test_cancel_reconnect_task_self_cancel_is_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling from inside the reconnect task drops the handle without raising."""
    _patch_manager_internals(monkeypatch)
    mgr = MCPManager(
        [_cfg("srv", transport="sse", url="https://x/mcp", command=None)]
    )
    observed: dict[str, bool] = {"survived": False}

    async def _self_cancelling(_cfg: MCPServerConfig) -> None:
        # While running, ask the manager to cancel *this* task: it must no-op
        # rather than raise CancelledError into us.
        mgr._cancel_reconnect_task("srv")
        observed["survived"] = True

    monkeypatch.setattr(mgr, "_reconnect_loop", _self_cancelling)
    mgr._schedule_reconnect(mgr._configs_all[0])
    task = mgr._reconnect_tasks.get("srv")
    if task is not None:
        await task
    assert observed["survived"] is True
    assert "srv" not in mgr._reconnect_tasks


@pytest.mark.asyncio
async def test_cancel_reconnect_task_cancels_live_external_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling a still-running task from outside it must actually .cancel() it."""
    _patch_manager_internals(monkeypatch)
    mgr = MCPManager(
        [_cfg("srv", transport="sse", url="https://x/mcp", command=None)]
    )

    async def _forever(_cfg: MCPServerConfig) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(mgr, "_reconnect_loop", _forever)
    mgr._schedule_reconnect(mgr._configs_all[0])
    task = mgr._reconnect_tasks["srv"]
    await asyncio.sleep(0)  # let the task start before we cancel it
    mgr._cancel_reconnect_task("srv")  # external caller → real .cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert "srv" not in mgr._reconnect_tasks


@pytest.mark.asyncio
async def test_reconnect_after_disable_re_adds_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reconnect() after disable re-appends the dropped config and reconnects."""
    _patch_manager_internals(
        monkeypatch, tools_by_server={"srv": [_fake_tool("x")]}
    )
    mgr = MCPManager([_cfg("srv")])
    await mgr.start_all()
    await mgr.disable("srv")
    assert all(c.name != "srv" for c in mgr._configs)  # disable dropped it
    text = await mgr.reconnect("srv")
    assert "reconnected" in text
    assert any(c.name == "srv" for c in mgr._configs)  # reconnect re-added it
