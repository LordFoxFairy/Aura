"""Tests for :class:`aura.core.runtime.mcp.McpRuntime` (Phase 2 Task 8).

The runtime is exercised in isolation — no :class:`Agent`, no
:class:`AgentLoop`, no real MCP transport. Each case constructs the
runtime with a fake manager factory so connect / disconnect / status
behaviour can be verified without spinning subprocesses.

Coverage targets (per Phase 2 plan Task 8 acceptance):

1. ``__init__`` is sync + bookkeeping-only (no manager spun).
2. ``connect_all`` with no servers configured is a no-op (returns
   ``None``, no factory call).
3. ``connect_all`` happy path returns merged tool pool, populates
   ``manager`` / ``commands`` / ``tools``.
4. ``disconnect_all`` runs ``stop_all`` and clears manager state.
5. ``connected_server_names`` reflects the manager's status snapshot.
6. ``replace_registry_contents`` swaps a registry in place.
7. ``connect_all`` failure is journaled + swallowed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from aura.config.schema import MCPServerConfig
from aura.core.persistence import journal
from aura.core.registry import ToolRegistry
from aura.core.runtime.mcp import McpRuntime
from aura.schemas.tool import ToolMetadata


# ``_FakeManager`` is a structural stand-in. mypy can't see the
# duck-typed conformance; we cast at call sites to keep the McpRuntime
# factory type strict for production callers while letting tests inject
# the lightweight fake.
def _as_factory(
    cls: type[Any],
) -> Any:
    """Cast a fake manager class to the production factory type."""
    return cast("Any", cls)

# ----------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------


class _Args(BaseModel):
    q: str = ""


async def _coro(q: str = "") -> dict[str, Any]:
    return {}


def _fake_tool(name: str) -> BaseTool:
    """Build a StructuredTool with an aura_metadata stamp so it survives
    ToolRegistry's strict-typing guard at registration."""
    tool = StructuredTool(
        name=name,
        description=name,
        args_schema=_Args,
        coroutine=_coro,
    )
    object.__setattr__(
        tool,
        "aura_metadata",
        ToolMetadata(
            is_read_only=False,
            is_destructive=True,
            is_concurrency_safe=False,
            rule_matcher=None,
            args_preview=None,
            timeout_sec=None,
        ),
    )
    return tool


class _FakeStatus:
    def __init__(self, name: str, state: str) -> None:
        self.name = name
        self.state = state


class _FakeManager:
    """Minimal MCPManager stand-in.

    ``start_all`` returns the (tools, commands) the runtime forwards to
    callers; ``stop_all`` toggles a flag the disconnect tests check.
    ``status`` / ``resources_catalogue`` are pure-sync stubs.
    """

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self.configs = configs
        self.start_called = 0
        self.stop_called = 0
        self.tools: list[BaseTool] = [_fake_tool("mcp__srv__one")]
        self.commands: list[Any] = ["cmd_a"]
        self.statuses: list[_FakeStatus] = [
            _FakeStatus("srv_a", "connected"),
            _FakeStatus("srv_b", "disconnected"),
        ]

    async def start_all(self) -> tuple[list[BaseTool], list[Any]]:
        self.start_called += 1
        return self.tools, self.commands

    async def stop_all(self) -> None:
        self.stop_called += 1

    def status(self) -> list[_FakeStatus]:
        return list(self.statuses)

    def resources_catalogue(self) -> list[tuple[str, str, str, str, str | None]]:
        return []


def _server_cfg(name: str = "srv") -> MCPServerConfig:
    return MCPServerConfig(name=name, command="echo", args=["hello"])


def _read_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ----------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------


def test_init_is_sync_and_does_not_spin_manager() -> None:
    """Construction must not call the factory or start any server.

    The Agent's sync ``__init__`` requires McpRuntime to stay sync —
    otherwise the existing Agent(...) call sites would have to thread
    an event loop through. Verifies the factory is untouched and the
    runtime advertises ``has_servers`` based purely on the configs
    list.
    """
    factory_calls: list[list[MCPServerConfig]] = []

    def _factory(configs: list[MCPServerConfig]) -> Any:
        factory_calls.append(configs)
        return _FakeManager(configs)

    runtime = McpRuntime(
        [_server_cfg("a"), _server_cfg("b")],
        manager_factory=_factory,
    )
    assert runtime.has_servers is True
    assert runtime.is_connected is False
    assert runtime.manager is None
    assert runtime.commands == []
    assert runtime.tools == []
    assert factory_calls == []  # factory not invoked at construction


@pytest.mark.asyncio
async def test_connect_all_no_servers_is_noop() -> None:
    """No configured servers → connect_all returns None, no factory call."""
    factory_calls: list[Any] = []

    def _factory(configs: list[MCPServerConfig]) -> Any:
        factory_calls.append(configs)
        return _FakeManager(configs)

    runtime = McpRuntime([], manager_factory=_factory)
    merged = await runtime.connect_all([])
    assert merged is None
    assert runtime.manager is None
    assert factory_calls == []


@pytest.mark.asyncio
async def test_connect_all_happy_path_returns_merged_pool(
    tmp_path: Path,
) -> None:
    """Successful connect: factory builds manager, start_all called,
    merged pool returned, journal event ``mcp_aconnect_done`` fires."""
    log_path = tmp_path / "j.jsonl"
    journal.configure(log_path)
    try:
        builtin = _fake_tool("read_file")
        runtime = McpRuntime(
            [_server_cfg("srv")],
            manager_factory=_as_factory(_FakeManager),
        )
        merged = await runtime.connect_all([builtin])
        assert merged is not None
        # Builtin + MCP both present.
        assert "read_file" in merged
        assert "mcp__srv__one" in merged
        # Runtime now exposes the fake manager + tool/command snapshots.
        assert isinstance(runtime.manager, _FakeManager)
        assert runtime.is_connected is True
        assert runtime.commands == ["cmd_a"]
        assert [t.name for t in runtime.tools] == ["mcp__srv__one"]
        # Journal saw the success event.
        events = _read_journal(log_path)
        done = [e for e in events if e.get("event") == "mcp_aconnect_done"]
        assert len(done) == 1
        assert done[0]["tool_count"] == 1
        assert done[0]["command_count"] == 1
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_disconnect_all_clears_manager_and_emits_mcp_stopped(
    tmp_path: Path,
) -> None:
    """Happy-path teardown: stop_all called, manager reset to None,
    ``mcp_stopped`` journal event fires (no timeout / error events)."""
    log_path = tmp_path / "j.jsonl"
    journal.configure(log_path)
    try:
        runtime = McpRuntime(
            [_server_cfg("srv")],
            manager_factory=_as_factory(_FakeManager),
        )
        await runtime.connect_all([])
        mgr = runtime.manager
        assert isinstance(mgr, _FakeManager)
        await runtime.disconnect_all(session_id="t1", timeout_sec=1.0)
        assert mgr.stop_called == 1
        assert runtime.manager is None
        assert runtime.is_connected is False
        assert runtime.tools == []
        assert runtime.commands == []
        events = _read_journal(log_path)
        stopped = [e for e in events if e.get("event") == "mcp_stopped"]
        assert len(stopped) == 1
        assert stopped[0]["session"] == "t1"
        # No timeout / error events on the happy path.
        assert not any(e.get("event") == "mcp_close_timeout" for e in events)
        assert not any(e.get("event") == "mcp_close_error" for e in events)
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_disconnect_all_no_manager_is_noop(tmp_path: Path) -> None:
    """Calling disconnect without a live manager is a silent no-op —
    the runtime must tolerate ``aclose`` after a failed connect."""
    log_path = tmp_path / "j.jsonl"
    journal.configure(log_path)
    try:
        runtime = McpRuntime([], manager_factory=_as_factory(_FakeManager))
        await runtime.disconnect_all(session_id="t2", timeout_sec=1.0)
        events = _read_journal(log_path)
        # No close-related events should be journaled.
        assert not any(
            e.get("event", "").startswith("mcp_close") for e in events
        )
        assert not any(e.get("event") == "mcp_stopped" for e in events)
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_connected_server_names_reflects_status_snapshot() -> None:
    """``connected_server_names`` returns only servers whose status is
    ``connected``. Used by aclose to populate ``servers_hanging``."""
    runtime = McpRuntime(
        [_server_cfg("srv")],
        manager_factory=_as_factory(_FakeManager),
    )
    # Pre-connect: empty.
    assert runtime.connected_server_names() == []
    await runtime.connect_all([])
    names = runtime.connected_server_names()
    assert names == ["srv_a"]


def test_replace_registry_contents_swaps_in_place() -> None:
    """``replace_registry_contents`` clears + re-fills a registry while
    preserving the registry instance identity (the loop binds the
    instance once at construction)."""
    initial = ToolRegistry([_fake_tool("old")])
    runtime = McpRuntime([], manager_factory=_as_factory(_FakeManager))
    new_tool = _fake_tool("new")
    runtime.replace_registry_contents(initial, {"new": new_tool})
    assert "old" not in initial
    assert "new" in initial
    assert list(initial.values()) == [new_tool]


@pytest.mark.asyncio
async def test_connect_all_swallows_factory_exception(tmp_path: Path) -> None:
    """A factory / start_all failure must be journaled as
    ``mcp_aconnect_failed`` and swallowed — the agent keeps running
    without that server's tools (graceful degradation)."""
    log_path = tmp_path / "j.jsonl"
    journal.configure(log_path)
    try:
        def _broken(configs: list[MCPServerConfig]) -> Any:
            raise RuntimeError("transport unavailable")

        runtime = McpRuntime(
            [_server_cfg("srv")],
            manager_factory=_broken,
        )
        merged = await runtime.connect_all([])
        assert merged is None
        assert runtime.manager is None
        events = _read_journal(log_path)
        failed = [e for e in events if e.get("event") == "mcp_aconnect_failed"]
        assert len(failed) == 1
        assert "transport unavailable" in failed[0]["error"]
    finally:
        journal.reset()
