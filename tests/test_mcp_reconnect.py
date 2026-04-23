"""Tests for MCPManager's auto-reconnect + per-op timeout resilience layers.

These scenarios exercise the two gaps closed by the D-audit bundle:

- **Auto-reconnect** (gap #1): remote-transport disconnects schedule an
  exponential-backoff retry loop (1s, 2s, 4s, 8s, 16s — capped at 60s,
  max 5 attempts) that mirrors ``useManageMCPConnections.ts`` in
  claude-code v2.1.88. Stdio servers intentionally skip the retry path —
  a dead subprocess is user-visible and backoff won't heal it.

- **Per-op timeout** (gap #4): client-facing MCP calls run under
  :func:`asyncio.wait_for`; a stalled server surfaces a descriptive
  :class:`RuntimeError` instead of wedging the agent loop.

All tests monkeypatch ``asyncio.sleep`` where backoff pauses would
otherwise slow the suite, and patch ``MultiServerMCPClient`` to avoid
spinning real transports. See ``test_mcp_manager.py`` for the
happy-path coverage these build on.
"""

from __future__ import annotations

import asyncio
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


def _stub_list_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``_list_prompts`` / ``_list_resources`` to empty lists.

    The reconnect + timeout tests only care about ``get_tools`` behaviour;
    the other discovery paths must not pollute state.
    """
    async def _empty(client: Any, server_name: str) -> list[Any]:
        return []

    monkeypatch.setattr(MCPManager, "_list_prompts", staticmethod(_empty))
    monkeypatch.setattr(MCPManager, "_list_resources", staticmethod(_empty))


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock,
) -> None:
    """Wire a single shared fake client into the manager's ctor path.

    Library populates ``.connections`` from the ctor arg; mimic that so
    ``session(name)`` / ``get_tools(server_name=...)`` lookups behave
    like the real library would.
    """
    from aura.core.mcp import manager as manager_mod

    def _make_client(connections: dict[str, Any]) -> MagicMock:
        fake_client.connections = dict(connections)
        return fake_client

    monkeypatch.setattr(manager_mod, "MultiServerMCPClient", _make_client)


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse backoff sleeps in the reconnect loop.

    The manager calls ``asyncio.sleep(backoff)`` inside
    ``_reconnect_loop``. Real waits (1s → 16s → ...) would add ~31s to
    each failure test. We monkeypatch the stdlib ``asyncio.sleep`` so
    cancellation semantics stay intact (still an awaitable; still
    raises CancelledError when the task is cancelled). Patching via
    string-path target ("asyncio.sleep") keeps mypy happy — re-export
    attribute checks would reject ``manager_mod.asyncio``.
    """
    real_sleep = asyncio.sleep

    async def _fast(delay: float) -> None:  # noqa: ARG001
        # Yield to the loop so CancelledError can reach us if the task
        # was cancelled — matching real ``asyncio.sleep`` cancel-points.
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _fast)


# ---------------------------------------------------------------------------
# Auto-reconnect — remote transports
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_reconnect_retries_sse_server_on_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSE server failure → reconnect task scheduled → succeeds on 2nd try.

    ``get_tools`` fails on attempt 1 (initial ``start_all`` call),
    succeeds on attempt 2 (first retry). The manager must:
    - schedule a background task on failure (remote transport),
    - tick through the backoff sleep,
    - call ``_connect_one`` again,
    - flip ``_state`` back to ``"connected"`` on success.
    """
    _stub_list_helpers(monkeypatch)

    calls: list[str] = []

    async def _get_tools(*, server_name: str) -> list[StructuredTool]:
        calls.append(server_name)
        if len(calls) == 1:
            raise RuntimeError("sse handshake dropped")
        return [_fake_tool("ok")]

    fake_client = MagicMock()
    fake_client.connections = {}
    fake_client.get_tools = AsyncMock(side_effect=_get_tools)
    _install_fake_client(monkeypatch, fake_client)

    mgr = MCPManager([
        MCPServerConfig(
            name="remote", transport="sse", url="http://x/sse",
        ),
    ])
    await mgr.start_all()
    # First attempt failed synchronously; state should be "error" and a
    # reconnect task must be in flight.
    status = {s.name: s for s in mgr.status()}
    assert status["remote"].state == "error"
    assert "remote" in mgr._reconnect_tasks  # noqa: SLF001

    # Drain the reconnect task — our fast-sleep fixture means it completes
    # almost immediately.
    await mgr._reconnect_tasks["remote"]  # noqa: SLF001

    # 2nd call succeeded; state must be back to connected.
    status = {s.name: s for s in mgr.status()}
    assert status["remote"].state == "connected"
    assert status["remote"].tool_count == 1
    assert len(calls) == 2
    await mgr.stop_all()


@pytest.mark.asyncio
async def test_auto_reconnect_gives_up_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Always-failing remote server retries 5x then sticks in error state.

    After the initial failure + 5 retries, ``_state`` stays ``"error"``,
    no infinite loop, and the task handle is cleared so the map is
    clean for a future operator-driven reconnect.
    """
    _stub_list_helpers(monkeypatch)

    calls: list[str] = []

    async def _always_fail(*, server_name: str) -> list[StructuredTool]:
        calls.append(server_name)
        raise RuntimeError("permanently broken")

    fake_client = MagicMock()
    fake_client.connections = {}
    fake_client.get_tools = AsyncMock(side_effect=_always_fail)
    _install_fake_client(monkeypatch, fake_client)

    mgr = MCPManager([
        MCPServerConfig(
            name="doomed",
            transport="streamable_http",
            url="http://x/mcp",
        ),
    ])
    await mgr.start_all()
    # Drive the scheduled retry task to completion.
    task = mgr._reconnect_tasks.get("doomed")  # noqa: SLF001
    assert task is not None
    await task

    # 1 initial + 5 retries = 6 total ``get_tools`` calls. ``_connect_one``
    # re-schedules on each failure, but the "already in-flight" guard
    # keeps the count bounded at the configured max.
    assert len(calls) == 1 + 5
    assert mgr._state["doomed"] == "error"  # noqa: SLF001
    # Handle cleared after exhaustion so a later /mcp reconnect can
    # freshly spawn a new loop.
    assert "doomed" not in mgr._reconnect_tasks  # noqa: SLF001
    await mgr.stop_all()


@pytest.mark.asyncio
async def test_stdio_transport_does_not_auto_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stdio server failure → state = error, no retry scheduled.

    Stdio process death is not something backoff can fix (the child has
    exited; reconnect would just spawn a fresh subprocess on the same
    broken command). The manager intentionally skips the retry path.
    """
    _stub_list_helpers(monkeypatch)

    async def _fail(*, server_name: str) -> list[StructuredTool]:
        raise RuntimeError("stdio spawn failed")

    fake_client = MagicMock()
    fake_client.connections = {}
    fake_client.get_tools = AsyncMock(side_effect=_fail)
    _install_fake_client(monkeypatch, fake_client)

    mgr = MCPManager([
        MCPServerConfig(name="local", command="npx", args=[]),
    ])
    await mgr.start_all()
    assert mgr._state["local"] == "error"  # noqa: SLF001
    # No task scheduled — stdio opts out of auto-reconnect.
    assert "local" not in mgr._reconnect_tasks  # noqa: SLF001
    await mgr.stop_all()


@pytest.mark.asyncio
async def test_stop_all_cancels_pending_reconnect_timers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stop_all must cancel any in-flight reconnect task cleanly.

    We schedule a reconnect via a failing start, then immediately call
    ``stop_all`` before the task can finish. The task handle must end
    up cancelled (or finished) with no unhandled exception.
    """
    _stub_list_helpers(monkeypatch)

    # Freeze reconnect attempts mid-flight: sleep indefinitely so the
    # task can't complete before we call stop_all. Override the autouse
    # ``_fast_sleep`` fixture by re-patching to an unbounded wait.
    async def _forever(delay: float) -> None:  # noqa: ARG001
        await asyncio.Event().wait()  # hangs until cancelled

    monkeypatch.setattr("asyncio.sleep", _forever)

    async def _fail(*, server_name: str) -> list[StructuredTool]:
        raise RuntimeError("boom")

    fake_client = MagicMock()
    fake_client.connections = {}
    fake_client.get_tools = AsyncMock(side_effect=_fail)
    _install_fake_client(monkeypatch, fake_client)

    mgr = MCPManager([
        MCPServerConfig(name="remote", transport="sse", url="http://x"),
    ])
    await mgr.start_all()
    task = mgr._reconnect_tasks.get("remote")  # noqa: SLF001
    assert task is not None
    assert not task.done()

    await mgr.stop_all()

    # Task handle map cleared, underlying task cancelled (or otherwise
    # finalised — the gather in stop_all absorbs the CancelledError).
    assert mgr._reconnect_tasks == {}  # noqa: SLF001
    assert task.done()


# ---------------------------------------------------------------------------
# Per-op timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_resource_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """A hanging read_resource must surface a descriptive RuntimeError.

    We plant a fake resource in the catalogue, then open a session whose
    ``read_resource`` never returns. ``_run_with_timeout`` must convert
    the asyncio timeout into a user-readable message naming the op +
    server + timeout seconds.
    """
    _stub_list_helpers(monkeypatch)
    from aura.core import journal

    journal.configure(tmp_path / "journal.jsonl")

    # Build a session context manager whose ``read_resource`` hangs.
    class _HangSession:
        async def __aenter__(self) -> _HangSession:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def read_resource(self, uri: Any) -> Any:
            await asyncio.Event().wait()  # never resolves

    fake_client = MagicMock()
    fake_client.connections = {"s": {}}
    fake_client.session = MagicMock(return_value=_HangSession())
    fake_client.get_tools = AsyncMock(return_value=[])
    _install_fake_client(monkeypatch, fake_client)

    mgr = MCPManager(
        [MCPServerConfig(name="s", transport="sse", url="http://x")],
        op_timeout_sec=0.05,
    )
    await mgr.start_all()

    # Manually plant a resource so ``read_resource`` dispatch has
    # something to look up (the real list_resources path is stubbed out).
    fake_resource = MagicMock()
    fake_resource.uri = "mem://doc"
    mgr._resources[("s", "mem://doc")] = fake_resource  # noqa: SLF001

    with pytest.raises(RuntimeError, match="read_resource.*timed out"):
        await mgr.read_resource("mem://doc")

    # State must not be corrupted — the manager stays "connected"; the
    # timeout is a transient op failure, not a full disconnect.
    assert mgr._state["s"] == "connected"  # noqa: SLF001
    await mgr.stop_all()
    journal.reset()


@pytest.mark.asyncio
async def test_get_prompt_timeout_from_manager_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """start_all must time out ``get_tools`` if the server hangs.

    We hang the top-level ``get_tools`` call so ``_run_with_timeout``
    fires, then assert the server lands in ``"error"`` state with a
    descriptive timeout message (no silent hang). Smoke-tests the
    wait_for path on the startup-critical call too.
    """
    _stub_list_helpers(monkeypatch)

    async def _hang(*, server_name: str) -> list[StructuredTool]:
        await asyncio.Event().wait()  # never returns
        return []  # unreachable; present to satisfy type checker

    fake_client = MagicMock()
    fake_client.connections = {}
    fake_client.get_tools = AsyncMock(side_effect=_hang)
    _install_fake_client(monkeypatch, fake_client)

    # Short timeout so we don't block the suite; disable retries by
    # using stdio transport (stdio never auto-reconnects).
    mgr = MCPManager(
        [MCPServerConfig(name="stuck", command="npx", args=[])],
        op_timeout_sec=0.05,
    )
    await mgr.start_all()

    status = {s.name: s for s in mgr.status()}
    assert status["stuck"].state == "error"
    assert status["stuck"].error_message is not None
    assert "timed out" in status["stuck"].error_message
    assert "get_tools" in status["stuck"].error_message
    await mgr.stop_all()


@pytest.mark.asyncio
async def test_op_timeout_resolution_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AURA_MCP_TIMEOUT_SEC env var overrides the default.

    Regression guard: we want the env var respected at ctor time so
    operators can lift or tighten the cap without a code change. The
    explicit kwarg still wins over env (tested implicitly by
    ``test_read_resource_timeout`` which passes 0.05s while env is
    unset).
    """
    monkeypatch.setenv("AURA_MCP_TIMEOUT_SEC", "7.5")
    mgr = MCPManager([])
    assert mgr.op_timeout_sec == pytest.approx(7.5)

    # Invalid env → silently fall back to default (30s). Don't want a
    # typo in a shell-rc to block MCP startup.
    monkeypatch.setenv("AURA_MCP_TIMEOUT_SEC", "not-a-number")
    mgr2 = MCPManager([])
    assert mgr2.op_timeout_sec == pytest.approx(30.0)
