"""Tests for Workstream B3 — Agent.aclose timeout-bounded MCP shutdown.

Contract (see `docs/specs/2026-04-23-aura-main-channel-parity.md` §B3):

- ``Agent.aclose(*, mcp_timeout=5.0)`` is the canonical async shutdown entry.
  It runs ``MCPManager.stop_all`` under ``asyncio.wait_for`` and, on
  timeout, cancels the hanging coroutine + emits a ``mcp_close_timeout``
  journal event with the list of servers still in the ``connected`` state.
- ``Agent.close()`` is a thin sync wrapper — no active event loop → spin
  one via ``asyncio.run(aclose(...))``; inside an active loop, it raises
  instead of the old fire-and-forget ``loop.create_task`` pattern.
- Fast, error-free stop_all paths emit ``mcp_stopped``; unexpected
  exceptions emit ``mcp_close_error``. No event overlap per invocation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from aura.config.schema import AuraConfig
from aura.core import journal
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel


def _minimal_config() -> AuraConfig:
    return AuraConfig.model_validate(
        {
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }
    )


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "aura.db")


def _journal_events(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class _HangingManager:
    """Fake MCPManager whose ``stop_all`` hangs forever on cancel-hostile sleep.

    Mirrors the real manager's narrow public surface Agent.aclose relies on:
    ``stop_all`` coroutine + ``status()`` (for the ``servers_hanging`` field).
    """

    def __init__(self, connected_servers: list[str]) -> None:
        self._connected = list(connected_servers)
        self.stop_called = False
        self.cancelled = False

    async def stop_all(self) -> None:
        self.stop_called = True
        try:
            # 60s is long enough that a 0.1s mcp_timeout MUST fire — any
            # accidental "wait for the real coroutine" path would blow the
            # <0.5s budget the contract promises.
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    def status(self) -> list[Any]:
        # Agent computes servers_hanging from .status() entries whose state
        # is still "connected" at shutdown time.
        from aura.core.mcp.manager import MCPServerStatus

        return [
            MCPServerStatus(
                name=name,
                transport="stdio",
                state="connected",
                error_message=None,
                tool_count=0,
                resource_count=0,
                prompt_count=0,
            )
            for name in self._connected
        ]


class _FastManager:
    """Fake MCPManager whose ``stop_all`` returns immediately (happy path)."""

    def __init__(self) -> None:
        self.stop_called = False

    async def stop_all(self) -> None:
        self.stop_called = True

    def status(self) -> list[Any]:
        return []


class _RaisingManager:
    """Fake MCPManager whose ``stop_all`` raises an unexpected error."""

    async def stop_all(self) -> None:
        raise RuntimeError("library tore itself in half")

    def status(self) -> list[Any]:
        return []


@pytest.mark.asyncio
async def test_aclose_cancels_hanging_stop_all_within_timeout(
    tmp_path: Path,
) -> None:
    """AC-B3-1: aclose with mcp_timeout=0.1 must return in <0.5s and journal.

    When ``mcp_manager.stop_all`` hangs (simulated by a 60s sleep), the
    Agent must cancel it at the timeout boundary, emit
    ``mcp_close_timeout`` with ``elapsed_sec`` / ``timeout_sec`` /
    ``servers_hanging``, and return control to the caller — no leaked
    tasks, no fire-and-forget.
    """
    log_path = tmp_path / "journal.jsonl"
    journal.configure(log_path)
    try:
        agent = Agent(
            config=_minimal_config(),
            model=FakeChatModel(turns=[]),
            storage=_storage(tmp_path),
            session_id="b3-timeout",
        )
        fake_mgr = _HangingManager(connected_servers=["hang_a", "hang_b"])
        agent._mcp_manager = fake_mgr  # type: ignore[assignment]  # noqa: SLF001

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await agent.aclose(mcp_timeout=0.1)
        elapsed = loop.time() - t0

        # Returned fast: <0.5s budget per the spec.
        assert elapsed < 0.5, (
            f"aclose took {elapsed:.3f}s, budget is 0.5s"
        )
        # The fake manager observed cancellation — proves we went through
        # wait_for(cancel) and not through "fire-and-forget + return".
        assert fake_mgr.stop_called
        assert fake_mgr.cancelled
        # Agent drops the manager reference on timeout so subsequent
        # close() calls are idempotent no-ops.
        assert agent._mcp_manager is None  # noqa: SLF001

        events = _journal_events(log_path)
        timeout_events = [e for e in events if e.get("event") == "mcp_close_timeout"]
        assert len(timeout_events) == 1
        evt = timeout_events[0]
        assert evt["session"] == "b3-timeout"
        assert evt["timeout_sec"] == pytest.approx(0.1)
        assert evt["elapsed_sec"] >= 0.1
        assert evt["elapsed_sec"] < 0.5
        assert sorted(evt["servers_hanging"]) == ["hang_a", "hang_b"]
        # Positive-path event must NOT fire on the timeout path.
        assert not any(e.get("event") == "mcp_stopped" for e in events)
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_aclose_fast_path_emits_mcp_stopped_no_timeout_event(
    tmp_path: Path,
) -> None:
    """AC-B3-2 positive path: non-hanging stop_all returns without timeout event.

    Happy-path regression: a clean ``stop_all`` must NOT produce a
    ``mcp_close_timeout`` event, and should emit ``mcp_stopped`` with
    ``session`` + ``elapsed_sec`` so operators can audit shutdowns.
    """
    log_path = tmp_path / "journal.jsonl"
    journal.configure(log_path)
    try:
        agent = Agent(
            config=_minimal_config(),
            model=FakeChatModel(turns=[]),
            storage=_storage(tmp_path),
            session_id="b3-fast",
        )
        fake_mgr = _FastManager()
        agent._mcp_manager = fake_mgr  # type: ignore[assignment]  # noqa: SLF001

        await agent.aclose(mcp_timeout=5.0)

        assert fake_mgr.stop_called
        assert agent._mcp_manager is None  # noqa: SLF001

        events = _journal_events(log_path)
        stopped = [e for e in events if e.get("event") == "mcp_stopped"]
        assert len(stopped) == 1
        assert stopped[0]["session"] == "b3-fast"
        assert "elapsed_sec" in stopped[0]
        # No timeout / error events on the happy path.
        assert not any(e.get("event") == "mcp_close_timeout" for e in events)
        assert not any(e.get("event") == "mcp_close_error" for e in events)
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_aclose_unexpected_error_emits_mcp_close_error(
    tmp_path: Path,
) -> None:
    """Unexpected ``stop_all`` errors surface as ``mcp_close_error``, not timeout.

    Third journal branch: not-timeout, not-clean. The exception must be
    captured (no re-raise — aclose is shutdown path, must be robust), the
    manager reference must still be dropped so close() is idempotent, and
    the event must carry a readable ``error`` string.
    """
    log_path = tmp_path / "journal.jsonl"
    journal.configure(log_path)
    try:
        agent = Agent(
            config=_minimal_config(),
            model=FakeChatModel(turns=[]),
            storage=_storage(tmp_path),
            session_id="b3-error",
        )
        agent._mcp_manager = _RaisingManager()  # type: ignore[assignment]  # noqa: SLF001

        # Must not propagate — shutdown swallows expected errors.
        await agent.aclose(mcp_timeout=1.0)

        assert agent._mcp_manager is None  # noqa: SLF001
        events = _journal_events(log_path)
        err = [e for e in events if e.get("event") == "mcp_close_error"]
        assert len(err) == 1
        assert err[0]["session"] == "b3-error"
        assert "library tore itself in half" in err[0]["error"]
        assert not any(e.get("event") == "mcp_close_timeout" for e in events)
        assert not any(e.get("event") == "mcp_stopped" for e in events)
    finally:
        journal.reset()


def test_sync_close_no_loop_runs_aclose_via_asyncio_run(tmp_path: Path) -> None:
    """Sync close() compat: no active loop → spawn one, aclose runs to completion.

    The CLI teardown (`aura/cli/__main__.py:441`) still calls the sync
    ``agent.close()`` after ``asyncio.run(_entry())`` has returned. That
    call site has no event loop. ``close()`` must transparently run the
    async shutdown via ``asyncio.run(aclose(...))`` — no fire-and-forget,
    no hang.
    """
    log_path = tmp_path / "journal.jsonl"
    journal.configure(log_path)
    try:
        agent = Agent(
            config=_minimal_config(),
            model=FakeChatModel(turns=[]),
            storage=_storage(tmp_path),
            session_id="b3-sync",
        )
        fake_mgr = _FastManager()
        agent._mcp_manager = fake_mgr  # type: ignore[assignment]  # noqa: SLF001

        agent.close()

        assert fake_mgr.stop_called
        assert agent._mcp_manager is None  # noqa: SLF001
        events = _journal_events(log_path)
        assert any(e.get("event") == "mcp_stopped" for e in events)
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_sync_close_inside_running_loop_with_mcp_raises(tmp_path: Path) -> None:
    """No fire-and-forget: sync close() inside an active loop with live MCP raises.

    The old contract silently scheduled ``stop_all`` as a detached task
    and returned — which leaked the coroutine past process exit. New
    contract: when an MCP manager is live, the sync entry is **not**
    valid from a running-loop caller. Async callers must switch to
    ``await agent.aclose(...)`` explicitly.

    Note: a bare agent with no MCP manager can still ``close()`` inside
    a loop — the async teardown is only required when there's async
    work to bound. See ``test_sync_close_inside_loop_no_mcp_is_sync_cleanup``
    below.
    """
    agent = Agent(
        config=_minimal_config(),
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
        session_id="b3-active-loop",
    )
    agent._mcp_manager = _FastManager()  # type: ignore[assignment]  # noqa: SLF001
    with pytest.raises(RuntimeError, match="aclose"):
        agent.close()


@pytest.mark.asyncio
async def test_sync_close_inside_loop_no_mcp_is_sync_cleanup(tmp_path: Path) -> None:
    """Bare agent (no MCP) inside a running loop: close() does sync cleanup.

    Preserves the pre-B3 no-op contract for unit tests that build a
    minimal Agent + call ``close()`` to release the SQLite handle. We
    don't need the async timeout machinery when there's no MCP manager
    to bound.
    """
    agent = Agent(
        config=_minimal_config(),
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
        session_id="b3-sync-inloop",
    )
    # No _mcp_manager assigned — this is the bare-agent path.
    agent.close()  # must not raise
