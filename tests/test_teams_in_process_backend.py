"""InProcessBackend — spawn / shutdown round-trip + handle invariants.

The in-process backend is the default for every teammate; this suite
verifies its handle's :meth:`shutdown`, :meth:`force_kill`, and
:meth:`is_alive` semantics against the real ``run_teammate`` loop.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest

from aura.application.teams.mailbox import Mailbox
from aura.domain.abort import AbortController
from aura.domain.events import Final
from aura.domain.team import TEAM_LEADER_NAME, TeammateMember, TeamMessage
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.teams.in_process import InProcessBackend, InProcessHandle
from aura.infrastructure.teams.registry import get_backend


class _ScriptedAgent:
    """Minimal AgentSession stand-in — yields a Final per astream call."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = replies or ["ack"]
        self._idx = 0

    async def astream(self, prompt: str, *, abort: Any = None) -> Any:
        msg = self.replies[min(self._idx, len(self.replies) - 1)]
        self._idx += 1
        yield Final(message=msg, reason="natural")


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "sessions.db")


@pytest.mark.asyncio
async def test_in_process_backend_spawn_returns_handle(tmp_path: Path) -> None:
    """spawn returns an InProcessHandle that exposes the running task."""
    storage = _storage(tmp_path)
    backend = InProcessBackend()
    member = TeammateMember(name="alice")
    stop = asyncio.Event()
    abort = AbortController()
    _agent: Any = _ScriptedAgent()
    _manager: Any = None
    handle = await backend.spawn(
        team_id="team-a",
        member=member,
        agent=_agent,
        manager=_manager,
        storage=storage,
        stop_event=stop,
        abort=abort,
        seed_prompt=None,
    )
    try:
        assert isinstance(handle, InProcessHandle)
        assert handle.pane_id is None
        assert handle.is_alive()
    finally:
        await handle.force_kill()


@pytest.mark.asyncio
async def test_in_process_backend_shutdown_round_trip(tmp_path: Path) -> None:
    """shutdown(stop_event) lets run_teammate exit at the next poll boundary."""
    storage = _storage(tmp_path)
    backend = InProcessBackend()
    member = TeammateMember(name="alice")
    stop = asyncio.Event()
    abort = AbortController()
    # Send a text message + queue a shutdown_request so the runtime
    # acks naturally; the backend's shutdown helper short-circuits via
    # stop_event.
    box = Mailbox(storage, "team-a")
    box.append(TeamMessage(
        msg_id=uuid.uuid4().hex,
        sender=TEAM_LEADER_NAME,
        recipient="alice",
        body="hi",
        kind="text",
    ))
    _agent: Any = _ScriptedAgent()
    _manager: Any = None
    handle = await backend.spawn(
        team_id="team-a",
        member=member,
        agent=_agent,
        manager=_manager,
        storage=storage,
        stop_event=stop,
        abort=abort,
        seed_prompt=None,
    )
    # Give the runtime a tick to consume the queued message.
    await asyncio.sleep(0.3)
    ok = await handle.shutdown(timeout_sec=10.0)
    assert ok is True, "graceful shutdown should succeed within timeout"
    assert handle.is_alive() is False


@pytest.mark.asyncio
async def test_in_process_backend_force_kill_idempotent(tmp_path: Path) -> None:
    """force_kill is idempotent and safe on already-dead handles."""
    storage = _storage(tmp_path)
    backend = InProcessBackend()
    member = TeammateMember(name="alice")
    stop = asyncio.Event()
    abort = AbortController()
    _agent: Any = _ScriptedAgent()
    _manager: Any = None
    handle = await backend.spawn(
        team_id="team-a",
        member=member,
        agent=_agent,
        manager=_manager,
        storage=storage,
        stop_event=stop,
        abort=abort,
        seed_prompt=None,
    )
    await handle.force_kill()
    assert handle.is_alive() is False
    # Second call should be a no-op, not raise.
    await handle.force_kill()


def test_registry_returns_singleton_per_type() -> None:
    """get_backend('in_process') returns the same instance on repeated calls."""
    a = get_backend("in_process")
    b = get_backend("in_process")
    assert a is b
    assert a.backend_type == "in_process"


def test_member_backend_type_round_trips_via_config_json() -> None:
    """``backend_type`` + ``tmux_pane_id`` round-trip through Pydantic JSON."""
    from aura.domain.team import TeamRecord

    member = TeammateMember(
        name="alice",
        backend_type="pane",
        tmux_pane_id="%42",
    )
    record = TeamRecord(
        team_id="team-a",
        name="alpha",
        leader_session_id="leader-1",
        members=[member],
    )
    raw = record.model_dump_json()
    restored = TeamRecord.model_validate_json(raw)
    assert restored.members[0].name == "alice"
    assert restored.members[0].backend_type == "pane"
    assert restored.members[0].tmux_pane_id == "%42"


def test_member_default_backend_type_is_in_process() -> None:
    """Existing config.json files (no backend_type field) load as in_process."""
    from aura.domain.team import TeamRecord

    legacy_json = """{
        "team_id": "team-a",
        "name": "alpha",
        "leader_session_id": "leader-1",
        "members": [{"name": "alice"}],
        "created_at": 1700000000.0,
        "cwd": "."
    }"""
    record = TeamRecord.model_validate_json(legacy_json)
    assert record.members[0].backend_type == "in_process"
    assert record.members[0].tmux_pane_id is None


async def _gate_forever(gate: asyncio.Event) -> None:
    """Block until the gate is set; lets a test pin a task in the running state."""
    await gate.wait()


def _handle_for(
    task: asyncio.Task[None],
    *,
    abort: AbortController | None = None,
) -> InProcessHandle:
    """Build a handle around a caller-controlled task, isolating handle semantics."""
    return InProcessHandle(
        task=task,
        stop_event=asyncio.Event(),
        abort=abort or AbortController(),
    )


@pytest.mark.asyncio
async def test_shutdown_short_circuits_when_task_already_done() -> None:
    """A handle whose task already exited must report graceful success without re-signalling."""
    done = asyncio.Event()
    done.set()
    task = asyncio.create_task(_gate_forever(done))
    await task
    handle = _handle_for(task)
    assert handle.is_alive() is False
    ok = await handle.shutdown(timeout_sec=10.0)
    assert ok is True
    # Early return must not flip the cooperative stop signal.
    assert handle.stop_event.is_set() is False


@pytest.mark.asyncio
async def test_shutdown_force_kills_and_returns_false_on_timeout() -> None:
    """A teammate that ignores the stop signal past the deadline is force-killed, yielding False."""
    gate = asyncio.Event()
    task = asyncio.create_task(_gate_forever(gate))
    await asyncio.sleep(0)
    handle = _handle_for(task)
    ok = await handle.shutdown(timeout_sec=0.05)
    assert ok is False
    assert handle.stop_event.is_set() is True
    assert handle.is_alive() is False
    assert handle.abort.aborted is True


@pytest.mark.asyncio
async def test_shutdown_absorbs_caller_cancel_and_shields_task() -> None:
    """A cancelled shutdown caller must not tear down the teammate; the shield keeps it alive."""
    gate = asyncio.Event()
    task = asyncio.create_task(_gate_forever(gate))
    await asyncio.sleep(0)
    handle = _handle_for(task)
    waiter = asyncio.ensure_future(handle.shutdown(timeout_sec=10.0))
    await asyncio.sleep(0)
    waiter.cancel()
    settled = await waiter
    # The cancel is absorbed; shutdown reports the still-running task as not-done.
    assert settled is False
    assert handle.is_alive() is True
    gate.set()
    await task


@pytest.mark.asyncio
async def test_force_kill_skips_abort_when_already_aborted() -> None:
    """A pre-aborted handle must not re-abort; force_kill still tears the task down idempotently."""
    abort = AbortController()
    abort.abort("prior_reason")
    gate = asyncio.Event()
    task = asyncio.create_task(_gate_forever(gate))
    await asyncio.sleep(0)
    handle = _handle_for(task, abort=abort)
    await handle.force_kill()
    assert handle.is_alive() is False
    # The original abort reason is preserved (idempotent controller).
    assert handle.abort.reason == "prior_reason"


class _RaisingAbort(AbortController):
    """Abort controller whose abort() raises, exercising force_kill's defensive suppression."""

    def abort(self, reason: str = "aborted") -> None:
        raise RuntimeError("abort backend exploded")


@pytest.mark.asyncio
async def test_force_kill_swallows_abort_failure() -> None:
    """A failing abort backend must never crash tear-down; the task is still cancelled."""
    abort = _RaisingAbort()
    gate = asyncio.Event()
    task = asyncio.create_task(_gate_forever(gate))
    await asyncio.sleep(0)
    handle = _handle_for(task, abort=abort)
    await handle.force_kill()
    assert handle.is_alive() is False


@pytest.mark.asyncio
async def test_force_kill_on_finished_task_skips_cancel() -> None:
    """force_kill on a naturally-finished task aborts the signal but issues no redundant cancel."""
    gate = asyncio.Event()
    gate.set()
    task = asyncio.create_task(_gate_forever(gate))
    await task
    abort = AbortController()
    handle = _handle_for(task, abort=abort)
    await handle.force_kill()
    assert handle.abort.aborted is True
    assert handle.is_alive() is False


@pytest.mark.asyncio
async def test_spawn_sync_matches_async_spawn(tmp_path: Path) -> None:
    """spawn and spawn_sync are the same construction path; the async wrapper adds no I/O."""
    storage = _storage(tmp_path)
    backend = InProcessBackend()
    member = TeammateMember(name="bob")
    stop = asyncio.Event()
    abort = AbortController()
    _agent: Any = _ScriptedAgent()
    _manager: Any = None
    handle = backend.spawn_sync(
        team_id="team-b",
        member=member,
        agent=_agent,
        manager=_manager,
        storage=storage,
        stop_event=stop,
        abort=abort,
        seed_prompt=None,
    )
    try:
        assert isinstance(handle, InProcessHandle)
        assert handle.pane_id is None
        assert handle.stop_event is stop
        assert handle.abort is abort
        assert handle.task.get_name() == "aura-teammate-bob"
    finally:
        await handle.force_kill()
