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

from aura.core.abort import AbortController
from aura.core.persistence.storage import SessionStorage
from aura.core.teams.backends.in_process import InProcessBackend, InProcessHandle
from aura.core.teams.backends.registry import get_backend
from aura.core.teams.mailbox import Mailbox
from aura.core.teams.types import TEAM_LEADER_NAME, TeammateMember, TeamMessage
from aura.schemas.events import Final


class _ScriptedAgent:
    """Minimal Agent stand-in — yields a Final per astream call."""

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
    handle = await backend.spawn(
        team_id="team-a",
        member=member,
        agent=_ScriptedAgent(),  # type: ignore[arg-type]
        manager=None,  # type: ignore[arg-type]  # unused by in-process
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
    handle = await backend.spawn(
        team_id="team-a",
        member=member,
        agent=_ScriptedAgent(),  # type: ignore[arg-type]
        manager=None,  # type: ignore[arg-type]
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
    handle = await backend.spawn(
        team_id="team-a",
        member=member,
        agent=_ScriptedAgent(),  # type: ignore[arg-type]
        manager=None,  # type: ignore[arg-type]
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
    from aura.core.teams.types import TeamRecord

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
    from aura.core.teams.types import TeamRecord

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
