"""PaneBackend — tmux pane spawn / shutdown round-trip.

Real-tmux tests are skipped when ``pane_backend_available()`` is False
(the typical CI environment); when developing locally inside tmux the
suite exercises a real ``tmux split-window`` -> ``kill-pane`` cycle.

The unavailable path (``$TMUX`` unset) is covered as well so the
registry's gating is verified regardless of where the suite runs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from aura.core.abort import AbortController
from aura.core.persistence.storage import SessionStorage
from aura.core.teams.backends import detection
from aura.core.teams.backends.pane import PaneBackend, PaneBackendError, PaneHandle
from aura.core.teams.backends.registry import (
    BackendUnavailable,
    _reset_for_tests,
    get_backend,
)
from aura.core.teams.types import TeammateMember


class _LeaderStub:
    """Minimal manager stand-in — pane shutdown only reads ``team`` + ``_storage``."""

    def __init__(self, storage: SessionStorage, team_id: str) -> None:
        from aura.core.teams.types import TeamRecord

        self._storage = storage
        self.team = TeamRecord(
            team_id=team_id,
            name=team_id,
            leader_session_id="leader-1",
        )

    def _post(self, _msg: Any) -> None:
        # PaneHandle.shutdown calls this when posting shutdown_request;
        # the stub is enough for unit-shape coverage.
        pass


def test_pane_registry_unavailable_outside_tmux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry refuses to hand out the pane backend when the env can't run it."""
    monkeypatch.setattr(detection, "is_inside_tmux", lambda: False)
    _reset_for_tests()
    with pytest.raises(BackendUnavailable, match="pane backend unavailable"):
        get_backend("pane")
    _reset_for_tests()


def test_pane_backend_spawn_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Direct PaneBackend instantiation also gates on environment."""
    monkeypatch.setattr(detection, "is_inside_tmux", lambda: False)
    backend = PaneBackend()
    storage = SessionStorage(tmp_path / "sessions.db")
    member = TeammateMember(name="alice", backend_type="pane")
    stop = asyncio.Event()
    abort = AbortController()

    async def _spawn() -> None:
        await backend.spawn(
            team_id="team-a",
            member=member,
            agent=None,  # type: ignore[arg-type]
            manager=None,  # type: ignore[arg-type]
            storage=storage,
            stop_event=stop,
            abort=abort,
            seed_prompt=None,
        )

    with pytest.raises(PaneBackendError, match="requires tmux"):
        asyncio.run(_spawn())


@pytest.mark.skipif(
    not detection.pane_backend_available(),
    reason="requires running inside a tmux session with tmux on PATH",
)
@pytest.mark.asyncio
async def test_pane_backend_spawn_creates_tmux_pane(tmp_path: Path) -> None:
    """When inside tmux, spawn allocates a real pane and stamps the id."""
    backend = PaneBackend()
    storage = SessionStorage(tmp_path / "sessions.db")
    member = TeammateMember(name="alice", backend_type="pane")
    leader = _LeaderStub(storage, "team-a")
    stop = asyncio.Event()
    abort = AbortController()
    handle = await backend.spawn(
        team_id="team-a",
        member=member,
        agent=None,  # type: ignore[arg-type]
        manager=leader,  # type: ignore[arg-type]
        storage=storage,
        stop_event=stop,
        abort=abort,
        seed_prompt=None,
    )
    try:
        assert isinstance(handle, PaneHandle)
        assert handle.pane_id is not None
        assert handle.pane_id.startswith("%")
        assert member.tmux_pane_id == handle.pane_id
    finally:
        await handle.force_kill()


@pytest.mark.skipif(
    not detection.pane_backend_available(),
    reason="requires running inside a tmux session with tmux on PATH",
)
@pytest.mark.asyncio
async def test_pane_backend_force_kill_kills_pane(tmp_path: Path) -> None:
    """force_kill closes the tmux pane via kill-pane."""
    backend = PaneBackend()
    storage = SessionStorage(tmp_path / "sessions.db")
    member = TeammateMember(name="alice", backend_type="pane")
    leader = _LeaderStub(storage, "team-a")
    stop = asyncio.Event()
    abort = AbortController()
    handle = await backend.spawn(
        team_id="team-a",
        member=member,
        agent=None,  # type: ignore[arg-type]
        manager=leader,  # type: ignore[arg-type]
        storage=storage,
        stop_event=stop,
        abort=abort,
        seed_prompt=None,
    )
    assert handle.is_alive()
    await handle.force_kill()
    # Liveness probe re-runs ``tmux list-panes``; after kill-pane the
    # id is no longer present.
    assert handle.is_alive() is False


def test_pane_backend_singleton_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry returns the same PaneBackend instance on repeated calls."""
    monkeypatch.setattr(detection, "is_inside_tmux", lambda: True)
    monkeypatch.setattr(detection, "tmux_available", lambda: True)
    _reset_for_tests()
    a = get_backend("pane")
    b = get_backend("pane")
    assert a is b
    assert a.backend_type == "pane"
    _reset_for_tests()
