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

from aura.domain.abort import AbortController
from aura.domain.team import TeammateMember
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.teams import detection
from aura.infrastructure.teams.pane import PaneBackend, PaneBackendError, PaneHandle
from aura.infrastructure.teams.registry import BackendUnavailable, get_backend


class _LeaderStub:
    """Minimal manager stand-in — pane shutdown only reads ``team`` + ``_storage``."""

    def __init__(self, storage: SessionStorage, team_id: str) -> None:
        from aura.domain.team import TeamRecord

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
    reset_teams_registry: None,
) -> None:
    """Registry refuses to hand out the pane backend when the env can't run it."""
    monkeypatch.setattr(detection, "is_inside_tmux", lambda: False)
    with pytest.raises(BackendUnavailable, match="pane backend unavailable"):
        get_backend("pane")


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

    _agent: Any = None
    _manager: Any = None

    async def _spawn() -> None:
        await backend.spawn(
            team_id="team-a",
            member=member,
            agent=_agent,
            manager=_manager,
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
    _agent: Any = None
    _leader: Any = leader
    handle = await backend.spawn(
        team_id="team-a",
        member=member,
        agent=_agent,
        manager=_leader,
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
    _agent: Any = None
    _leader: Any = leader
    handle = await backend.spawn(
        team_id="team-a",
        member=member,
        agent=_agent,
        manager=_leader,
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
    reset_teams_registry: None,
) -> None:
    """Registry returns the same PaneBackend instance on repeated calls."""
    monkeypatch.setattr(detection, "is_inside_tmux", lambda: True)
    monkeypatch.setattr(detection, "tmux_available", lambda: True)
    a = get_backend("pane")
    b = get_backend("pane")
    assert a is b
    assert a.backend_type == "pane"
