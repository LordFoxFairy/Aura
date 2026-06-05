"""PaneBackend — tmux pane spawn / shutdown round-trip.

Real-tmux tests are skipped when ``pane_backend_available()`` is False
(the typical CI environment); when developing locally inside tmux the
suite exercises a real ``tmux split-window`` -> ``kill-pane`` cycle.

The unavailable path (``$TMUX`` unset) is covered as well so the
registry's gating is verified regardless of where the suite runs.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from aura.application.teams.mailbox import Mailbox
from aura.application.teams.team_port import TeamPort
from aura.domain.abort import AbortController
from aura.domain.team import (
    TEAM_LEADER_NAME,
    TeammateMember,
    TeamMessage,
    TeamRecord,
)
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.teams import detection
from aura.infrastructure.teams import pane as pane_module
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


# --- Mocked-seam lifecycle coverage (no real tmux) -------------------------
#
# These tests replace the single shell-out seam (``subprocess.run`` for the
# raw ``_run_tmux`` helper, or the module-level ``_run_tmux`` / ``_pane_alive``
# for the higher-level handle methods) so the full pane lifecycle and every
# error branch run deterministically without a tmux binary or a tmux session.


@dataclass
class _FakeCompleted:
    """Stand-in for ``subprocess.CompletedProcess`` carrying scripted tmux output."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class _TmuxScript:
    """Records every ``tmux`` argv and replies per first subcommand."""

    replies: dict[str, _FakeCompleted]
    calls: list[list[str]] = field(default_factory=list)

    def run(
        self,
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> _FakeCompleted:
        del capture_output, text, timeout, check
        self.calls.append(cmd)
        subcommand = cmd[1] if len(cmd) > 1 else ""
        return self.replies[subcommand]


class _TeamPortStub:
    """TeamPort-conforming leader stub backed by a real on-disk Mailbox."""

    def __init__(self, storage: SessionStorage, team_id: str) -> None:
        self._storage = storage
        self._team = TeamRecord(
            team_id=team_id,
            name=team_id,
            leader_session_id="leader-1",
        )
        self.posted: list[TeamMessage] = []

    @property
    def is_active(self) -> bool:
        return True

    @property
    def team(self) -> TeamRecord | None:
        return self._team

    @property
    def storage(self) -> SessionStorage:
        return self._storage

    def post_message(self, msg: TeamMessage) -> None:
        self.posted.append(msg)

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        body: str,
        kind: str = "text",
    ) -> list[TeamMessage]:
        del sender, recipient, body, kind
        return []

    def confirm_shutdown(self, member_name: str, *, body: str = "") -> None:
        del member_name, body

    @property
    def pending_protocol_events(self) -> tuple[Any, ...]:
        return ()

    def drain_protocol_events(self) -> list[Any]:
        return []

    async def cleanup_session_teams(self) -> None:
        return None


def _make_handle(
    manager: TeamPort,
    *,
    pane_id: str | None,
    member_name: str = "alice",
) -> tuple[PaneHandle, asyncio.Event, AbortController]:
    stop = asyncio.Event()
    abort = AbortController()
    handle = PaneHandle(
        pane_id=pane_id,
        member_name=member_name,
        manager=manager,
        stop_event=stop,
        abort=abort,
    )
    return handle, stop, abort


def _silence_journal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop real journal I/O — these tests assert behavior, not forensics files."""

    def _noop(_event: str, **_fields: Any) -> None:
        return None

    monkeypatch.setattr("aura.infrastructure.teams.pane.journal.write", _noop)


def _recording_runner(sink: list[list[str]]) -> Any:
    """Build a ``_run_tmux`` replacement that records argv and returns empty stdout."""

    def _run(args: list[str]) -> str:
        sink.append(args)
        return ""

    return _run


# --- _run_tmux raw seam: error + success branches --------------------------


def test_run_tmux_missing_binary_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A box without tmux must fail loudly, not silently degrade."""

    def _raise(*_args: Any, **_kwargs: Any) -> _FakeCompleted:
        raise FileNotFoundError("tmux")

    monkeypatch.setattr("aura.infrastructure.teams.pane.subprocess.run", _raise)
    with pytest.raises(PaneBackendError, match="tmux binary not found"):
        pane_module._run_tmux(["list-panes"])


def test_run_tmux_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung tmux must surface as a bounded error, never block the leader forever."""

    def _raise(*_args: Any, **_kwargs: Any) -> _FakeCompleted:
        raise subprocess.TimeoutExpired(cmd="tmux", timeout=5.0)

    monkeypatch.setattr("aura.infrastructure.teams.pane.subprocess.run", _raise)
    with pytest.raises(PaneBackendError, match="timed out after"):
        pane_module._run_tmux(["kill-pane", "-t", "%1"])


def test_run_tmux_nonzero_returncode_reports_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing tmux command must propagate rc and stderr for diagnosis."""
    script = _TmuxScript(
        replies={"kill-pane": _FakeCompleted(returncode=1, stderr="no such pane")},
    )
    monkeypatch.setattr("aura.infrastructure.teams.pane.subprocess.run", script.run)
    with pytest.raises(PaneBackendError, match=r"(?s)rc=1.*no such pane"):
        pane_module._run_tmux(["kill-pane", "-t", "%9"])


def test_run_tmux_success_strips_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returned pane ids must be whitespace-stripped so downstream matching works."""
    script = _TmuxScript(
        replies={"split-window": _FakeCompleted(returncode=0, stdout="  %7 \n")},
    )
    monkeypatch.setattr("aura.infrastructure.teams.pane.subprocess.run", script.run)
    assert pane_module._run_tmux(["split-window"]) == "%7"


# --- _pane_alive: present / absent / tmux-error ----------------------------


@pytest.mark.parametrize(
    ("listing", "pane_id", "expected"),
    [
        ("%1\n%2\n%3", "%2", True),
        ("%1\n%3", "%2", False),
        ("", "%2", False),
    ],
)
def test_pane_alive_membership(
    monkeypatch: pytest.MonkeyPatch,
    listing: str,
    pane_id: str,
    expected: bool,
) -> None:
    """Liveness is exact membership in tmux's pane listing, not a substring guess."""
    monkeypatch.setattr(pane_module, "_run_tmux", lambda _args: listing)
    assert pane_module._pane_alive(pane_id) is expected


def test_pane_alive_treats_tmux_error_as_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If tmux itself errors while probing, the pane is presumed gone, not alive."""

    def _boom(_args: list[str]) -> str:
        raise PaneBackendError("tmux failed")

    monkeypatch.setattr(pane_module, "_run_tmux", _boom)
    assert pane_module._pane_alive("%2") is False


# --- is_alive / _kill_pane / force_kill ------------------------------------


def test_is_alive_none_pane_is_false() -> None:
    """A handle that never got a pane id is never alive — no tmux call attempted."""
    manager: TeamPort = _TeamPortStub(SessionStorage(Path(":memory:")), "team-a")
    handle, _stop, _abort = _make_handle(manager, pane_id=None)
    assert handle.is_alive() is False


def test_is_alive_delegates_to_pane_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live pane id probes tmux and reports True."""
    manager: TeamPort = _TeamPortStub(SessionStorage(Path(":memory:")), "team-a")
    handle, _stop, _abort = _make_handle(manager, pane_id="%5")
    monkeypatch.setattr(pane_module, "_pane_alive", lambda pid: pid == "%5")
    assert handle.is_alive() is True


async def test_kill_pane_none_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Killing a handle with no pane id must not shell out at all."""
    manager: TeamPort = _TeamPortStub(SessionStorage(Path(":memory:")), "team-a")
    handle, _stop, _abort = _make_handle(manager, pane_id=None)

    def _forbidden(_args: list[str]) -> str:
        raise AssertionError("tmux must not run for a None pane")

    monkeypatch.setattr(pane_module, "_run_tmux", _forbidden)
    await handle.force_kill()  # must complete without raising


async def test_force_kill_aborts_and_kills(monkeypatch: pytest.MonkeyPatch) -> None:
    """force_kill signals abort once and issues kill-pane on the correct target."""
    manager: TeamPort = _TeamPortStub(SessionStorage(Path(":memory:")), "team-a")
    handle, _stop, abort = _make_handle(manager, pane_id="%3")
    killed: list[list[str]] = []
    monkeypatch.setattr(pane_module, "_run_tmux", _recording_runner(killed))
    await handle.force_kill()
    assert abort.aborted is True
    assert killed == [["kill-pane", "-t", "%3"]]


async def test_force_kill_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling force_kill twice must not double-abort nor crash (idempotency edge)."""
    manager: TeamPort = _TeamPortStub(SessionStorage(Path(":memory:")), "team-a")
    handle, _stop, abort = _make_handle(manager, pane_id="%3")
    calls: list[list[str]] = []
    monkeypatch.setattr(pane_module, "_run_tmux", _recording_runner(calls))
    await handle.force_kill()
    abort_reason_after_first = abort.aborted
    await handle.force_kill()
    assert abort_reason_after_first is True
    # Both calls still issue kill-pane; only the abort is guarded against repeat.
    assert calls == [["kill-pane", "-t", "%3"], ["kill-pane", "-t", "%3"]]


async def test_kill_pane_swallows_tmux_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pane already gone is a success state — kill must not raise, only journal."""
    manager: TeamPort = _TeamPortStub(SessionStorage(Path(":memory:")), "team-a")
    handle, _stop, _abort = _make_handle(manager, pane_id="%3")
    journaled: list[dict[str, Any]] = []

    def _capture(event: str, **fields: Any) -> None:
        journaled.append({"event": event, **fields})

    def _boom(_args: list[str]) -> str:
        raise PaneBackendError("can't find pane %3")

    monkeypatch.setattr("aura.infrastructure.teams.pane.journal.write", _capture)
    monkeypatch.setattr(pane_module, "_run_tmux", _boom)
    await handle.force_kill()  # no exception escapes
    assert any(j["event"] == "team_pane_kill_error" for j in journaled)


# --- shutdown: ack / timeout / dead-pane / missing-team --------------------


async def test_shutdown_dead_pane_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pane that's already dead is a clean stop — return True, never post/kill."""
    manager: TeamPort = _TeamPortStub(SessionStorage(Path(":memory:")), "team-a")
    handle, _stop, _abort = _make_handle(manager, pane_id="%3")
    monkeypatch.setattr(pane_module, "_pane_alive", lambda _pid: False)
    assert await handle.shutdown() is True


async def test_shutdown_none_pane_returns_true() -> None:
    """No pane id means nothing to stop — trivially clean."""
    manager: TeamPort = _TeamPortStub(SessionStorage(Path(":memory:")), "team-a")
    handle, _stop, _abort = _make_handle(manager, pane_id=None)
    assert await handle.shutdown() is True


async def test_shutdown_missing_team_force_kills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the leader has no team record we can't mailbox-ack — force kill, return False."""

    class _NoTeamStub(_TeamPortStub):
        @property
        def team(self) -> TeamRecord | None:
            return None

    manager: TeamPort = _NoTeamStub(SessionStorage(Path(":memory:")), "team-a")
    handle, _stop, abort = _make_handle(manager, pane_id="%3")
    _silence_journal(monkeypatch)
    monkeypatch.setattr(pane_module, "_pane_alive", lambda _pid: True)
    killed: list[list[str]] = []
    monkeypatch.setattr(pane_module, "_run_tmux", _recording_runner(killed))
    assert await handle.shutdown() is False
    assert abort.aborted is True
    assert ["kill-pane", "-t", "%3"] in killed


async def test_shutdown_acks_when_response_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A fresh shutdown_response from the member yields a clean True, then kills."""
    storage = SessionStorage(tmp_path / "sessions.db")

    # The teammate "replies" the instant the leader posts the request — this fires
    # AFTER the baseline snapshot, so the ack is genuinely fresh.
    class _AckingStub(_TeamPortStub):
        def post_message(self, msg: TeamMessage) -> None:
            super().post_message(msg)
            Mailbox(storage, "team-a").append(
                TeamMessage(
                    msg_id="ack-1",
                    sender="alice",
                    recipient=TEAM_LEADER_NAME,
                    body="done",
                    kind="shutdown_response",
                ),
            )

    stub = _AckingStub(storage, "team-a")
    manager: TeamPort = stub
    handle, stop, _abort = _make_handle(manager, pane_id="%3", member_name="alice")
    _silence_journal(monkeypatch)
    monkeypatch.setattr(pane_module, "_pane_alive", lambda _pid: True)
    killed: list[list[str]] = []
    monkeypatch.setattr(pane_module, "_run_tmux", _recording_runner(killed))

    acked = await handle.shutdown(timeout_sec=1.0)
    assert acked is True
    assert stop.is_set() is True
    assert any(m.kind == "shutdown_request" for m in stub.posted)
    assert ["kill-pane", "-t", "%3"] in killed


async def test_shutdown_ignores_stale_baseline_ack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A pre-existing ack (in baseline) must NOT false-positive a new shutdown."""
    storage = SessionStorage(tmp_path / "sessions.db")
    manager: TeamPort = _TeamPortStub(storage, "team-a")
    handle, _stop, _abort = _make_handle(manager, pane_id="%3", member_name="alice")
    _silence_journal(monkeypatch)
    monkeypatch.setattr(pane_module, "_pane_alive", lambda _pid: True)
    monkeypatch.setattr(pane_module, "_run_tmux", lambda _args: "")

    # Stale ack already on disk before shutdown starts -> snapshotted into baseline.
    mailbox = Mailbox(storage, "team-a")
    mailbox.append(
        TeamMessage(
            msg_id="stale-ack",
            sender="alice",
            recipient=TEAM_LEADER_NAME,
            body="old",
            kind="shutdown_response",
        ),
    )
    acked = await handle.shutdown(timeout_sec=0.1)
    assert acked is False


async def test_shutdown_times_out_without_ack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No member response within the budget -> False, but pane still gets killed."""
    storage = SessionStorage(tmp_path / "sessions.db")
    manager: TeamPort = _TeamPortStub(storage, "team-a")
    handle, _stop, _abort = _make_handle(manager, pane_id="%3", member_name="alice")
    _silence_journal(monkeypatch)
    monkeypatch.setattr(pane_module, "_pane_alive", lambda _pid: True)
    killed: list[list[str]] = []
    monkeypatch.setattr(pane_module, "_run_tmux", _recording_runner(killed))
    acked = await handle.shutdown(timeout_sec=0.05)
    assert acked is False
    assert ["kill-pane", "-t", "%3"] in killed


async def test_shutdown_ignores_wrong_kind_and_sender(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only a shutdown_response from the matching member counts as an ack."""
    storage = SessionStorage(tmp_path / "sessions.db")
    manager: TeamPort = _TeamPortStub(storage, "team-a")
    handle, _stop, _abort = _make_handle(manager, pane_id="%3", member_name="alice")
    _silence_journal(monkeypatch)
    monkeypatch.setattr(pane_module, "_pane_alive", lambda _pid: True)
    monkeypatch.setattr(pane_module, "_run_tmux", lambda _args: "")

    mailbox = Mailbox(storage, "team-a")
    # Right sender, wrong kind.
    mailbox.append(
        TeamMessage(
            msg_id="m1",
            sender="alice",
            recipient=TEAM_LEADER_NAME,
            body="chat",
            kind="text",
        ),
    )
    # Right kind, wrong sender.
    mailbox.append(
        TeamMessage(
            msg_id="m2",
            sender="bob",
            recipient=TEAM_LEADER_NAME,
            body="done",
            kind="shutdown_response",
        ),
    )
    acked = await handle.shutdown(timeout_sec=0.1)
    assert acked is False


# --- spawn happy path + command construction -------------------------------


async def test_spawn_creates_pane_and_stamps_member(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """spawn splits a pane, persists its id on the member, and sends the launch keys."""
    storage = SessionStorage(tmp_path / "sessions.db")
    stub = _TeamPortStub(storage, "team-a")
    manager: TeamPort = stub
    member = TeammateMember(name="alice", backend_type="pane")
    backend = PaneBackend()
    _silence_journal(monkeypatch)
    monkeypatch.setattr(pane_module, "pane_backend_available", lambda: True)
    calls: list[list[str]] = []

    def _fake_run(args: list[str]) -> str:
        calls.append(args)
        return "%11" if args[0] == "split-window" else ""

    monkeypatch.setattr(pane_module, "_run_tmux", _fake_run)
    agent: Any = None
    handle = await backend.spawn(
        team_id="team-a",
        member=member,
        agent=agent,
        manager=manager,
        storage=storage,
        stop_event=asyncio.Event(),
        abort=AbortController(),
        seed_prompt=None,
    )
    assert isinstance(handle, PaneHandle)
    assert handle.pane_id == "%11"
    assert member.tmux_pane_id == "%11"
    assert calls[0][0] == "split-window"
    send = next(c for c in calls if c[0] == "send-keys")
    assert send[1:3] == ["-t", "%11"]


async def test_spawn_raises_when_split_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A blank pane id from tmux is a hard failure — never proceed to send-keys."""
    storage = SessionStorage(tmp_path / "sessions.db")
    manager: TeamPort = _TeamPortStub(storage, "team-a")
    member = TeammateMember(name="alice", backend_type="pane")
    backend = PaneBackend()
    _silence_journal(monkeypatch)
    monkeypatch.setattr(pane_module, "pane_backend_available", lambda: True)
    sent: list[list[str]] = []

    def _fake_run(args: list[str]) -> str:
        if args[0] == "send-keys":
            sent.append(args)
        return ""  # split-window yields empty -> error before send-keys

    monkeypatch.setattr(pane_module, "_run_tmux", _fake_run)
    agent: Any = None
    with pytest.raises(PaneBackendError, match="no pane_id"):
        await backend.spawn(
            team_id="team-a",
            member=member,
            agent=agent,
            manager=manager,
            storage=storage,
            stop_event=asyncio.Event(),
            abort=AbortController(),
            seed_prompt=None,
        )
    assert sent == []


def test_build_command_includes_optional_flags() -> None:
    """Optional model / system-prompt / seed are appended only when truthy."""
    storage = SessionStorage(Path("/tmp/aura-test/sessions.db"))
    member = TeammateMember(
        name="alice",
        backend_type="pane",
        agent_type="general-purpose",
        model_name="opus",
        system_prompt="be terse",
    )
    argv = PaneBackend._build_subprocess_command(
        team_id="team-a",
        member=member,
        storage=storage,
        seed_prompt="kick off",
    )
    assert argv[0] == sys.executable
    assert argv[1:4] == ["-m", "cli", "teammate"]
    assert "--model" in argv and argv[argv.index("--model") + 1] == "opus"
    assert "--system-prompt" in argv
    assert argv[argv.index("--seed-prompt") + 1] == "kick off"
    assert "--storage-root" in argv


def test_build_command_omits_optional_and_blank_seed() -> None:
    """Missing model/prompt and a whitespace-only seed must add no stray flags."""
    storage = SessionStorage(Path("/tmp/aura-test/sessions.db"))
    member = TeammateMember(name="alice", backend_type="pane")
    argv = PaneBackend._build_subprocess_command(
        team_id="team-a",
        member=member,
        storage=storage,
        seed_prompt="   ",
    )
    assert "--model" not in argv
    assert "--system-prompt" not in argv
    assert "--seed-prompt" not in argv


def test_resolve_storage_root_memory_falls_back_to_home() -> None:
    """An in-memory DB has no parent dir — subprocess storage root falls back to ~/.aura."""
    storage = SessionStorage(Path(":memory:"))
    root = pane_module._resolve_storage_root(storage)
    assert root.endswith("/.aura")


def test_resolve_storage_root_uses_parent_dir() -> None:
    """A file-backed DB hands the subprocess its containing directory."""
    storage = SessionStorage(Path("/tmp/aura-test/nested/sessions.db"))
    root = pane_module._resolve_storage_root(storage)
    assert root == "/tmp/aura-test/nested"
