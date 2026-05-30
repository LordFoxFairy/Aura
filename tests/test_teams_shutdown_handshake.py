"""TEAMS Phase A.1 — shutdown_response round-trip + force-kill fallback.

Pins the four-step contract:

1. The runtime emits a real ``shutdown_response`` to the leader's mailbox
   when it consumes a ``shutdown_request``.
2. ``aremove_member`` accepts that response as the graceful ack — no
   force-kill, no abort.
3. ``aremove_member`` falls back to abort + cancel (force-kill) when no
   response arrives within ``timeout_sec``.
4. The response carries the teammate's name in ``sender`` so the leader
   can correlate ack-to-request even with multiple in-flight removes.

Plus an idempotence check: a second ``aremove_member`` on the same name
raises a clear ``TeamError`` rather than silently no-op'ing.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from aura.application.tasks.factory import SubagentFactory
from aura.application.tasks.store import TasksStore
from aura.application.teams.mailbox import Mailbox
from aura.application.teams.manager import TeamError, TeamManager
from aura.application.teams.runtime import run_teammate
from aura.config.schema import AuraConfig
from aura.domain.abort import AbortController
from aura.domain.events import Final
from aura.domain.permission.safety import DEFAULT_SAFETY
from aura.domain.permission.session import RuleSet
from aura.domain.team import TEAM_LEADER_NAME, TeamMessage
from aura.infrastructure.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _cfg() -> AuraConfig:
    # ``teams.enabled=True`` from v0.18 — see test_teams_manager.py for
    # the rationale (the gate would otherwise make spawned Agent.join_team
    # raise inside SubagentFactory.spawn).
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "teams": {"enabled": True},
    })


def _factory() -> SubagentFactory:
    return SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        parent_ruleset=RuleSet(),
        parent_safety=DEFAULT_SAFETY,
        parent_mode_provider=lambda: "default",
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="ack"))],
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )


def _leader_stub(storage: SessionStorage) -> Any:
    leader = MagicMock()
    leader.session_id = "leader-1"
    leader.cwd = Path.cwd()
    leader._storage = storage
    leader.join_team = MagicMock()
    leader.leave_team = MagicMock()
    return leader


async def _noop_runner(**_kwargs: Any) -> None:
    """Coroutine stand-in for ``runtime_runner`` when the test drives
    ``run_teammate`` directly — does nothing and returns immediately."""
    return None


async def _silent_runtime(**_kwargs: Any) -> None:
    """Runtime stand-in that ignores shutdown_request — never acks.

    Used by the timeout/force-kill test. Waits on its abort signal
    until the leader fires it; returns cleanly when cancelled.
    """
    abort: AbortController = _kwargs["abort"]
    try:
        await abort.signal.wait()
    except asyncio.CancelledError:
        return


class _ScriptedAgent:
    """Real-Agent surrogate for run_teammate.

    Carries a ``team`` attribute so the runtime can call
    ``agent.team.send(...)`` for the shutdown_response leg.
    """

    def __init__(self, team: TeamManager) -> None:
        self.team = team
        self._team_member_name: str | None = None

    async def astream(self, prompt: str, *, abort: Any = None) -> Any:
        yield Final(message="ack", reason="natural")


@pytest.mark.asyncio
async def test_shutdown_request_triggers_response(tmp_path: Path) -> None:
    """Runtime consuming a shutdown_request writes a shutdown_response back.

    Drives ``run_teammate`` directly (no manager-spawned task) so the
    test pins the runtime's contract in isolation: input is one
    shutdown_request line; output is one shutdown_response line in the
    leader's inbox with the teammate as ``sender``.
    """
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader_stub(storage)
    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=_noop_runner,  # unused — we drive run_teammate directly
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    leader.team = mgr
    box = Mailbox(storage, mgr.team.team_id)  # type: ignore[union-attr]  # narrowed by assert above; mypy keeps union
    # Pre-load alice's inbox with a shutdown_request so the runtime
    # consumes it on its first poll.
    box.append(TeamMessage(
        msg_id=uuid.uuid4().hex,
        sender=TEAM_LEADER_NAME,
        recipient="alice",
        body="please stop",
        kind="shutdown_request",
    ))
    agent = _ScriptedAgent(team=mgr)
    abort = AbortController()
    stop = asyncio.Event()
    await asyncio.wait_for(
        run_teammate(
            agent=agent,  # type: ignore[arg-type]  # deliberately off-type arg to exercise path
            team_id=mgr.team.team_id,  # type: ignore[union-attr]  # narrowed by assert above; mypy keeps union
            member_name="alice",
            storage=storage,
            stop_event=stop,
            abort=abort,
        ),
        timeout=10,
    )
    leader_inbox = box.read_all(TEAM_LEADER_NAME)
    responses = [
        m for m in leader_inbox
        if m.kind == "shutdown_response" and m.sender == "alice"
    ]
    assert len(responses) == 1
    assert "please stop" in responses[0].body
    assert responses[0].recipient == TEAM_LEADER_NAME


@pytest.mark.asyncio
async def test_remove_member_accepts_response_as_ack(tmp_path: Path) -> None:
    """``aremove_member`` returns True (ack) when the runtime confirms in time.

    The runtime stand-in ACKs by calling ``manager.confirm_shutdown``
    (the in-process path) and writing a ``shutdown_response`` to the
    leader inbox (parity with the real runtime). ``aremove_member`` must
    observe the future resolution and skip the force-kill path entirely.
    """
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader_stub(storage)

    captured: dict[str, AbortController] = {}

    async def acking_runtime(**kwargs: Any) -> None:
        captured["abort"] = kwargs["abort"]
        team_id = kwargs["team_id"]
        member_name = kwargs["member_name"]
        # Wait for the stop_event (set by aremove_member after sending
        # the shutdown_request) — then ack and exit.
        try:
            await asyncio.wait_for(kwargs["stop_event"].wait(), timeout=10)
        except TimeoutError:
            return
        Mailbox(kwargs["storage"], team_id).append(TeamMessage(
            msg_id=uuid.uuid4().hex,
            sender=member_name,
            recipient=TEAM_LEADER_NAME,
            body=f"shutting down: ack from {member_name}",
            kind="shutdown_response",
        ))
        kwargs["agent"].team.confirm_shutdown(member_name)

    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=acking_runtime,
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    await asyncio.sleep(0)
    acked = await mgr.aremove_member("alice", timeout_sec=2.0)
    assert acked is True
    # No force-kill: abort controller stayed clean.
    assert captured["abort"].aborted is False
    # Membership is gone.
    assert all(m.name != "alice" for m in mgr.list_members())
    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.status == "cancelled"


@pytest.mark.asyncio
async def test_graceful_remove_marks_cancelled_only_after_ack(
    tmp_path: Path,
) -> None:
    """The TaskRecord should not become terminal during the ack window."""
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader_stub(storage)
    ack_now = asyncio.Event()
    stop_seen = asyncio.Event()

    async def delayed_ack_runtime(**kwargs: Any) -> None:
        await kwargs["stop_event"].wait()
        stop_seen.set()
        await ack_now.wait()
        Mailbox(kwargs["storage"], kwargs["team_id"]).append(TeamMessage(
            msg_id=uuid.uuid4().hex,
            sender=kwargs["member_name"],
            recipient=TEAM_LEADER_NAME,
            body="shutting down after delay",
            kind="shutdown_response",
        ))
        kwargs["agent"].team.confirm_shutdown(kwargs["member_name"])

    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=delayed_ack_runtime,
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    await asyncio.sleep(0)
    record = mgr._tasks_store.list(kind="teammate")[0]

    remove_task = asyncio.create_task(
        mgr.aremove_member("alice", timeout_sec=2.0),
    )
    try:
        await asyncio.wait_for(stop_seen.wait(), timeout=1.0)
        assert record.status == "running"
        assert not mgr._tasks_store.terminal_event(record.id).is_set()

        ack_now.set()
        assert await remove_task is True
        record = mgr._tasks_store.list(kind="teammate")[0]
        assert record.status == "cancelled"
        assert mgr._tasks_store.terminal_event(record.id).is_set()
    finally:
        if not remove_task.done():
            remove_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await remove_task


@pytest.mark.asyncio
async def test_remove_member_falls_back_to_force_kill_on_timeout(
    tmp_path: Path,
) -> None:
    """When the runtime does NOT ack, aremove_member force-kills + journals.

    ``_silent_runtime`` deliberately ignores shutdown_request and waits
    on its abort signal. ``aremove_member`` should time out, fire the
    abort, and return ``False``.
    """
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader_stub(storage)
    captured: dict[str, AbortController] = {}

    async def silent(**kwargs: Any) -> None:
        captured["abort"] = kwargs["abort"]
        try:
            await kwargs["abort"].signal.wait()
        except asyncio.CancelledError:
            return

    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=silent,
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    await asyncio.sleep(0)
    acked = await mgr.aremove_member("alice", timeout_sec=0.3)
    assert acked is False
    assert captured["abort"].aborted is True
    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.status == "cancelled"


@pytest.mark.asyncio
async def test_response_includes_member_name_for_correlation(
    tmp_path: Path,
) -> None:
    """Two members removed concurrently — each ack carries its own sender."""
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader_stub(storage)

    async def acking_runtime(**kwargs: Any) -> None:
        member_name = kwargs["member_name"]
        try:
            await asyncio.wait_for(kwargs["stop_event"].wait(), timeout=10)
        except TimeoutError:
            return
        Mailbox(kwargs["storage"], kwargs["team_id"]).append(TeamMessage(
            msg_id=uuid.uuid4().hex,
            sender=member_name,
            recipient=TEAM_LEADER_NAME,
            body=f"shutting down: ack from {member_name}",
            kind="shutdown_response",
        ))
        kwargs["agent"].team.confirm_shutdown(member_name)

    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=acking_runtime,
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    mgr.add_member("bob")
    await asyncio.sleep(0)
    # Remove sequentially so each ack lands in turn; the correlation
    # check is per-message sender, not interleaving.
    a = await mgr.aremove_member("alice", timeout_sec=2.0)
    b = await mgr.aremove_member("bob", timeout_sec=2.0)
    assert a is True
    assert b is True
    box = Mailbox(storage, mgr.team.team_id)  # type: ignore[union-attr]  # narrowed by assert above; mypy keeps union
    leader_inbox = box.read_all(TEAM_LEADER_NAME)
    senders = {
        m.sender for m in leader_inbox if m.kind == "shutdown_response"
    }
    assert senders == {"alice", "bob"}


@pytest.mark.asyncio
async def test_double_shutdown_is_idempotent(tmp_path: Path) -> None:
    """A second aremove_member on a removed member raises TeamError.

    Idempotence here means: the SECOND call does not silently succeed
    (which would suggest stale state) and does not crash with a
    KeyError or AttributeError. It surfaces a clean TeamError.
    """
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader_stub(storage)

    async def acking_runtime(**kwargs: Any) -> None:
        member_name = kwargs["member_name"]
        try:
            await asyncio.wait_for(kwargs["stop_event"].wait(), timeout=10)
        except TimeoutError:
            return
        Mailbox(kwargs["storage"], kwargs["team_id"]).append(TeamMessage(
            msg_id=uuid.uuid4().hex,
            sender=member_name,
            recipient=TEAM_LEADER_NAME,
            body=f"shutting down: ack from {member_name}",
            kind="shutdown_response",
        ))
        kwargs["agent"].team.confirm_shutdown(member_name)

    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=acking_runtime,
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    await asyncio.sleep(0)
    first = await mgr.aremove_member("alice", timeout_sec=2.0)
    assert first is True
    # Second call: alice is no longer a member.
    with pytest.raises(TeamError, match="not found"):
        await mgr.aremove_member("alice", timeout_sec=0.5)


@pytest.mark.asyncio
async def test_confirm_shutdown_is_idempotent_when_no_waiter(
    tmp_path: Path,
) -> None:
    """``confirm_shutdown`` is safe to call without an active waiter.

    Real runtimes can race the leader's teardown path (pane subprocess
    acking after force-kill, in-process runtime acking twice). The
    manager must absorb both ``no-future`` and ``future-already-done``
    paths silently rather than raising.
    """
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader_stub(storage)
    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=_noop_runner,
    )
    mgr.create_team("alpha")
    # No future exists for "alice" — must no-op.
    mgr.confirm_shutdown("alice")
    # Allocate a future, resolve it, then ack again — also no-op.
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[bool] = loop.create_future()
    mgr._shutdown_acks["bob"] = fut
    fut.set_result(True)
    mgr.confirm_shutdown("bob")  # already done — must not raise
