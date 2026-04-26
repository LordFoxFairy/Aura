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
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from aura.config.schema import AuraConfig
from aura.core.abort import AbortController
from aura.core.permissions.safety import DEFAULT_SAFETY
from aura.core.permissions.session import RuleSet
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.store import TasksStore
from aura.core.teams.mailbox import Mailbox
from aura.core.teams.manager import TeamError, TeamManager
from aura.core.teams.runtime import run_teammate
from aura.core.teams.types import TEAM_LEADER_NAME, TeamMessage
from aura.schemas.events import Final
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
    box = Mailbox(storage, mgr.team.team_id)  # type: ignore[union-attr]
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
            agent=agent,  # type: ignore[arg-type]
            team_id=mgr.team.team_id,  # type: ignore[union-attr]
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
    """``aremove_member`` returns True (ack) when the runtime responds in time.

    The runtime stand-in ACKs by writing a shutdown_response to the
    leader's mailbox before exiting. ``aremove_member`` must observe it
    and skip the force-kill path entirely.
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
    box = Mailbox(storage, mgr.team.team_id)  # type: ignore[union-attr]
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
