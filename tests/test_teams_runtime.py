"""run_teammate — long-lived loop driving an AgentSession on mailbox messages."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest

from aura.application.session import AgentSession
from aura.application.tasks.store import TasksStore
from aura.application.teams.mailbox import Mailbox
from aura.application.teams.manager import TeamManager
from aura.application.teams.runtime import (
    _format_envelope,
    run_teammate,
    run_teammate_main,
)
from aura.application.teams.team_port import TeammateBinding, TeamPort
from aura.config.schema import AuraConfig
from aura.domain.abort import AbortController, AbortException
from aura.domain.events import Final, PermissionAudit, ToolCallProgress, ToolCallStarted
from aura.domain.team import TeamMessage, TeamMessageKind, TeamRecord
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.wire.events import CoordinationEvent
from tests.conftest import FakeChatModel


@pytest.fixture(autouse=True)
def _stub_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # add_member spawns a real teammate; llm.create resolves $OPENAI_API_KEY at construction time.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-tests")


def _msg(body: str = "hi", kind: TeamMessageKind = "text", sender: str = "leader") -> TeamMessage:
    return TeamMessage(
        msg_id=uuid.uuid4().hex,
        sender=sender,
        recipient="alice",
        body=body,
        kind=kind,
    )


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "sessions.db")


def _leader_agent(tmp_path: Path) -> AgentSession:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "teams": {"enabled": True},
    })
    return AgentSession(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
    )


def _manager(tmp_path: Path) -> TeamManager:
    leader = _leader_agent(tmp_path)
    return TeamManager(
        leader=leader,
        storage=leader.storage,
        factory=leader.subagent_factory,
        running_aborts=leader.running_aborts,
        tasks_store=leader.tasks_store,
        runtime_runner=run_teammate,
    )


class _ScriptedAgent:
    """Minimal AgentSession stand-in: yields a Final per astream call.

    Records every prompt it sees so tests can assert envelope shape.
    """

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = replies or ["ack"]
        self.prompts_seen: list[str] = []
        self._idx = 0
        self._teammate: TeammateBinding | None = None

    @property
    def team(self) -> TeamPort | None:
        return None

    @property
    def teammate(self) -> TeammateBinding | None:
        return self._teammate

    async def astream(self, prompt: str, *, abort: Any = None) -> Any:
        self.prompts_seen.append(prompt)
        msg = self.replies[min(self._idx, len(self.replies) - 1)]
        self._idx += 1
        yield Final(message=msg, reason="natural")


class _ProgressAgent(_ScriptedAgent):
    async def astream(self, prompt: str, *, abort: Any = None) -> Any:
        self.prompts_seen.append(prompt)
        yield ToolCallStarted("bash", {"command": "echo hi"}, id="tc_1")
        yield ToolCallProgress("bash", "stdout", "hi\n", id="tc_1")
        yield PermissionAudit("bash", "auto-allowed")
        yield Final(message="done", reason="natural")


def test_format_envelope_wraps_each_sender() -> None:
    msgs = [
        _msg(body="hello", sender="leader"),
        _msg(body="follow up", sender="leader"),
    ]
    out = _format_envelope(msgs)
    assert "<from-leader>" in out
    assert "</from-leader>" in out
    assert "hello" in out
    assert "follow up" in out


@pytest.mark.asyncio
async def test_runtime_processes_text_message(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    box = Mailbox(storage, "team-a")
    agent = _ScriptedAgent(replies=["got it"])
    abort = AbortController()
    stop = asyncio.Event()
    box.append(_msg(body="please work"))
    _agent: Any = agent
    task = asyncio.create_task(run_teammate(
        agent=_agent,
        team_id="team-a",
        member_name="alice",
        storage=storage,
        stop_event=stop,
        abort=abort,
    ))
    # Give the runtime a window to consume the message.
    for _ in range(40):
        await asyncio.sleep(0.05)
        if agent.prompts_seen:
            break
    stop.set()
    await asyncio.wait_for(task, timeout=10)
    assert agent.prompts_seen, "runtime never consumed the message"
    assert "please work" in agent.prompts_seen[0]
    # And the .seen cursor advanced — no more unseen.
    assert box.read_unseen("alice") == []


@pytest.mark.asyncio
async def test_runtime_records_teammate_task_progress(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    box = Mailbox(storage, "team-a")
    store = TasksStore()
    record = store.create(
        "teammate: alice",
        "(idle teammate; awaiting messages)",
        kind="teammate",
    )
    agent = _ProgressAgent(replies=["got it"])
    agent._teammate = TeammateBinding(task_id=record.id, tasks_store=store)
    abort = AbortController()
    stop = asyncio.Event()
    box.append(_msg(body="please work"))

    _agent: Any = agent
    task = asyncio.create_task(run_teammate(
        agent=_agent,
        team_id="team-a",
        member_name="alice",
        storage=storage,
        stop_event=stop,
        abort=abort,
    ))
    for _ in range(40):
        await asyncio.sleep(0.05)
        refreshed = store.get(record.id)
        if refreshed is not None and refreshed.progress.tool_count > 0:
            break
    stop.set()
    await asyncio.wait_for(task, timeout=10)

    refreshed = store.get(record.id)
    assert refreshed is not None
    assert refreshed.progress.tool_count == 1
    assert refreshed.progress.last_activity_at is not None
    assert "bash" in refreshed.progress.recent_activities
    assert "bash:stdout> hi" in refreshed.progress.recent_activities
    assert "permission:bash" in refreshed.progress.recent_activities
    assert "final" in refreshed.progress.recent_activities


@pytest.mark.asyncio
async def test_runtime_seed_prompt_runs_immediately(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    agent = _ScriptedAgent(replies=["seed-ack"])
    abort = AbortController()
    stop = asyncio.Event()
    _agent: Any = agent
    task = asyncio.create_task(run_teammate(
        agent=_agent,
        team_id="team-a",
        member_name="alice",
        storage=storage,
        stop_event=stop,
        abort=abort,
        seed_prompt="please scan the repo",
    ))
    # Wait until seed prompt is consumed
    for _ in range(40):
        await asyncio.sleep(0.05)
        if agent.prompts_seen:
            break
    stop.set()
    await asyncio.wait_for(task, timeout=10)
    assert agent.prompts_seen[0] == "please scan the repo"


@pytest.mark.asyncio
async def test_runtime_shutdown_request_exits_cleanly(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    box = Mailbox(storage, "team-a")
    agent = _ScriptedAgent()
    abort = AbortController()
    stop = asyncio.Event()
    box.append(_msg(kind="shutdown_request", body="please go"))
    _agent: Any = agent
    task = asyncio.create_task(run_teammate(
        agent=_agent,
        team_id="team-a",
        member_name="alice",
        storage=storage,
        stop_event=stop,
        abort=abort,
    ))
    await asyncio.wait_for(task, timeout=15)
    # No model invocation — shutdown short-circuits before the turn.
    assert agent.prompts_seen == []


@pytest.mark.asyncio
async def test_runtime_abort_stops_loop(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    agent = _ScriptedAgent()
    abort = AbortController()
    stop = asyncio.Event()
    _agent: Any = agent
    task = asyncio.create_task(run_teammate(
        agent=_agent,
        team_id="team-a",
        member_name="alice",
        storage=storage,
        stop_event=stop,
        abort=abort,
    ))
    await asyncio.sleep(0.1)
    abort.abort("test")
    await asyncio.wait_for(task, timeout=15)


@pytest.mark.asyncio
async def test_team_lifecycle_events_create_add_remove_timeout_path(
    tmp_path: Path,
) -> None:
    mgr = _manager(tmp_path)
    mgr.create_team("demo")
    created_events = mgr.drain_protocol_events()
    assert [e["action"] for e in created_events] == ["team_lifecycle"]
    assert created_events[0]["payload"] == {"state": "active"}

    mgr.add_member("alice")
    add_events = mgr.drain_protocol_events()
    add_actions = [e["action"] for e in add_events]
    assert add_actions == [
        "member_lifecycle",
        "member_lifecycle",
        "member_lifecycle",
    ]
    assert [e["payload"] for e in add_events] == [
        {"state": "starting"},
        {"state": "ready", "previous_state": "starting"},
        {"state": "idle", "previous_state": "ready"},
    ]

    acked = await mgr.aremove_member("alice", timeout_sec=0.01)
    assert acked is False
    remove_events = mgr.drain_protocol_events()
    remove_actions = [e["action"] for e in remove_events]
    assert remove_actions == ["member_lifecycle", "member_lifecycle"]
    assert remove_events[0]["payload"] == {
        "state": "draining",
        "previous_state": "idle",
    }
    assert remove_events[1]["payload"] == {
        "state": "terminated",
        "previous_state": "draining",
        "reason": "forced_timeout",
    }


@pytest.mark.asyncio
async def test_team_lifecycle_events_delete_emits_draining_then_terminated(
    tmp_path: Path,
) -> None:
    mgr = _manager(tmp_path)
    mgr.create_team("demo")
    mgr.drain_protocol_events()

    mgr.delete_team()
    events = mgr.drain_protocol_events()
    assert [e["action"] for e in events] == ["team_lifecycle", "team_lifecycle"]
    assert events[0]["payload"] == {
        "state": "draining",
        "previous_state": "active",
    }
    assert events[1]["payload"] == {
        "state": "terminated",
        "previous_state": "draining",
    }


@pytest.mark.asyncio
async def test_team_view_snapshot_includes_additive_lifecycle_state(
    tmp_path: Path,
) -> None:
    mgr = _manager(tmp_path)
    record = mgr.create_team("demo")
    assert record.lifecycle_state == "active"

    mgr.add_member("alice")
    snap = mgr.view_state("demo")
    member = next(m for m in snap.members if m.name == "alice")
    assert member.status == "active"
    assert member.lifecycle_state == "idle"


class _RaisingAgent(_ScriptedAgent):
    async def astream(self, prompt: str, *, abort: Any = None) -> Any:
        self.prompts_seen.append(prompt)
        raise RuntimeError("turn blew up")
        yield  # noqa: W0101 — unreachable; marks this an async generator


class _AbortingAgent(_ScriptedAgent):
    async def astream(self, prompt: str, *, abort: Any = None) -> Any:
        self.prompts_seen.append(prompt)
        raise AbortException
        yield  # noqa: W0101 — unreachable; marks this an async generator


@pytest.mark.asyncio
async def test_runtime_survives_per_turn_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn that raises is journaled and skipped — one bad message can't kill the teammate."""
    events: list[str] = []
    monkeypatch.setattr(
        "aura.infrastructure.persistence.journal.write",
        lambda name, **_: events.append(name),
    )
    storage = _storage(tmp_path)
    Mailbox(storage, "team-a").append(_msg(body="boom"))
    agent: Any = _RaisingAgent()
    stop = asyncio.Event()
    task = asyncio.create_task(run_teammate(
        agent=agent, team_id="team-a", member_name="alice",
        storage=storage, stop_event=stop, abort=AbortController(),
    ))
    for _ in range(40):
        await asyncio.sleep(0.05)
        if "team_runtime_turn_failed" in events:
            break
    stop.set()
    await asyncio.wait_for(task, timeout=10)  # completes cleanly, not crashed
    assert "team_runtime_turn_failed" in events


@pytest.mark.asyncio
async def test_runtime_breaks_loop_on_abort_during_turn(tmp_path: Path) -> None:
    """An AbortException mid-turn exits the loop on its own — not swallowed as a normal failure."""
    storage = _storage(tmp_path)
    Mailbox(storage, "team-a").append(_msg(body="go"))
    agent: Any = _AbortingAgent()
    task = asyncio.create_task(run_teammate(
        agent=agent, team_id="team-a", member_name="alice",
        storage=storage, stop_event=asyncio.Event(), abort=AbortController(),
    ))
    await asyncio.wait_for(task, timeout=10)  # abort breaks the loop without stop.set()
    assert agent.prompts_seen  # the turn was attempted before aborting


class _RecordingTeam:
    """TeamPort stub recording shutdown acks so the runtime's confirm/send path is observable."""

    def __init__(self, *, is_active: bool = True) -> None:
        self.is_active = is_active
        self.team: TeamRecord | None = None
        self.pending_protocol_events: tuple[CoordinationEvent, ...] = ()
        self.confirmed: list[tuple[str, str]] = []
        self.sent: list[tuple[str, str, str, TeamMessageKind]] = []

    @property
    def storage(self) -> SessionStorage:
        raise NotImplementedError

    def post_message(self, msg: TeamMessage) -> None: ...

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        body: str,
        kind: TeamMessageKind = "text",
    ) -> list[TeamMessage]:
        self.sent.append((sender, recipient, body, kind))
        return []

    def confirm_shutdown(self, member_name: str, *, body: str = "") -> None:
        self.confirmed.append((member_name, body))

    def drain_protocol_events(self) -> list[CoordinationEvent]:
        return []

    async def cleanup_session_teams(self) -> None: ...


class _TeamAgent(_ScriptedAgent):
    """ScriptedAgent bound to a TeamPort so the shutdown ack branch can be exercised."""

    def __init__(self, team: TeamPort, replies: list[str] | None = None) -> None:
        super().__init__(replies=replies)
        self._team = team

    @property
    def team(self) -> TeamPort | None:
        return self._team


@pytest.mark.asyncio
async def test_runtime_shutdown_acks_via_active_team(tmp_path: Path) -> None:
    """Active team must receive confirm_shutdown + a shutdown_response so the leader unblocks."""
    storage = _storage(tmp_path)
    team = _RecordingTeam(is_active=True)
    agent = _TeamAgent(team)
    box = Mailbox(storage, "team-a")
    box.append(_msg(kind="shutdown_request", body="wind down"))
    _agent: Any = agent
    task = asyncio.create_task(run_teammate(
        agent=_agent, team_id="team-a", member_name="alice",
        storage=storage, stop_event=asyncio.Event(), abort=AbortController(),
    ))
    await asyncio.wait_for(task, timeout=15)
    assert agent.prompts_seen == []  # shutdown short-circuits before any model turn
    assert team.confirmed == [("alice", "wind down")]
    assert team.sent == [("alice", "leader", "shutting down: wind down", "shutdown_response")]


@pytest.mark.asyncio
async def test_runtime_shutdown_skips_team_acks_when_inactive(tmp_path: Path) -> None:
    """An inactive team is not acked — only the in-process active path round-trips to the leader."""
    storage = _storage(tmp_path)
    team = _RecordingTeam(is_active=False)
    agent = _TeamAgent(team)
    box = Mailbox(storage, "team-a")
    box.append(_msg(kind="shutdown_request", body="halt"))
    _agent: Any = agent
    task = asyncio.create_task(run_teammate(
        agent=_agent, team_id="team-a", member_name="alice",
        storage=storage, stop_event=asyncio.Event(), abort=AbortController(),
    ))
    await asyncio.wait_for(task, timeout=15)
    assert team.confirmed == []  # is_active gate suppresses the ack
    assert team.sent == []
    assert box.read_unseen("alice") == []  # message still acked/consumed


@pytest.mark.asyncio
async def test_runtime_ignores_non_text_non_shutdown_kinds(tmp_path: Path) -> None:
    """A lone shutdown_response yields no text turn — only text kinds drive the model."""
    storage = _storage(tmp_path)
    agent = _ScriptedAgent()
    stop = asyncio.Event()
    box = Mailbox(storage, "team-a")
    box.append(_msg(kind="shutdown_response", body="ok"))
    _agent: Any = agent
    task = asyncio.create_task(run_teammate(
        agent=_agent, team_id="team-a", member_name="alice",
        storage=storage, stop_event=stop, abort=AbortController(),
    ))
    for _ in range(40):
        await asyncio.sleep(0.05)
        if box.read_unseen("alice") == []:
            break
    stop.set()
    await asyncio.wait_for(task, timeout=10)
    assert agent.prompts_seen == []  # response-only batch never reaches a turn


class _ClosableAgent:
    """build_agent stand-in recording aclose so run_teammate_main's finally is observable."""

    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self, *, mcp_timeout: float = 5.0) -> None:
        self.closed += 1


def _minimal_config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


@pytest.mark.asyncio
async def test_run_teammate_main_returns_zero_and_closes_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry point must always tear the agent down and report success once the loop returns."""
    agent = _ClosableAgent()
    seen: dict[str, Any] = {}

    def _fake_build(config: AuraConfig, *, session_id: str = "") -> Any:
        seen["session_id"] = session_id
        seen["default"] = config.router["default"]
        return agent

    async def _fake_run(**kwargs: Any) -> None:
        seen["ran"] = kwargs["member_name"]

    monkeypatch.setattr(
        "aura.application.teams.runtime.load_config", _minimal_config,
    )
    monkeypatch.setattr("aura.application.teams.runtime.build_agent", _fake_build)
    monkeypatch.setattr("aura.application.teams.runtime.run_teammate", _fake_run)

    rc = await run_teammate_main(
        team_id="t1", member_name="bob", storage_root=str(tmp_path),
    )
    assert rc == 0
    assert agent.closed == 1  # finally closed the agent exactly once
    assert seen["session_id"] == "team-t1-bob"
    assert seen["default"] == "openai:gpt-4o-mini"  # no model override applied
    assert seen["ran"] == "bob"


@pytest.mark.asyncio
async def test_run_teammate_main_applies_model_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """model_name must rewrite router['default'] so the teammate uses the chosen model."""
    captured: dict[str, str] = {}

    def _fake_build(config: AuraConfig, *, session_id: str = "") -> Any:
        captured["default"] = config.router["default"]
        return _ClosableAgent()

    async def _fake_run(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "aura.application.teams.runtime.load_config", _minimal_config,
    )
    monkeypatch.setattr("aura.application.teams.runtime.build_agent", _fake_build)
    monkeypatch.setattr("aura.application.teams.runtime.run_teammate", _fake_run)

    rc = await run_teammate_main(
        team_id="t2", member_name="carol", storage_root=str(tmp_path),
        model_name="openai:gpt-4o", agent_type="explore", system_prompt="hi",
    )
    assert rc == 0
    assert captured["default"] == "openai:gpt-4o"  # override merged into router


@pytest.mark.asyncio
async def test_run_teammate_main_closes_agent_even_when_loop_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashing loop must still flow through aclose — the finally is the only teardown path."""
    agent = _ClosableAgent()

    def _fake_build(config: AuraConfig, *, session_id: str = "") -> Any:
        return agent

    async def _fake_run(**kwargs: Any) -> None:
        raise RuntimeError("loop exploded")

    monkeypatch.setattr(
        "aura.application.teams.runtime.load_config", _minimal_config,
    )
    monkeypatch.setattr("aura.application.teams.runtime.build_agent", _fake_build)
    monkeypatch.setattr("aura.application.teams.runtime.run_teammate", _fake_run)

    with pytest.raises(RuntimeError, match="loop exploded"):
        await run_teammate_main(
            team_id="t3", member_name="dave", storage_root=str(tmp_path),
        )
    assert agent.closed == 1  # finally ran despite the loop raising
