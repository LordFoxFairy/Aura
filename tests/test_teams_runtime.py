"""run_teammate — long-lived loop driving an AgentSession on mailbox messages."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest

from aura.application.session import AgentSession
from aura.application.tasks.store import TasksStore
from aura.application.teams.mailbox import Mailbox, QueueMailboxNotifier
from aura.application.teams.manager import TeamManager
from aura.application.teams.runtime import (
    _drive_one_turn,
    _format_envelope,
    _wait_for_message,
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


class _BlankProgressAgent(_ScriptedAgent):
    """Emits a whitespace-only ToolCallProgress chunk to exercise the empty-chunk skip."""

    async def astream(self, prompt: str, *, abort: Any = None) -> Any:
        self.prompts_seen.append(prompt)
        yield ToolCallStarted("bash", {"command": "true"}, id="tc_blank")
        yield ToolCallProgress("bash", "stdout", "   \n", id="tc_blank")
        yield Final(message="done", reason="natural")


@pytest.mark.asyncio
async def test_drive_turn_skips_blank_progress_chunk(tmp_path: Path) -> None:
    """A whitespace-only tool chunk must not pollute the activity log — only real lines persist."""
    storage = _storage(tmp_path)
    store = TasksStore()
    record = store.create("teammate: alice", "(idle)", kind="teammate")
    agent = _BlankProgressAgent()
    agent._teammate = TeammateBinding(task_id=record.id, tasks_store=store)
    _agent: Any = agent

    out = await _drive_one_turn(
        agent=_agent, prompt="go", abort=AbortController(),
        storage=storage, team_id="team-a", member_name="alice",
    )

    assert out == "done"
    refreshed = store.get(record.id)
    assert refreshed is not None
    assert refreshed.progress.tool_count == 1  # the start still counted
    notes = refreshed.progress.recent_activities
    assert "bash" in notes
    assert not any("stdout>" in n for n in notes)  # blank chunk produced no note


class _LongFinalAgent(_ScriptedAgent):
    """Yields an oversized Final so the 500-char transcript truncation boundary is exercised."""

    def __init__(self, body: str) -> None:
        super().__init__(replies=[body])
        self._body = body

    async def astream(self, prompt: str, *, abort: Any = None) -> Any:
        self.prompts_seen.append(prompt)
        yield Final(message=self._body, reason="natural")


@pytest.mark.asyncio
async def test_drive_turn_truncates_transcript_to_500_chars(tmp_path: Path) -> None:
    """Transcript is a bounded audit trail — a huge final answer is capped at 500 chars on disk."""
    storage = _storage(tmp_path)
    body = "x" * 1200
    agent: Any = _LongFinalAgent(body)

    out = await _drive_one_turn(
        agent=agent, prompt="go", abort=AbortController(),
        storage=storage, team_id="team-a", member_name="alice",
    )

    assert out == body  # the returned final text is NOT truncated, only the transcript
    written = storage.team_transcript_path("team-a", "alice").read_text(encoding="utf-8")
    assert "x" * 500 in written
    assert "x" * 501 not in written  # capped at the 500-char slice


@pytest.mark.asyncio
async def test_drive_turn_without_teammate_binding_skips_task_recording(
    tmp_path: Path,
) -> None:
    """An unbound agent (no task) must still complete a turn — task recording is purely additive."""
    storage = _storage(tmp_path)
    agent = _ScriptedAgent(replies=["ok"])
    assert agent.teammate is None  # no binding: every record_* call must short-circuit
    _agent: Any = agent

    out = await _drive_one_turn(
        agent=_agent, prompt="go", abort=AbortController(),
        storage=storage, team_id="team-a", member_name="alice",
    )

    assert out == "ok"  # turn drives to a Final despite no tracking sink
    written = storage.team_transcript_path("team-a", "alice").read_text(encoding="utf-8")
    assert "Final" in written  # transcript still records the event type


@pytest.mark.asyncio
async def test_drive_turn_suppresses_transcript_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unwritable transcript can't abort the turn — the OSError is suppressed, final returns."""
    storage = _storage(tmp_path)
    agent = _ScriptedAgent(replies=["resilient"])
    _agent: Any = agent

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", _boom)

    out = await _drive_one_turn(
        agent=_agent, prompt="go", abort=AbortController(),
        storage=storage, team_id="team-a", member_name="alice",
    )

    assert out == "resilient"  # suppress(OSError) keeps the turn alive


@pytest.mark.parametrize("seed", ["", "   ", "\n\t "])
@pytest.mark.asyncio
async def test_runtime_blank_seed_prompt_does_not_drive_turn(
    tmp_path: Path, seed: str,
) -> None:
    """A blank/whitespace seed must be ignored — only real text triggers an immediate turn."""
    storage = _storage(tmp_path)
    agent = _ScriptedAgent(replies=["should-not-run"])
    stop = asyncio.Event()
    stop.set()  # stop immediately so the loop body never runs; only the seed path matters
    _agent: Any = agent

    await asyncio.wait_for(
        run_teammate(
            agent=_agent, team_id="team-a", member_name="alice",
            storage=storage, stop_event=stop, abort=AbortController(),
            seed_prompt=seed,
        ),
        timeout=10,
    )

    assert agent.prompts_seen == []  # blank seed never reached _drive_one_turn


class _OneShotNotifier:
    """Notifier driving exactly N successful loop turns, then stopping the loop.

    Each ``wait_new`` call before the budget returns True (a message is ready) and
    leaves the in-task loop running so coverage traces the loop body; the call past
    the budget sets ``stop_event`` and returns False so the loop exits deterministically
    with no polling, sleeps, or real filesystem watching.
    """

    def __init__(self, stop_event: asyncio.Event, *, turns: int = 1) -> None:
        self._stop = stop_event
        self._turns = turns
        self.calls = 0

    async def wait_new(self, member: str, *, timeout: float) -> bool:
        del member, timeout
        self.calls += 1
        if self.calls > self._turns:
            self._stop.set()
            return False
        return True

    def signal(self, member: str) -> None:
        del member


@pytest.mark.asyncio
async def test_wait_for_message_returns_false_when_stop_fires_first(
    tmp_path: Path,
) -> None:
    """A stop signal must win the race against the mailbox so the loop can shut down."""
    del tmp_path
    stop = asyncio.Event()
    stop.set()  # stop already pending: it must be the branch that resolves the wait
    notifier = QueueMailboxNotifier()
    result = await asyncio.wait_for(
        _wait_for_message(notifier, "alice", stop, timeout=0.05),
        timeout=5,
    )
    assert result is False  # stop_task in done -> caller breaks out of the loop


@pytest.mark.asyncio
async def test_wait_for_message_returns_true_on_new_message() -> None:
    """A signaled mailbox must report a message so the loop proceeds to drain it."""
    stop = asyncio.Event()
    notifier = QueueMailboxNotifier()
    notifier.signal("alice")  # a message arrived before the wait started
    result = await asyncio.wait_for(
        _wait_for_message(notifier, "alice", stop, timeout=0.05),
        timeout=5,
    )
    assert result is True  # wait_task wins -> caller reads the unseen batch


@pytest.mark.asyncio
async def test_wait_for_message_returns_false_on_timeout() -> None:
    """An idle slice must report no message so the loop re-checks its exit conditions."""
    stop = asyncio.Event()
    notifier = QueueMailboxNotifier()  # never signaled
    result = await asyncio.wait_for(
        _wait_for_message(notifier, "alice", stop, timeout=0.02),
        timeout=5,
    )
    assert result is False  # wait_task.result() is False -> loop `continue`


@pytest.mark.asyncio
async def test_runtime_loop_drives_text_then_acks_and_exits(tmp_path: Path) -> None:
    """One queued text message must drive exactly one model turn and advance the seen cursor."""
    storage = _storage(tmp_path)
    box = Mailbox(storage, "team-a")
    box.append(_msg(body="ship it"))
    agent = _ScriptedAgent(replies=["done"])
    stop = asyncio.Event()
    notifier: Any = _OneShotNotifier(stop, turns=1)
    _agent: Any = agent

    await asyncio.wait_for(
        run_teammate(
            agent=_agent, team_id="team-a", member_name="alice",
            storage=storage, stop_event=stop, abort=AbortController(),
            notifier=notifier,
        ),
        timeout=5,
    )

    assert agent.prompts_seen == ["<from-leader>\nship it\n</from-leader>"]
    assert box.read_unseen("alice") == []  # message acked, cursor advanced


@pytest.mark.asyncio
async def test_runtime_loop_skips_empty_unseen_batch(tmp_path: Path) -> None:
    """A wake-up with no actual unseen message must not drive a turn or crash the loop."""
    storage = _storage(tmp_path)
    Mailbox(storage, "team-a")  # inbox exists but is empty
    agent = _ScriptedAgent(replies=["unused"])
    stop = asyncio.Event()
    # turns=1 lets wait_new report a message, but the mailbox is empty -> `continue`.
    notifier: Any = _OneShotNotifier(stop, turns=1)
    _agent: Any = agent

    await asyncio.wait_for(
        run_teammate(
            agent=_agent, team_id="team-a", member_name="alice",
            storage=storage, stop_event=stop, abort=AbortController(),
            notifier=notifier,
        ),
        timeout=5,
    )

    assert agent.prompts_seen == []  # empty unseen short-circuits before any turn


@pytest.mark.asyncio
async def test_runtime_loop_acks_response_only_batch_without_turn(
    tmp_path: Path,
) -> None:
    """A batch of only non-text kinds is consumed/acked but never reaches the model."""
    storage = _storage(tmp_path)
    box = Mailbox(storage, "team-a")
    box.append(_msg(kind="shutdown_response", body="ok"))
    agent = _ScriptedAgent(replies=["unused"])
    stop = asyncio.Event()
    notifier: Any = _OneShotNotifier(stop, turns=1)
    _agent: Any = agent

    await asyncio.wait_for(
        run_teammate(
            agent=_agent, team_id="team-a", member_name="alice",
            storage=storage, stop_event=stop, abort=AbortController(),
            notifier=notifier,
        ),
        timeout=5,
    )

    assert agent.prompts_seen == []  # no text kinds -> `continue`, no turn
    assert box.read_unseen("alice") == []  # still acked so it can't redeliver forever


@pytest.mark.asyncio
async def test_runtime_loop_shutdown_acks_active_team(tmp_path: Path) -> None:
    """A shutdown_request consumed in the loop must confirm + respond to unblock the leader."""
    storage = _storage(tmp_path)
    box = Mailbox(storage, "team-a")
    box.append(_msg(kind="shutdown_request", body="wind down"))
    team = _RecordingTeam(is_active=True)
    agent = _TeamAgent(team)
    stop = asyncio.Event()
    notifier: Any = _OneShotNotifier(stop, turns=1)
    _agent: Any = agent

    await asyncio.wait_for(
        run_teammate(
            agent=_agent, team_id="team-a", member_name="alice",
            storage=storage, stop_event=stop, abort=AbortController(),
            notifier=notifier,
        ),
        timeout=5,
    )

    assert agent.prompts_seen == []  # shutdown breaks before any model turn
    assert team.confirmed == [("alice", "wind down")]
    assert team.sent == [
        ("alice", "leader", "shutting down: wind down", "shutdown_response"),
    ]


@pytest.mark.asyncio
async def test_runtime_loop_survives_per_turn_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn raising mid-loop is journaled and the loop continues to its clean exit."""
    events: list[str] = []
    monkeypatch.setattr(
        "aura.infrastructure.persistence.journal.write",
        lambda name, **_: events.append(name),
    )
    storage = _storage(tmp_path)
    Mailbox(storage, "team-a").append(_msg(body="boom"))
    agent: Any = _RaisingAgent()
    stop = asyncio.Event()
    notifier: Any = _OneShotNotifier(stop, turns=1)

    await asyncio.wait_for(
        run_teammate(
            agent=agent, team_id="team-a", member_name="alice",
            storage=storage, stop_event=stop, abort=AbortController(),
            notifier=notifier,
        ),
        timeout=5,
    )

    assert "team_runtime_turn_failed" in events  # logged, not propagated
    assert "team_runtime_exited" in events  # loop still reached its finally


@pytest.mark.asyncio
async def test_runtime_loop_breaks_on_abort_during_turn(tmp_path: Path) -> None:
    """An AbortException raised by the turn must break the loop without a stop signal."""
    storage = _storage(tmp_path)
    Mailbox(storage, "team-a").append(_msg(body="go"))
    agent: Any = _AbortingAgent()
    stop = asyncio.Event()
    # turns is generous; the AbortException must be what exits the loop, not the budget.
    notifier: Any = _OneShotNotifier(stop, turns=99)

    await asyncio.wait_for(
        run_teammate(
            agent=agent, team_id="team-a", member_name="alice",
            storage=storage, stop_event=stop, abort=AbortController(),
            notifier=notifier,
        ),
        timeout=5,
    )

    assert agent.prompts_seen  # the turn was attempted before aborting
    assert not stop.is_set()  # abort broke the loop; the stop budget never tripped


@pytest.mark.asyncio
async def test_runtime_loop_stops_when_abort_already_set(tmp_path: Path) -> None:
    """A pre-aborted controller must skip the loop body — the while guard short-circuits."""
    storage = _storage(tmp_path)
    Mailbox(storage, "team-a").append(_msg(body="never read"))
    agent = _ScriptedAgent(replies=["unused"])
    abort = AbortController()
    abort.abort("pre-aborted")  # `not abort.aborted` is False on entry
    stop = asyncio.Event()
    notifier: Any = _OneShotNotifier(stop, turns=99)
    _agent: Any = agent

    await asyncio.wait_for(
        run_teammate(
            agent=_agent, team_id="team-a", member_name="alice",
            storage=storage, stop_event=stop, abort=abort,
            notifier=notifier,
        ),
        timeout=5,
    )

    assert agent.prompts_seen == []  # loop guard rejected entry; no turn, no wake-up
    assert notifier.calls == 0  # _wait_for_message never invoked


@pytest.mark.asyncio
async def test_runtime_loop_idempotent_double_shutdown(tmp_path: Path) -> None:
    """Two shutdown_requests in one batch must ack the team once — shutdown is idempotent."""
    storage = _storage(tmp_path)
    box = Mailbox(storage, "team-a")
    box.append(_msg(kind="shutdown_request", body="first"))
    box.append(_msg(kind="shutdown_request", body="second"))
    team = _RecordingTeam(is_active=True)
    agent = _TeamAgent(team)
    stop = asyncio.Event()
    notifier: Any = _OneShotNotifier(stop, turns=1)
    _agent: Any = agent

    await asyncio.wait_for(
        run_teammate(
            agent=_agent, team_id="team-a", member_name="alice",
            storage=storage, stop_event=stop, abort=AbortController(),
            notifier=notifier,
        ),
        timeout=5,
    )

    # `next(...)` picks the first shutdown; the loop breaks after one confirm/send pair.
    assert team.confirmed == [("alice", "first")]
    assert len(team.sent) == 1


@pytest.mark.asyncio
async def test_drive_turn_records_progress_and_permission_notes(
    tmp_path: Path,
) -> None:
    """Tool progress and permission audits must both land as activity notes for the task."""
    storage = _storage(tmp_path)
    store = TasksStore()
    record = store.create("teammate: alice", "(idle)", kind="teammate")
    agent = _ProgressAgent(replies=["done"])
    agent._teammate = TeammateBinding(task_id=record.id, tasks_store=store)
    _agent: Any = agent

    out = await _drive_one_turn(
        agent=_agent, prompt="go", abort=AbortController(),
        storage=storage, team_id="team-a", member_name="alice",
    )

    assert out == "done"
    refreshed = store.get(record.id)
    assert refreshed is not None
    notes = refreshed.progress.recent_activities
    assert "bash:stdout> hi" in notes  # ToolCallProgress note path
    assert "permission:bash" in notes  # PermissionAudit note path


@pytest.mark.asyncio
async def test_drive_turn_abort_journals_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An aborted turn must journal the abort and re-raise so the loop's abort handler runs."""
    events: list[str] = []
    monkeypatch.setattr(
        "aura.infrastructure.persistence.journal.write",
        lambda name, **_: events.append(name),
    )
    storage = _storage(tmp_path)
    agent: Any = _AbortingAgent()

    with pytest.raises(AbortException):
        await _drive_one_turn(
            agent=agent, prompt="go", abort=AbortController(),
            storage=storage, team_id="team-a", member_name="alice",
        )

    assert "team_runtime_aborted" in events  # the except-AbortException branch ran


@pytest.mark.parametrize("body", ["", " ", "0", "False"])
@pytest.mark.asyncio
async def test_drive_turn_returns_exact_final_for_edge_bodies(
    tmp_path: Path, body: str,
) -> None:
    """The returned final text must be the verbatim message, even for falsy/edge strings."""
    storage = _storage(tmp_path)
    agent: Any = _LongFinalAgent(body)

    out = await _drive_one_turn(
        agent=agent, prompt="go", abort=AbortController(),
        storage=storage, team_id="team-a", member_name="alice",
    )

    assert out == body  # no truthiness coercion of the model's final answer


@pytest.mark.asyncio
async def test_run_teammate_main_empty_model_name_keeps_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty model_name is falsy and must NOT rewrite router['default']."""
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
        team_id="t4", member_name="erin", storage_root=str(tmp_path),
        model_name="",
    )
    assert rc == 0
    assert captured["default"] == "openai:gpt-4o-mini"  # empty override ignored


@pytest.mark.asyncio
async def test_run_teammate_main_suppresses_aclose_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing aclose in the finally must be swallowed so teardown still returns 0."""

    class _BadCloseAgent:
        async def aclose(self, *, mcp_timeout: float = 5.0) -> None:
            raise RuntimeError("close blew up")

    def _fake_build(config: AuraConfig, *, session_id: str = "") -> Any:
        return _BadCloseAgent()

    async def _fake_run(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "aura.application.teams.runtime.load_config", _minimal_config,
    )
    monkeypatch.setattr("aura.application.teams.runtime.build_agent", _fake_build)
    monkeypatch.setattr("aura.application.teams.runtime.run_teammate", _fake_run)

    rc = await run_teammate_main(
        team_id="t5", member_name="frank", storage_root=str(tmp_path),
    )
    assert rc == 0  # suppress(Exception) around aclose keeps the return value intact
