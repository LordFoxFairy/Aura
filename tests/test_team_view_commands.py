"""``/team enter|leave|view|teammate`` UX commands and active-team slot handling."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from aura.application.commands.team import TeamCommand
from aura.application.session import AgentSession
from aura.application.tasks.store import TasksStore
from aura.application.teams.mailbox import Mailbox
from aura.application.teams.manager import TeamManager
from aura.application.teams.state import Member, TeamError
from aura.application.teams.view import TeamViewBuilder
from aura.config.schema import AuraConfig
from aura.domain.team import TeammateMember, TeamMessage, TeamRecord
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.wire.serialize import team_message_to_wire
from tests.conftest import FakeChatModel


@pytest.fixture(autouse=True)
def _stub_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a dummy OPENAI_API_KEY so the SubagentSpawner's spawn —
    which goes through the real ``llm.create`` path inside ``add_member``
    — doesn't blow up on missing credentials. The FakeChatModel never
    actually calls out, but the key is resolved at spawn time before
    the model is even constructed.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-tests")


def _agent(tmp_path: Path) -> AgentSession:
    # ``teams.enabled=True`` opens the gate so /team subcommands run.
    cfg = AuraConfig.model_validate(
        {
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
            "teams": {"enabled": True},
        }
    )
    return AgentSession(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "sessions.db"),
    )


async def _no_runtime(**_kwargs: Any) -> None:
    """Stand-in for run_teammate — exits immediately."""
    return


def _install_no_runtime_manager(agent: AgentSession) -> TeamManager:
    """Replace the lazy-built manager with one whose runtime is a no-op.

    The default ``_ensure_manager`` would import the real ``run_teammate``
    on first use; we want every ``add_member`` call to skip the runtime
    so transcript files / mailbox writes don't depend on a background
    asyncio task. Constructing the manager here and stamping
    ``agent._team_manager`` matches what ``_ensure_manager`` would have
    done.
    """
    mgr = TeamManager(
        leader=agent,
        storage=agent.storage,
        factory=agent.subagent_factory,
        running_aborts=agent._running_aborts,
        tasks_store=agent._tasks_store,
        runtime_runner=_no_runtime,
    )
    agent._team_manager = mgr
    return mgr


@pytest.mark.asyncio
async def test_team_enter_sets_active_team(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    # /create joins the leader and makes the new team active.
    assert agent.state.slots.active_team == "demo"
    result = await cmd.handle("enter demo", agent)
    assert result.handled is True
    assert "entered team" in result.text
    assert agent.state.slots.active_team == "demo"


@pytest.mark.asyncio
async def test_team_enter_unknown_team_errors(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    result = await cmd.handle("enter ghost", agent)
    assert "team not found" in result.text
    assert agent.state.slots.active_team is None


@pytest.mark.asyncio
async def test_team_leave_clears_active_team(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("enter demo", agent)
    assert agent.state.slots.active_team == "demo"
    result = await cmd.handle("leave", agent)
    assert "left team" in result.text
    assert agent.state.slots.active_team is None


@pytest.mark.asyncio
async def test_team_view_active_returns_snapshot_with_members_and_messages(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    mgr = _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("enter demo", agent)
    # Add a member so the view has a row to render.
    await cmd.handle("add scout general-purpose", agent)
    # Send a message so recent_messages is non-empty.
    mgr.send(sender="leader", recipient="scout", body="ping")
    result = await cmd.handle("view", agent)
    assert result.handled is True
    assert result.kind == "view"
    # Members section is present and labels the new member.
    assert "scout" in result.text
    assert "general-purpose" in result.text
    # Recent messages section is present and shows the body.
    assert "leader -> scout" in result.text
    assert "ping" in result.text
    # Footer hint guides the next step.
    assert "/team teammate" in result.text


@pytest.mark.asyncio
async def test_team_send_text_message_maps_to_coordination_wire_event(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    mgr = _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("enter demo", agent)
    await cmd.handle("add scout general-purpose", agent)

    sent = mgr.send(sender="leader", recipient="scout", body="ping")
    payload = team_message_to_wire(sent[0], team_id="demo")

    assert payload == {
        "event": "coordination",
        "family": "team",
        "action": "message_sent",
        "team_id": "demo",
        "member_id": "scout",
        "payload": {
            "msg_id": sent[0].msg_id,
            "sender": "leader",
            "recipient": "scout",
            "body": "ping",
            "kind": "text",
            "sent_at": sent[0].sent_at,
        },
    }


@pytest.mark.asyncio
async def test_team_control_message_maps_to_control_coordination_wire_event(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    mgr = _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("enter demo", agent)
    await cmd.handle("add scout general-purpose", agent)

    sent = mgr.send(
        sender="leader",
        recipient="scout",
        body="shutdown",
        kind="shutdown_request",
    )
    payload = team_message_to_wire(sent[0], team_id="demo")

    assert payload == {
        "event": "coordination",
        "family": "team",
        "action": "control_sent",
        "team_id": "demo",
        "member_id": "scout",
        "payload": {
            "msg_id": sent[0].msg_id,
            "sender": "leader",
            "recipient": "scout",
            "body": "shutdown",
            "kind": "shutdown_request",
            "sent_at": sent[0].sent_at,
        },
    }


@pytest.mark.asyncio
async def test_team_send_enqueues_protocol_events_for_external_stream(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    mgr = _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("enter demo", agent)
    await cmd.handle("add scout general-purpose", agent)

    mgr.drain_protocol_events()  # create/add lifecycle events precede the send; isolate it
    sent = mgr.send(sender="leader", recipient="scout", body="ping")

    assert mgr.pending_protocol_events == (team_message_to_wire(sent[0], team_id="demo"),)


@pytest.mark.asyncio
async def test_agent_drain_protocol_events_includes_team_send_events(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    mgr = _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("enter demo", agent)
    await cmd.handle("add scout general-purpose", agent)

    agent.drain_protocol_events()  # create/add lifecycle events precede the send; isolate it
    sent = mgr.send(sender="leader", recipient="scout", body="ping")

    assert agent.drain_protocol_events() == [
        team_message_to_wire(sent[0], team_id="demo"),
    ]
    assert agent.drain_protocol_events() == []


@pytest.mark.asyncio
async def test_team_view_explicit_name_works_without_active(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create alpha", agent)
    await cmd.handle("leave", agent)
    assert agent.state.slots.active_team is None
    result = await cmd.handle("view alpha", agent)
    assert result.kind == "view"
    assert "alpha" in result.text
    assert "team:" in result.text


@pytest.mark.asyncio
async def test_team_view_no_active_no_arg_errors_clearly(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    result = await cmd.handle("view", agent)
    assert "no active team" in result.text


@pytest.mark.asyncio
async def test_team_teammate_renders_last_50_messages(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("enter demo", agent)
    await cmd.handle("add scout general-purpose", agent)
    # Manually populate the transcript file so we don't depend on the
    # runtime loop. The render expects "<unix-ts> <Event> <body>" lines.
    transcript = agent.storage.team_transcript_path("demo", "scout")
    with transcript.open("w", encoding="utf-8") as f:
        for i in range(60):  # > 50 so the cap actually trims
            f.write(f"{int(time.time())} Final message-{i:03d}\n")
    result = await cmd.handle("teammate scout", agent)
    assert result.handled is True
    assert result.kind == "view"
    assert "transcript: scout" in result.text
    assert "Esc to return" in result.text
    # Cap is 50 — earliest entries (000..009) must NOT appear.
    assert "message-000" not in result.text
    assert "message-009" not in result.text
    # Tail entries (50..59) must all be visible.
    assert "message-059" in result.text
    assert "message-050" in result.text


@pytest.mark.asyncio
async def test_team_teammate_unknown_member_errors(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("enter demo", agent)
    result = await cmd.handle("teammate ghost", agent)
    assert "member not found" in result.text


@pytest.mark.asyncio
async def test_clear_session_resets_active_team(tmp_path: Path) -> None:
    """``AgentSession.clear_session`` rebinds ``state.slots.active_team=None``;
    the active-team pointer must NOT survive a /clear."""
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("enter demo", agent)
    assert agent.state.slots.active_team == "demo"
    agent.clear_session()
    assert agent.state.slots.active_team is None


@pytest.mark.asyncio
async def test_team_view_snapshot_includes_subagent_and_transcript_counts(
    tmp_path: Path,
) -> None:
    """The aggregator surfaces subagent + transcript counts the renderer
    prints. Smoke check that the numbers flow end-to-end."""
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("enter demo", agent)
    await cmd.handle("add scout general-purpose", agent)
    # Drop a transcript file so the count shows non-zero.
    transcript = agent.storage.team_transcript_path("demo", "scout")
    with transcript.open("w", encoding="utf-8") as f:
        f.write(f"{int(time.time())} Final hello\n")
    result = await cmd.handle("view", agent)
    assert "teammate transcripts:" in result.text
    # Exactly one transcript file written above.
    assert "teammate transcripts: 1" in result.text


@pytest.mark.asyncio
async def test_view_state_aggregator_returns_dataclass(tmp_path: Path) -> None:
    """``TeamManager.view_state`` is the data layer the slash command
    renders on top of — verify it returns a dataclass with the expected
    shape so REPL / future Tauri UIs can consume it directly."""
    agent = _agent(tmp_path)
    mgr = _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("add scout general-purpose", agent)
    mgr.send(sender="leader", recipient="scout", body="hi")
    snap = mgr.view_state("demo")
    assert snap.team_id == "demo"
    assert snap.name == "demo"
    assert any(m.name == "scout" for m in snap.members)
    assert any(msg.recipient == "scout" and msg.body == "hi" for msg in snap.recent_messages)
    assert isinstance(snap.subagent_count, int)
    assert isinstance(snap.transcript_count, int)


@pytest.mark.asyncio
async def test_team_view_recent_messages_capped_at_ten(tmp_path: Path) -> None:
    """Aggregator must cap recent_messages at 10 even with more on disk."""
    agent = _agent(tmp_path)
    mgr = _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("add scout general-purpose", agent)
    for i in range(15):
        mgr.send(sender="leader", recipient="scout", body=f"msg-{i}")
    snap = mgr.view_state("demo")
    assert len(snap.recent_messages) == 10
    # Cap applies to the most-recent slice; oldest msg must be absent.
    assert all("msg-0" not in m.body for m in snap.recent_messages[:5])


# --- Direct TeamViewBuilder boundary coverage -------------------------------
# The slash-command tests above always route through a live TeamManager.
# To exercise the off-record disk-reload path, the not-found/unreadable guards,
# the dead/shutting-down status branches, and the artifact-count error handling
# we drive TeamViewBuilder directly with controlled team()/members/storage.


def _record(
    team_id: str = "t1",
    *,
    members: list[TeammateMember] | None = None,
) -> TeamRecord:
    return TeamRecord(
        team_id=team_id,
        name=team_id,
        leader_session_id="sid",
        members=members if members is not None else [],
    )


@dataclass(frozen=True)
class _Built:
    builder: TeamViewBuilder
    storage: SessionStorage


def _builder(
    tmp_path: Path,
    *,
    team: Callable[[], TeamRecord | None],
    members: dict[str, Member] | None = None,
    tasks_store: TasksStore | None = None,
) -> _Built:
    storage = SessionStorage(tmp_path / "sessions.db")
    builder = TeamViewBuilder(
        team=team,
        members=members if members is not None else {},
        storage=storage,
        tasks_store=tasks_store if tasks_store is not None else TasksStore(),
    )
    return _Built(builder=builder, storage=storage)


def test_view_state_no_active_no_arg_raises(tmp_path: Path) -> None:
    """A bare view_state() with no active team must fail loud, not return an
    empty/placeholder snapshot the UI would silently render as a real team."""
    built = _builder(tmp_path, team=lambda: None)
    with pytest.raises(TeamError, match="no team is active"):
        built.builder.view_state()


def test_view_state_explicit_id_missing_on_disk_raises(tmp_path: Path) -> None:
    """Requesting a team that was never persisted must surface a clear
    'not found on disk' error so a typo'd id is not mistaken for an empty team."""
    built = _builder(tmp_path, team=lambda: None)
    with pytest.raises(TeamError, match="not found on disk"):
        built.builder.view_state("ghost")


@pytest.mark.parametrize(
    "corrupt",
    ["{not json", "", "[1, 2,", '{"team_id": "x"'],
    ids=["truncated-brace", "empty-file", "truncated-array", "missing-close"],
)
def test_view_state_unreadable_config_raises(
    tmp_path: Path,
    corrupt: str,
) -> None:
    """Malformed config.json on disk must raise a descriptive TeamError rather
    than crash the whole /team surface with a raw JSONDecodeError."""
    built = _builder(tmp_path, team=lambda: None)
    built.storage.team_config_path("broke").write_text(
        corrupt,
        encoding="utf-8",
    )
    with pytest.raises(TeamError, match="config is unreadable"):
        built.builder.view_state("broke")


def test_view_state_off_record_reload_reflects_disk_writer(
    tmp_path: Path,
) -> None:
    """When the requested id differs from the in-memory team, the snapshot must
    reload config.json so a concurrent writer's roster is visible, not stale."""
    built = _builder(tmp_path, team=lambda: _record("live"))
    disk = _record("other", members=[TeammateMember(name="scout")])
    built.storage.team_config_path("other").write_text(
        disk.model_dump_json(),
        encoding="utf-8",
    )
    snap = built.builder.view_state("other")
    assert snap.team_id == "other"
    assert [m.name for m in snap.members] == ["scout"]


def test_view_state_explicit_id_matching_live_uses_in_memory(
    tmp_path: Path,
) -> None:
    """Passing the active team's own id must short-circuit to the live record,
    not pay an off-record disk reload (and must not need config.json present)."""
    live = _record("live", members=[TeammateMember(name="m1")])
    built = _builder(tmp_path, team=lambda: live)
    snap = built.builder.view_state("live")
    assert snap.team_id == "live"
    assert [m.name for m in snap.members] == ["m1"]


def test_inactive_member_renders_as_dead(tmp_path: Path) -> None:
    """A member flagged is_active=False must show as 'dead' regardless of any
    live slot — a torn-down teammate must never read as still working."""
    record = _record("t", members=[TeammateMember(name="gone", is_active=False)])
    built = _builder(tmp_path, team=lambda: record)
    snap = built.builder.view_state()
    assert snap.members[0].status == "dead"


def test_member_without_slot_defaults_to_unknown_lifecycle(
    tmp_path: Path,
) -> None:
    """A roster member with no runtime slot must report 'unknown' lifecycle and
    fall back to its configured model_name, never inheriting another's spec."""
    record = _record(
        "t",
        members=[TeammateMember(name="solo", model_name="cfg:model")],
    )
    built = _builder(tmp_path, team=lambda: record, members={})
    snap = built.builder.view_state()
    row = snap.members[0]
    assert row.status == "active"
    assert row.lifecycle_state == "unknown"
    assert row.model_spec == "cfg:model"
    assert row.tokens_used == 0
    assert row.last_active is None


def test_empty_lifecycle_state_coerces_to_unknown(tmp_path: Path) -> None:
    """An empty-string lifecycle_state on a slot must coerce to 'unknown' so the
    UI never prints a blank lifecycle column."""
    record = _record("t", members=[TeammateMember(name="m1")])
    members = {"m1": Member(lifecycle_state="")}
    built = _builder(tmp_path, team=lambda: record, members=members)
    snap = built.builder.view_state()
    assert snap.members[0].lifecycle_state == "unknown"


async def test_live_member_shutting_down_status(tmp_path: Path) -> None:
    """A live member with a pending shutdown_waiter must read 'shutting-down' so
    the operator sees the in-flight teardown rather than a false 'active'."""
    record = _record("t", members=[TeammateMember(name="m1")])

    async def _waiter() -> bool:
        return True

    task: asyncio.Task[bool] = asyncio.ensure_future(_waiter())
    members = {"m1": Member(lifecycle_state="running", shutdown_waiter=task)}
    built = _builder(tmp_path, team=lambda: record, members=members)
    snap = built.builder.view_state()
    assert snap.members[0].status == "shutting-down"
    await task


def test_off_record_member_with_waiter_is_not_shutting_down(
    tmp_path: Path,
) -> None:
    """The shutting-down status is gated on the team being live; an off-record
    snapshot must ignore in-memory slots and never claim 'shutting-down'."""
    built = _builder(tmp_path, team=lambda: _record("live"))
    disk = _record("other", members=[TeammateMember(name="m1")])
    built.storage.team_config_path("other").write_text(
        disk.model_dump_json(),
        encoding="utf-8",
    )
    snap = built.builder.view_state("other")
    # No slot is consulted off-record, so status stays 'active'.
    assert snap.members[0].status == "active"
    assert snap.members[0].lifecycle_state == "unknown"


def test_live_member_inherits_task_tokens_and_model_override(
    tmp_path: Path,
) -> None:
    """A running member's row must reflect its task's accumulated tokens, last
    activity, and resolved model_spec so the dashboard mirrors live progress."""
    tasks = TasksStore()
    task = tasks.create("d", "p", model_spec="openai:gpt-4o")
    tasks.record_token_usage(task.id, input_tokens=7, output_tokens=3)
    task.progress.last_activity_at = 123.0
    record = _record(
        "t",
        members=[TeammateMember(name="m1", model_name="cfg:default")],
    )
    members = {"m1": Member(lifecycle_state="running", task_id=task.id)}
    built = _builder(
        tmp_path,
        team=lambda: record,
        members=members,
        tasks_store=tasks,
    )
    row = built.builder.view_state().members[0]
    assert row.tokens_used == 10
    assert row.last_active == 123.0
    assert row.model_spec == "openai:gpt-4o"


def test_live_member_empty_model_spec_keeps_configured_name(
    tmp_path: Path,
) -> None:
    """An empty resolved model_spec on the task must NOT clobber the member's
    configured model_name — the inherited default stays visible."""
    tasks = TasksStore()
    task = tasks.create("d", "p", model_spec="")
    record = _record(
        "t",
        members=[TeammateMember(name="m1", model_name="cfg:default")],
    )
    members = {"m1": Member(lifecycle_state="running", task_id=task.id)}
    built = _builder(
        tmp_path,
        team=lambda: record,
        members=members,
        tasks_store=tasks,
    )
    assert built.builder.view_state().members[0].model_spec == "cfg:default"


def test_live_member_with_missing_task_record_stays_zeroed(
    tmp_path: Path,
) -> None:
    """A slot pointing at a task id the store has dropped must degrade to zero
    tokens / no activity, never raise — a vanished task is a normal race."""
    record = _record("t", members=[TeammateMember(name="m1")])
    members = {"m1": Member(lifecycle_state="running", task_id="vanished")}
    built = _builder(tmp_path, team=lambda: record, members=members)
    row = built.builder.view_state().members[0]
    assert row.tokens_used == 0
    assert row.last_active is None


def test_transcript_count_survives_iterdir_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the transcripts dir errors mid-scan (e.g. disk yanked), the count must
    fall back to 0 instead of bubbling an OSError out of the snapshot."""
    record = _record("live")
    built = _builder(tmp_path, team=lambda: record)
    (built.storage.team_root("live") / "transcripts").mkdir(
        parents=True,
        exist_ok=True,
    )
    real_iterdir = Path.iterdir

    def boom(self: Path) -> Any:
        if self.name == "transcripts":
            raise OSError("disk gone")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", boom)
    assert built.builder.view_state().transcript_count == 0


def test_subagent_count_suppresses_storage_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing list_subagent_transcripts must be swallowed to a 0 count so the
    /team view still renders rather than dying on an infra hiccup."""
    record = _record("live")
    built = _builder(tmp_path, team=lambda: record)

    def boom() -> Any:
        raise RuntimeError("backend down")

    monkeypatch.setattr(built.storage, "list_subagent_transcripts", boom)
    assert built.builder.view_state().subagent_count == 0


def test_transcript_count_ignores_non_jsonl_and_subdirs(
    tmp_path: Path,
) -> None:
    """Only .jsonl files count as transcripts; stray notes / nested dirs must be
    excluded so the dashboard number matches actual teammate logs."""
    record = _record("live")
    built = _builder(tmp_path, team=lambda: record)
    tx_dir = built.storage.team_root("live") / "transcripts"
    tx_dir.mkdir(parents=True, exist_ok=True)
    (tx_dir / "scout.jsonl").write_text("x\n", encoding="utf-8")
    (tx_dir / "notes.txt").write_text("x\n", encoding="utf-8")
    (tx_dir / "nested").mkdir()
    assert built.builder.view_state().transcript_count == 1


def test_recent_messages_sorted_newest_first_and_capped(
    tmp_path: Path,
) -> None:
    """Recent messages must be globally newest-first across all recipients and
    capped at 10 so the panel shows the latest activity, not insertion order."""
    record = _record("live", members=[TeammateMember(name="m1")])
    built = _builder(tmp_path, team=lambda: record)
    mailbox = Mailbox(built.storage, "live")
    for i in range(14):
        mailbox.append(
            TeamMessage(
                msg_id=f"id-{i:02d}",
                sender="leader",
                recipient="m1",
                body=f"body-{i:02d}",
                sent_at=float(i),
            ),
        )
    snap = built.builder.view_state()
    assert len(snap.recent_messages) == 10
    sent = [m.sent_at for m in snap.recent_messages]
    assert sent == sorted(sent, reverse=True)
    assert snap.recent_messages[0].body == "body-13"


def test_view_state_is_idempotent_across_repeated_calls(
    tmp_path: Path,
) -> None:
    """Invoking view_state twice on the same unchanged state must yield an
    identical snapshot — it is a pure read and must not mutate or drift."""
    tasks = TasksStore()
    task = tasks.create("d", "p")
    tasks.record_token_usage(task.id, input_tokens=2, output_tokens=4)
    record = _record("t", members=[TeammateMember(name="m1")])
    members = {"m1": Member(lifecycle_state="running", task_id=task.id)}
    built = _builder(
        tmp_path,
        team=lambda: record,
        members=members,
        tasks_store=tasks,
    )
    first = built.builder.view_state()
    second = built.builder.view_state()
    assert first == second
    assert first.members == second.members
