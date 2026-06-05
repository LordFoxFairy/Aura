"""``/team`` slash command — dispatch verbs, error mapping."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from aura.application.commands.factory import build_default_registry
from aura.application.commands.registry import dispatch
from aura.application.commands.team import (
    TeamCommand,
    _AddUsageError,
    _ensure_manager,
    _format_age,
    _parse_add_args,
    _read_transcript_tail,
    _render_teammate,
    _render_view,
    _resolve_team_id,
)
from aura.application.commands.team import TeamCommand as CapabilityTeamCommand
from aura.application.session import AgentSession
from aura.application.teams.manager import TeamManager
from aura.application.teams.view_types import TeammateMemberStatus, TeamViewSnapshot
from aura.config.schema import AuraConfig
from aura.domain.team import TeammateMember, TeamMessage, TeamRecord
from aura.infrastructure.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel


def test_team_command_core_facade_points_at_capabilities_module() -> None:
    assert TeamCommand is CapabilityTeamCommand
    assert TeamCommand.__module__ == "aura.application.commands.team"


def _members(agent: AgentSession) -> list[TeammateMember]:
    """Read the live team's members through the leader's TeamManager.

    The manager is stored on a private attribute by ``_ensure_manager``;
    this helper centralizes the cast so the test bodies stay readable.
    """
    mgr = cast(TeamManager, agent._team_manager)
    return mgr.list_members()


def _agent(tmp_path: Path, *, teams_enabled: bool = True) -> AgentSession:
    # ``teams.enabled=True`` opens the gate so /team verbs reach handlers.
    cfg = AuraConfig.model_validate(
        {
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
            "teams": {"enabled": teams_enabled},
        }
    )
    return AgentSession(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "sessions.db"),
    )


@pytest.mark.asyncio
async def test_team_command_help_when_no_args(tmp_path: Path) -> None:
    cmd = TeamCommand()
    result = await cmd.handle("", _agent(tmp_path))
    assert result.handled is True
    assert "/team" in result.text
    assert "create" in result.text


@pytest.mark.asyncio
async def test_team_create_sets_active_team(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    result = await cmd.handle("create demo", agent)
    assert "team 'demo' created" in result.text
    assert agent.team is not None
    assert agent.state.slots.active_team == "demo"


@pytest.mark.asyncio
async def test_team_send_to_unknown_member_returns_error(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("send ghost hello", agent)
    assert "team error" in result.text
    assert "ghost" in result.text


@pytest.mark.asyncio
async def test_team_list_includes_active_marker(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("list", agent)
    assert "demo" in result.text
    # Active team is prefixed with '* '
    assert "* demo" in result.text


@pytest.mark.asyncio
async def test_team_help_subcommand(tmp_path: Path) -> None:
    cmd = TeamCommand()
    result = await cmd.handle("help", _agent(tmp_path))
    assert "subcommand" in result.text


@pytest.mark.asyncio
async def test_team_unknown_subcommand(tmp_path: Path) -> None:
    cmd = TeamCommand()
    result = await cmd.handle("zonk", _agent(tmp_path))
    assert "unknown subcommand" in result.text


@pytest.mark.asyncio
async def test_dispatch_team_through_registry(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    # ``TeamCommand`` only registers when the agent's config opts in.
    r = build_default_registry(agent)
    result = await dispatch("/team", agent, r)
    assert result.handled is True
    assert "subcommand" in result.text


@pytest.mark.asyncio
async def test_team_delete_clears_team(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    assert agent.team is not None
    result = await cmd.handle("delete", agent)
    assert "deleted" in result.text
    assert agent.team is None


@pytest.fixture
def _fake_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub OPENAI_API_KEY so the factory can build a model in tests.

    The ``/team add`` path runs the SubagentSpawner, which validates a
    provider's credential env var even when the resulting model never
    fires (these tests don't pump events). A literal placeholder is
    enough — no network is hit.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-fake-key")


@pytest.mark.asyncio
async def test_team_add_default_backend_is_in_process(
    tmp_path: Path,
    _fake_openai_key: None,
) -> None:
    """``/team add alice`` defaults to in_process — backend not surfaced."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("add alice", agent)
    assert "added" in result.text
    # backend suffix is suppressed for the default value (terse output).
    assert "backend=" not in result.text
    members = _members(agent)
    assert any(m.name == "alice" and m.backend_type == "in_process" for m in members)


@pytest.mark.asyncio
async def test_team_add_explicit_in_process_backend(
    tmp_path: Path,
    _fake_openai_key: None,
) -> None:
    """``--backend in_process`` is accepted explicitly + persisted."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("add bob --backend in_process", agent)
    assert "added" in result.text
    member = next(m for m in _members(agent) if m.name == "bob")
    assert member.backend_type == "in_process"


@pytest.mark.asyncio
async def test_team_add_pane_backend_outside_tmux_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_teams_registry: None,
) -> None:
    """``--backend pane`` without ``$TMUX`` surfaces a friendly error.

    The registry's ``pane_backend_available()`` walks ``$TMUX`` + the
    PATH; we strip ``$TMUX`` so the gate fires regardless of host env.
    """
    monkeypatch.delenv("TMUX", raising=False)
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("add carol --backend pane", agent)
    assert "error" in result.text.lower()
    assert "tmux" in result.text.lower()
    # No half-spawned member: the registry check fires before we touch
    # team state.
    assert all(m.name != "carol" for m in _members(agent))


@pytest.mark.asyncio
async def test_team_add_unknown_backend_rejected(tmp_path: Path) -> None:
    """Typoed backend value gives a usage hint, not a stack trace."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("add dan --backend zomg", agent)
    assert "unknown backend" in result.text
    assert "in_process" in result.text  # hint lists valid values
    assert all(m.name != "dan" for m in _members(agent))


@pytest.mark.asyncio
async def test_team_add_backend_flag_anywhere(
    tmp_path: Path,
    _fake_openai_key: None,
) -> None:
    """``--backend`` can precede positional args — flag-position-agnostic."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    # Flag interleaved between positional args.
    result = await cmd.handle(
        "add eve general-purpose --backend in_process",
        agent,
    )
    assert "added" in result.text
    member = next(m for m in _members(agent) if m.name == "eve")
    assert member.agent_type == "general-purpose"
    assert member.backend_type == "in_process"


@pytest.mark.asyncio
async def test_team_add_backend_missing_value(tmp_path: Path) -> None:
    """Trailing ``--backend`` with no value gives a usage hint."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("add frank --backend", agent)
    assert "usage" in result.text.lower()
    assert "backend" in result.text.lower()


@pytest.mark.parametrize(
    ("now", "then", "expected"),
    [
        (1000.0, None, "-"),
        (1000.0, 1000.0, "0s ago"),
        (1000.0, 1030.0, "0s ago"),  # future timestamp clamps to 0, never negative
        (1000.0, 941.0, "59s ago"),
        (1000.0, 940.0, "1m ago"),  # 60s bucket boundary
        (1000.0, 1000.0 - 3599, "59m ago"),
        (1000.0, 1000.0 - 3600, "1h ago"),  # 3600s boundary
        (1000.0, 1000.0 - 86399, "23h ago"),
        (1000.0, 1000.0 - 86400, "1d ago"),  # 86400s boundary
    ],
)
def test_format_age_buckets_and_clamps_future(
    now: float,
    then: float | None,
    expected: str,
) -> None:
    """Age buckets gate the /team table layout; a future or boundary value must not skew them."""
    assert _format_age(now, then) == expected


@pytest.mark.parametrize(
    ("rest", "positional", "backend"),
    [
        ("", [], "in_process"),
        ("alice", ["alice"], "in_process"),
        ("alice bob", ["alice", "bob"], "in_process"),
        ("--backend pane alice", ["alice"], "pane"),
        ("alice --backend pane", ["alice"], "pane"),  # --backend may trail positionals
        ("--backend in_process alice", ["alice"], "in_process"),
    ],
)
def test_parse_add_args_keeps_positionals_and_backend(
    rest: str,
    positional: list[str],
    backend: str,
) -> None:
    """Positional order must survive an interleaved --backend flag, or members get mis-named."""
    pos, be = _parse_add_args(rest)
    assert pos == positional
    assert be == backend


@pytest.mark.parametrize(
    "rest", ["--backend", "alice --backend", "--backend bogus", "a --backend zzz b"]
)
def test_parse_add_args_rejects_missing_or_unknown_backend(rest: str) -> None:
    """A dangling or unknown --backend must fail loudly, not silently default the backend."""
    with pytest.raises(_AddUsageError):
        _parse_add_args(rest)


def _write_disk_team(
    agent: AgentSession, *, team_id: str, name: str, body: str | None = None
) -> None:
    """Drop a config.json under a team slug without going through the live manager.

    ``body=None`` writes a valid TeamRecord; a non-None ``body`` is written
    verbatim so we can exercise the corrupt-config branch.
    """
    path = agent.storage.team_config_path(team_id)
    if body is None:
        record = TeamRecord(
            team_id=team_id, name=name, leader_session_id="sid-stub"
        )
        body = record.model_dump_json()
    path.write_text(body, encoding="utf-8")


def _snapshot(
    *,
    members: list[TeammateMemberStatus],
    messages: list[TeamMessage],
) -> TeamViewSnapshot:
    return TeamViewSnapshot(
        team_id="t1",
        name="Demo Team",
        members=members,
        recent_messages=messages,
        subagent_count=3,
        transcript_count=2,
    )


# ── _resolve_team_id (disk slug / display-name / corrupt fallback) ──────────


@pytest.mark.asyncio
async def test_resolve_team_id_matches_disk_slug(tmp_path: Path) -> None:
    """A bare slug already on disk resolves directly so /team enter <slug> works off-record."""
    agent = _agent(tmp_path)
    mgr = _ensure_manager(agent)
    _write_disk_team(agent, team_id="alpha", name="Alpha Squad")
    assert _resolve_team_id(name="alpha", manager=mgr, agent=agent) == "alpha"


@pytest.mark.asyncio
async def test_resolve_team_id_matches_disk_display_name(tmp_path: Path) -> None:
    """Display name resolves to its slug so users can /team enter the human label, not the slug."""
    agent = _agent(tmp_path)
    mgr = _ensure_manager(agent)
    _write_disk_team(agent, team_id="alpha", name="Alpha Squad")
    assert _resolve_team_id(name="Alpha Squad", manager=mgr, agent=agent) == "alpha"


@pytest.mark.asyncio
async def test_resolve_team_id_skips_corrupt_config(tmp_path: Path) -> None:
    """A corrupt config.json must not block resolving a sibling team by display name."""
    agent = _agent(tmp_path)
    mgr = _ensure_manager(agent)
    _write_disk_team(agent, team_id="broken", name="ignored", body="{not json")
    _write_disk_team(agent, team_id="good", name="Good Team")
    assert _resolve_team_id(name="Good Team", manager=mgr, agent=agent) == "good"


@pytest.mark.asyncio
async def test_resolve_team_id_unknown_returns_none(tmp_path: Path) -> None:
    """An unknown handle resolves to None so callers can map it to a 'not found' message."""
    agent = _agent(tmp_path)
    mgr = _ensure_manager(agent)
    assert _resolve_team_id(name="ghost", manager=mgr, agent=agent) is None


@pytest.mark.asyncio
async def test_resolve_team_id_prefers_live_team(tmp_path: Path) -> None:
    """The live team must win over disk lookup so an active session never reads a stale slug."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    mgr = _ensure_manager(agent)
    assert _resolve_team_id(name="demo", manager=mgr, agent=agent) == "demo"


# ── _render_view (members / messages / truncation boundaries) ───────────────


def test_render_view_no_members_shows_placeholder() -> None:
    """An empty roster must render '(no members)' rather than a bare header row."""
    text = _render_view(_snapshot(members=[], messages=[]))
    assert "(no members)" in text
    assert "recent messages (last 0)" in text
    assert "(none)" in text


def test_render_view_member_row_uses_inherits_for_null_model() -> None:
    """A null model_spec renders '(inherits)' so an inherited default stays legible in the table."""
    member = TeammateMemberStatus(
        name="alice",
        agent_type="general-purpose",
        model_spec=None,
        status="active",
        tokens_used=0,
        last_active=None,
        lifecycle_state="idle",
    )
    text = _render_view(_snapshot(members=[member], messages=[]))
    assert "alice" in text
    assert "(inherits)" in text
    # last_active=None renders as the '-' age placeholder, never '0s ago'.
    assert "-" in text


def test_render_view_truncates_long_message_body() -> None:
    """An over-80-char body is clipped to keep the message log one line per entry."""
    long_body = "x" * 200
    msg = TeamMessage(
        msg_id="m1", sender="leader", recipient="alice", body=long_body
    )
    text = _render_view(_snapshot(members=[], messages=[msg]))
    assert "x" * 77 + "..." in text
    assert "x" * 81 not in text


def test_render_view_flattens_multiline_message_body() -> None:
    """Newlines in a body must collapse to spaces so a message stays a single table row."""
    msg = TeamMessage(
        msg_id="m1", sender="leader", recipient="alice", body="line1\nline2"
    )
    text = _render_view(_snapshot(members=[], messages=[msg]))
    assert "line1 line2" in text


# ── _render_teammate (timestamp parse / malformed passthrough / clip) ───────


def test_render_teammate_empty_lines_shows_hint() -> None:
    """An empty transcript must explain the teammate hasn't run, not render a blank pane."""
    text = _render_teammate(member="alice", team_id="t1", lines=[], cap=50)
    assert "transcript empty" in text
    assert "showing last 0 of 50 max" in text


def test_render_teammate_formats_epoch_prefixed_line() -> None:
    """A numeric epoch prefix is rendered as HH:MM:SS so the transcript reads as a timeline."""
    text = _render_teammate(
        member="alice", team_id="t1", lines=["0 tool_call hello\n"], cap=50
    )
    assert "00:00:00" in text
    assert "tool_call" in text
    assert "hello" in text


def test_render_teammate_passes_malformed_line_through() -> None:
    """A non-epoch line is shown verbatim so a debug surface never hides raw transcript bytes."""
    text = _render_teammate(
        member="alice", team_id="t1", lines=["not-a-timestamp blob\n"], cap=50
    )
    assert "not-a-timestamp blob" in text
    assert "00:" not in text


def test_render_teammate_clips_oversized_event_body() -> None:
    """An over-200-char event body is clipped so one transcript event stays one line."""
    body = "z" * 400
    text = _render_teammate(
        member="alice", team_id="t1", lines=[f"0 event {body}\n"], cap=50
    )
    assert "z" * 197 + "..." in text
    assert "z" * 201 not in text


def test_render_teammate_bad_epoch_falls_back_to_raw_prefix() -> None:
    """An out-of-range epoch must fall back to the raw digits, not crash the transcript view."""
    text = _render_teammate(
        member="alice",
        team_id="t1",
        lines=["999999999999999999 event body\n"],
        cap=50,
    )
    assert "999999999999999999" in text
    assert "event" in text


# ── _read_transcript_tail (tail cap / missing file / OSError) ────────────────


def test_read_transcript_tail_returns_last_n_lines(tmp_path: Path) -> None:
    """Only the last ``cap`` lines load so a huge transcript never floods the pane or memory."""
    agent = _agent(tmp_path)
    path = agent.storage.team_transcript_path("t1", "alice")
    path.write_text("".join(f"line{i}\n" for i in range(10)), encoding="utf-8")
    tail = _read_transcript_tail(agent.storage, "t1", "alice", cap=3)
    assert tail == ["line7\n", "line8\n", "line9\n"]


def test_read_transcript_tail_missing_file_returns_empty(tmp_path: Path) -> None:
    """A never-written transcript yields [] so the view shows the empty hint, not an error."""
    agent = _agent(tmp_path)
    assert _read_transcript_tail(agent.storage, "t1", "ghost", cap=50) == []


def test_read_transcript_tail_swallows_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable transcript must degrade to [] so a transient FS fault never crashes /team."""
    agent = _agent(tmp_path)
    path = agent.storage.team_transcript_path("t1", "alice")
    path.write_text("data\n", encoding="utf-8")

    def _boom(*_a: object, **_k: object) -> object:
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "open", _boom)
    assert _read_transcript_tail(agent.storage, "t1", "alice", cap=50) == []


# ── dispatch: enter / leave / view / teammate / remove / members / send ─────


@pytest.mark.asyncio
async def test_enter_usage_when_no_name(tmp_path: Path) -> None:
    """Bare /team enter must print usage, not silently set a None active team."""
    result = await TeamCommand().handle("enter", _agent(tmp_path))
    assert "usage: /team enter" in result.text


@pytest.mark.asyncio
async def test_enter_unknown_team_reports_not_found(tmp_path: Path) -> None:
    """Entering a missing team must say 'not found', never set the slot to a phantom slug."""
    agent = _agent(tmp_path)
    result = await TeamCommand().handle("enter ghost", agent)
    assert "team not found: ghost" in result.text
    assert agent.state.slots.active_team is None


@pytest.mark.asyncio
async def test_enter_live_team_joins_leader(tmp_path: Path) -> None:
    """Re-entering the live team after leave must re-attach the leader, not just move a pointer."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    agent.leave_team()
    assert agent.team is None
    result = await cmd.handle("enter demo", agent)
    assert "leader joined" in result.text
    assert agent.team is not None


@pytest.mark.asyncio
async def test_enter_off_record_team_warns_unavailable(tmp_path: Path) -> None:
    """Entering a disk-only team sets the pointer but warns add/send are off until rehydrate."""
    agent = _agent(tmp_path)
    _ensure_manager(agent)
    _write_disk_team(agent, team_id="alpha", name="Alpha Squad")
    result = await TeamCommand().handle("enter alpha", agent)
    assert "off-record team" in result.text
    assert agent.state.slots.active_team == "alpha"


@pytest.mark.asyncio
async def test_leave_with_no_active_team(tmp_path: Path) -> None:
    """Leaving with nothing active is idempotent and reports '(no active team)', not an error."""
    result = await TeamCommand().handle("leave", _agent(tmp_path))
    assert "(no active team)" in result.text


@pytest.mark.asyncio
async def test_leave_detaches_joined_leader(tmp_path: Path) -> None:
    """Leaving a joined team must detach the leader AND clear the active-team slot in one step."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("leave", agent)
    assert "left team demo" in result.text
    assert "leader detached" in result.text
    assert agent.team is None
    assert agent.state.slots.active_team is None


@pytest.mark.asyncio
async def test_leave_pointer_only_without_join(tmp_path: Path) -> None:
    """Leaving an off-record (pointer-only) team clears the slot without a leader-detach claim."""
    agent = _agent(tmp_path)
    _ensure_manager(agent)
    _write_disk_team(agent, team_id="alpha", name="Alpha Squad")
    cmd = TeamCommand()
    await cmd.handle("enter alpha", agent)
    result = await cmd.handle("leave", agent)
    assert "left team alpha" in result.text
    assert "leader detached" not in result.text
    assert agent.state.slots.active_team is None


@pytest.mark.asyncio
async def test_view_no_active_team_prompts_for_name(tmp_path: Path) -> None:
    """/team view with nothing active must coach the user to pass or enter a name, not throw."""
    result = await TeamCommand().handle("view", _agent(tmp_path))
    assert "no active team" in result.text


@pytest.mark.asyncio
async def test_view_unknown_named_team_not_found(tmp_path: Path) -> None:
    """/team view <unknown> resolves to 'not found' rather than rendering an empty snapshot."""
    result = await TeamCommand().handle("view ghost", _agent(tmp_path))
    assert "team not found: ghost" in result.text


@pytest.mark.asyncio
async def test_view_active_team_renders_snapshot(tmp_path: Path) -> None:
    """Viewing the active team must render its id/name header so the user sees real team state."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("view", agent)
    assert result.kind == "view"
    assert "team: demo (id=demo)" in result.text
    assert "(no members)" in result.text


@pytest.mark.asyncio
async def test_teammate_usage_when_no_member(tmp_path: Path) -> None:
    """Bare /team teammate must print usage, never attempt a transcript read with empty name."""
    result = await TeamCommand().handle("teammate", _agent(tmp_path))
    assert "usage: /team teammate" in result.text


@pytest.mark.asyncio
async def test_teammate_no_active_team(tmp_path: Path) -> None:
    """/team teammate with nothing active must coach 'enter first', not dereference a None team."""
    result = await TeamCommand().handle("teammate alice", _agent(tmp_path))
    assert "no active team" in result.text


@pytest.mark.asyncio
async def test_teammate_unknown_member_lists_roster(tmp_path: Path) -> None:
    """An unknown member must list the actual roster so the user can correct the name."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("teammate ghost", agent)
    assert "member not found: ghost" in result.text
    assert "(none)" in result.text


@pytest.mark.asyncio
async def test_teammate_renders_transcript_for_known_member(
    tmp_path: Path, _fake_openai_key: None
) -> None:
    """A known member with a transcript renders its tail so the user can inspect that teammate."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("add alice", agent)
    path = agent.storage.team_transcript_path("demo", "alice")
    path.write_text("0 tool_call hi\n", encoding="utf-8")
    result = await cmd.handle("teammate alice", agent)
    assert result.kind == "view"
    assert "transcript: alice (team=demo)" in result.text
    assert "tool_call" in result.text


@pytest.mark.asyncio
async def test_remove_usage_when_no_name(tmp_path: Path) -> None:
    """Bare /team remove must print usage, never attempt to tear down an empty member name."""
    agent = _agent(tmp_path)
    await TeamCommand().handle("create demo", agent)
    result = await TeamCommand().handle("remove", agent)
    assert "usage: /team remove" in result.text


@pytest.mark.asyncio
async def test_members_empty_roster(tmp_path: Path) -> None:
    """An empty team must render '(no members)' so the user knows the roster, not a blank line."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("members", agent)
    assert "(no members)" in result.text


@pytest.mark.asyncio
async def test_members_lists_added_member(
    tmp_path: Path, _fake_openai_key: None
) -> None:
    """A populated roster lists each member with its active state so liveness is visible."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("add alice", agent)
    result = await cmd.handle("members", agent)
    assert result.kind == "view"
    assert "alice" in result.text
    assert "active" in result.text


@pytest.mark.asyncio
async def test_send_usage_when_body_missing(tmp_path: Path) -> None:
    """/team send with only a recipient must print usage, never deliver an empty-body message."""
    agent = _agent(tmp_path)
    await TeamCommand().handle("create demo", agent)
    result = await TeamCommand().handle("send leader", agent)
    assert "usage: /team send" in result.text


@pytest.mark.asyncio
async def test_send_to_leader_reports_message_id(tmp_path: Path) -> None:
    """A valid send to 'leader' reports the message id so the user can trace delivery."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    result = await cmd.handle("send leader hello there", agent)
    assert "sent 1 message(s)" in result.text
    assert "msg_id=" in result.text


@pytest.mark.asyncio
async def test_create_usage_when_no_name(tmp_path: Path) -> None:
    """Bare /team create must print usage, never slugify an empty name into a phantom team."""
    result = await TeamCommand().handle("create", _agent(tmp_path))
    assert "usage: /team create" in result.text


@pytest.mark.asyncio
async def test_delete_with_no_active_team(tmp_path: Path) -> None:
    """/team delete with nothing active reports '(no active team)', never rmtree a phantom dir."""
    result = await TeamCommand().handle("delete", _agent(tmp_path))
    assert "(no active team)" in result.text


@pytest.mark.asyncio
async def test_list_empty_when_no_teams(tmp_path: Path) -> None:
    """/team list with an empty storage root reports no teams rather than an empty body."""
    result = await TeamCommand().handle("list", _agent(tmp_path))
    assert "(no teams on disk)" in result.text


@pytest.mark.asyncio
async def test_create_then_delete_is_idempotent_on_second_delete(
    tmp_path: Path,
) -> None:
    """A repeated delete must stay safe: the second call sees no team and never re-rmtree's."""
    agent = _agent(tmp_path)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    first = await cmd.handle("delete", agent)
    assert "deleted" in first.text
    second = await cmd.handle("delete", agent)
    assert "(no active team)" in second.text
