"""``/team enter / leave / view / teammate`` UX commands — V14 surface.

Covers:

- ``enter`` stamps ``state.custom["active_team_id"]`` and short-errors
  on unknown teams.
- ``leave`` clears the slot (and detaches the leader if joined).
- ``view`` returns a snapshot for the active team or an explicit name,
  and errors clearly when neither is set.
- ``teammate`` reads the last 50 transcript lines from the team's
  on-disk transcript and rejects unknown member names.
- ``Agent.clear_session`` resets the active-team pointer (the slot
  lives on ``LoopState.custom`` which ``reset()`` clears).

Tests bypass the runtime by passing ``runtime_runner=_no_runtime`` to
the manager so adding members never actually spawns a background loop —
the only thing we exercise here is the slash-command + snapshot path.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.commands.team import TeamCommand
from aura.core.persistence.storage import SessionStorage
from aura.core.teams.manager import TeamManager
from tests.conftest import FakeChatModel


@pytest.fixture(autouse=True)
def _stub_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a dummy OPENAI_API_KEY so the SubagentFactory's spawn —
    which goes through the real ``llm.create`` path inside ``add_member``
    — doesn't blow up on missing credentials. The FakeChatModel never
    actually calls out, but the key is resolved at spawn time before
    the model is even constructed.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-tests")


def _agent(tmp_path: Path) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "sessions.db"),
    )


async def _no_runtime(**_kwargs: Any) -> None:
    """Stand-in for run_teammate — exits immediately."""
    return


def _install_no_runtime_manager(agent: Agent) -> TeamManager:
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
        storage=agent._storage,
        factory=agent._subagent_factory,
        running_aborts=agent._running_aborts,
        tasks_store=agent._tasks_store,
        runtime_runner=_no_runtime,
    )
    agent._team_manager = mgr  # type: ignore[attr-defined]
    return mgr


@pytest.mark.asyncio
async def test_team_enter_sets_active_team(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    # /create has joined the leader; verify the entered slot was empty
    assert agent.state.custom.get("active_team_id") is None
    result = await cmd.handle("enter demo", agent)
    assert result.handled is True
    assert "entered team" in result.text
    assert agent.state.custom.get("active_team_id") == "demo"


@pytest.mark.asyncio
async def test_team_enter_unknown_team_errors(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    result = await cmd.handle("enter ghost", agent)
    assert "team not found" in result.text
    assert agent.state.custom.get("active_team_id") is None


@pytest.mark.asyncio
async def test_team_leave_clears_active_team(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("enter demo", agent)
    assert agent.state.custom.get("active_team_id") == "demo"
    result = await cmd.handle("leave", agent)
    assert "left team" in result.text
    assert agent.state.custom.get("active_team_id") is None


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
async def test_team_view_explicit_name_works_without_active(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create alpha", agent)
    # No /team enter — active_team_id stays None.
    assert agent.state.custom.get("active_team_id") is None
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
    transcript = agent._storage.team_transcript_path("demo", "scout")
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
    """Agent.clear_session calls LoopState.reset → custom.clear; the
    active-team pointer must NOT survive a /clear."""
    agent = _agent(tmp_path)
    _install_no_runtime_manager(agent)
    cmd = TeamCommand()
    await cmd.handle("create demo", agent)
    await cmd.handle("enter demo", agent)
    assert agent.state.custom.get("active_team_id") == "demo"
    agent.clear_session()
    assert agent.state.custom.get("active_team_id") is None


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
    transcript = agent._storage.team_transcript_path("demo", "scout")
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
    assert any(
        msg.recipient == "scout" and msg.body == "hi"
        for msg in snap.recent_messages
    )
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
