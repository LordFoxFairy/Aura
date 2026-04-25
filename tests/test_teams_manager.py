"""TeamManager — lifecycle, persistence, send/recv routing."""

from __future__ import annotations

import asyncio
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
from aura.core.teams.manager import TeamError, TeamManager
from aura.core.teams.types import TEAM_LEADER_NAME, TeamRecord
from tests.conftest import FakeChatModel, FakeTurn


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
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


def _leader_stub(storage: SessionStorage, *, session_id: str = "leader-1") -> Any:
    """Lightweight stand-in for an Agent with the surface TeamManager touches."""
    leader = MagicMock()
    leader.session_id = session_id
    leader.cwd = Path.cwd()
    leader._storage = storage
    leader.join_team = MagicMock()
    leader.leave_team = MagicMock()
    return leader


async def _no_runtime(**_kwargs: Any) -> None:
    """Replacement for run_teammate — exits immediately so tests don't block."""
    return


def _mgr(
    tmp_path: Path, *, runtime_runner: Any = _no_runtime,
) -> tuple[TeamManager, SessionStorage]:
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader_stub(storage)
    return TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=runtime_runner,
    ), storage


def test_create_team_persists_config(tmp_path: Path) -> None:
    mgr, storage = _mgr(tmp_path)
    record = mgr.create_team("alpha")
    assert record.team_id == "alpha"
    cfg_path = storage.team_config_path("alpha")
    assert cfg_path.exists()
    loaded = TeamRecord.model_validate_json(cfg_path.read_text())
    assert loaded.team_id == "alpha"
    assert loaded.leader_session_id == "leader-1"


def test_create_team_rejects_double_create(tmp_path: Path) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    with pytest.raises(TeamError, match="one team per leader"):
        mgr.create_team("beta")


def test_create_team_collision_appends_suffix(tmp_path: Path) -> None:
    mgr, storage = _mgr(tmp_path)
    # Pre-create a team folder so 'alpha' is taken on disk.
    storage.team_root("alpha").joinpath("config.json").write_text("{}")
    record = mgr.create_team("alpha")
    assert record.team_id == "alpha-2"


@pytest.mark.asyncio
async def test_add_and_remove_member(tmp_path: Path) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    member = mgr.add_member("alice")
    assert member.name == "alice"
    assert any(m.name == "alice" for m in mgr.list_members())
    mgr.remove_member("alice", force=True)
    assert all(m.name != "alice" for m in mgr.list_members())
    # Wait for runtime tasks to settle.
    await asyncio.sleep(0)


def test_add_member_rejects_reserved_names(tmp_path: Path) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    # Reserved-name check fires BEFORE asyncio.create_task, so this works
    # in a sync test context.
    with pytest.raises(TeamError):
        mgr.add_member(TEAM_LEADER_NAME)
    with pytest.raises(TeamError):
        mgr.add_member("broadcast")


def test_add_member_rejects_invalid_slug(tmp_path: Path) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    with pytest.raises(TeamError, match="must match"):
        mgr.add_member("alice bob")  # space rejected — slug check fires before spawn


@pytest.mark.asyncio
async def test_add_member_rejects_duplicate(tmp_path: Path) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    with pytest.raises(TeamError, match="already exists"):
        mgr.add_member("alice")


@pytest.mark.asyncio
async def test_send_text_to_member(tmp_path: Path) -> None:
    mgr, storage = _mgr(tmp_path)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    sent = mgr.send(sender="leader", recipient="alice", body="hello")
    assert len(sent) == 1
    inbox = mgr.mailbox().read_all("alice")
    assert [m.body for m in inbox] == ["hello"]


@pytest.mark.asyncio
async def test_send_broadcast_fans_out(tmp_path: Path) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    mgr.add_member("bob")
    sent = mgr.send(sender="leader", recipient="broadcast", body="hi all")
    assert len(sent) == 2
    box = mgr.mailbox()
    assert [m.body for m in box.read_all("alice")] == ["hi all"]
    assert [m.body for m in box.read_all("bob")] == ["hi all"]


def test_send_unknown_recipient_errors(tmp_path: Path) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    with pytest.raises(TeamError, match="unknown recipient"):
        mgr.send(sender="leader", recipient="ghost", body="hi")


@pytest.mark.asyncio
async def test_remove_member_aborts_controller(tmp_path: Path) -> None:
    """Force-removing a member fires its AbortController so cascade triggers cleanup.

    Phase A.1: ``remove_member(force=True)`` keeps the synchronous
    abort semantics; the non-force path schedules an async waiter that
    only aborts after the shutdown_response timeout. The test pins the
    sync-abort contract for the force path.
    """
    captured: dict[str, AbortController] = {}

    async def capture_runner(**kwargs: Any) -> None:
        captured["abort"] = kwargs["abort"]
        # Wait until aborted, then return cleanly.
        try:
            await kwargs["abort"].signal.wait()
        except asyncio.CancelledError:
            return

    mgr, _ = _mgr(tmp_path, runtime_runner=capture_runner)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    # Yield so the runtime task starts and registers its abort.
    await asyncio.sleep(0)
    mgr.remove_member("alice", force=True)
    # The controller must be flipped.
    assert captured["abort"].aborted is True


@pytest.mark.asyncio
async def test_delete_team_clears_state(tmp_path: Path) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    mgr.delete_team()
    assert mgr.team is None
    assert mgr.list_members() == []


def test_load_round_trip(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "sessions.db")
    mgr1 = TeamManager(
        leader=_leader_stub(storage),
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=_no_runtime,
    )
    mgr1.create_team("alpha")
    mgr2 = TeamManager.load(
        leader=_leader_stub(storage),
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        team_id="alpha",
    )
    assert mgr2.team is not None
    assert mgr2.team.team_id == "alpha"


@pytest.mark.asyncio
async def test_send_rejects_oversize_body(tmp_path: Path) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    big = "x" * 5_000
    with pytest.raises(TeamError, match="MAX_BODY_CHARS"):
        mgr.send(sender="leader", recipient="alice", body=big)
