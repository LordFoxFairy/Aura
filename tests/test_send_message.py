"""SendMessage tool — outside-team error, recipient validation, fan-out."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from aura.config.schema import AuraConfig
from aura.core.permissions.safety import DEFAULT_SAFETY
from aura.core.permissions.session import RuleSet
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.store import TasksStore
from aura.core.teams.manager import TeamManager
from aura.schemas.tool import ToolError
from aura.tools.send_message import SendMessage
from tests.conftest import FakeChatModel, FakeTurn


def _cfg() -> AuraConfig:
    # ``teams.enabled=True`` from v0.18 — without it, the spawned
    # teammate Agent's ``join_team`` (called inside SubagentFactory.spawn
    # for add_member) raises because the gate (claude-code parity with
    # isAgentSwarmsEnabled()) defaults to False.
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


async def _no_runtime(**_kwargs: Any) -> None:
    return


def _leader(storage: SessionStorage) -> Any:
    leader = MagicMock()
    leader.session_id = "leader-1"
    leader.cwd = Path.cwd()
    leader._storage = storage
    leader.join_team = MagicMock()
    leader.leave_team = MagicMock()
    return leader


@pytest.mark.asyncio
async def test_send_message_outside_team_raises(tmp_path: Path) -> None:
    agent = MagicMock()
    agent.team = None
    tool = SendMessage(agent=agent)
    with pytest.raises(ToolError, match="not in a team"):
        await tool._arun(to="alice", body="hi")


@pytest.mark.asyncio
async def test_send_message_unknown_recipient_raises(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader(storage)
    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=_no_runtime,
    )
    mgr.create_team("alpha")
    leader.team = mgr
    tool = SendMessage(agent=leader)
    with pytest.raises(ToolError, match="unknown recipient"):
        await tool._arun(to="ghost", body="hi")


@pytest.mark.asyncio
async def test_send_message_to_member_routes_to_mailbox(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader(storage)
    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=_no_runtime,
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    leader.team = mgr
    leader._team_member_name = None  # leader has no member name
    tool = SendMessage(agent=leader)
    result = await tool._arun(to="alice", body="please scan")
    assert result["recipient"] == "alice"
    assert result["sender"] == "leader"
    assert result["fanout"] == 1
    inbox = mgr.mailbox().read_all("alice")
    assert [m.body for m in inbox] == ["please scan"]


@pytest.mark.asyncio
async def test_send_message_sender_is_team_member_when_set(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader(storage)
    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=_no_runtime,
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    teammate = MagicMock()
    teammate.team = mgr
    teammate._team_member_name = "alice"
    tool = SendMessage(agent=teammate)
    result = await tool._arun(to="leader", body="here is my report")
    assert result["sender"] == "alice"
    assert result["recipient"] == "leader"
