"""TeamManager — lifecycle, persistence, send/recv routing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.config.schema import AuraConfig
from aura.core import llm
from aura.core.abort import AbortController
from aura.core.permissions.rule import Rule
from aura.core.permissions.safety import DEFAULT_SAFETY
from aura.core.permissions.session import RuleSet
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.store import TasksStore
from aura.core.teams.manager import TeamError, TeamManager
from aura.core.teams.types import TEAM_LEADER_NAME, TeamRecord
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolMetadata
from tests.conftest import FakeChatModel, FakeTurn


def _cfg() -> AuraConfig:
    # ``teams.enabled=True`` from v0.18 — the gate (claude-code parity
    # with isAgentSwarmsEnabled()) defaults to False; without this flag,
    # spawning a teammate Agent via SubagentFactory would call
    # ``join_team`` which now raises when teams are disabled.
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "teams": {"enabled": True},
    })


class _EchoParams(BaseModel):
    value: str = "x"


_NON_DESTRUCTIVE_META = ToolMetadata(
    is_read_only=False,
    is_destructive=False,
    is_concurrency_safe=False,
    rule_matcher=None,
    args_preview=None,
    timeout_sec=None,
)


class _AllowedTool(BaseTool):
    name: str = "team_allowed_tool"
    description: str = "test-only allowed tool"
    args_schema: type[BaseModel] = _EchoParams
    aura_metadata: ToolMetadata = _NON_DESTRUCTIVE_META

    def _run(self, value: str = "x") -> str:
        return value


class _AskTool(BaseTool):
    name: str = "team_ask_tool"
    description: str = "test-only ask-path tool"
    args_schema: type[BaseModel] = _EchoParams
    aura_metadata: ToolMetadata = _NON_DESTRUCTIVE_META

    def _run(self, value: str = "x") -> str:
        return value


def _factory(*, parent_ruleset: RuleSet | None = None) -> SubagentFactory:
    return SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        parent_ruleset=parent_ruleset or RuleSet(),
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
    tmp_path: Path,
    *,
    runtime_runner: Any = _no_runtime,
    factory: SubagentFactory | None = None,
    running_aborts: dict[str, AbortController] | None = None,
) -> tuple[TeamManager, SessionStorage]:
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader_stub(storage)
    return TeamManager(
        leader=leader,
        storage=storage,
        factory=factory or _factory(),
        running_aborts=running_aborts if running_aborts is not None else {},
        tasks_store=TasksStore(),
        runtime_runner=runtime_runner,
    ), storage


async def _wait_for_teammate_terminal(mgr: TeamManager) -> Any:
    record = mgr._tasks_store.list(kind="teammate")[0]
    await asyncio.wait_for(
        mgr._tasks_store.terminal_event(record.id).wait(),
        timeout=1,
    )
    return record


class _FakePaneHandle:
    pane_id = "pane-1"

    def __init__(self) -> None:
        self.force_killed = False

    async def shutdown(self, *, timeout_sec: float = 5.0) -> bool:
        return True

    async def force_kill(self) -> None:
        self.force_killed = True

    def is_alive(self) -> bool:
        return not self.force_killed


class _FakePaneBackend:
    async def spawn(self, **_kwargs: Any) -> _FakePaneHandle:
        return _FakePaneHandle()


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


@pytest.mark.asyncio
async def test_teammate_task_completes_on_natural_runtime_return(
    tmp_path: Path,
) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")

    mgr.add_member("alice")

    record = await _wait_for_teammate_terminal(mgr)
    assert record.status == "completed"


@pytest.mark.asyncio
async def test_teammate_task_fails_on_runtime_exception(
    tmp_path: Path,
) -> None:
    async def broken_runtime(**_kwargs: Any) -> None:
        raise RuntimeError("runtime exploded")

    mgr, _ = _mgr(tmp_path, runtime_runner=broken_runtime)
    mgr.create_team("alpha")

    mgr.add_member("alice")

    record = await _wait_for_teammate_terminal(mgr)
    assert record.status == "failed"
    assert record.error == "RuntimeError: runtime exploded"


@pytest.mark.asyncio
async def test_force_remove_marks_teammate_task_cancelled(
    tmp_path: Path,
) -> None:
    async def parked_runtime(**kwargs: Any) -> None:
        await kwargs["abort"].signal.wait()

    mgr, _ = _mgr(tmp_path, runtime_runner=parked_runtime)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    await asyncio.sleep(0)

    mgr.remove_member("alice", force=True)
    await asyncio.sleep(0)

    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.status == "cancelled"
    assert mgr._member_task_ids == {}
    assert mgr._running_aborts == {}
    assert mgr._stop_events == {}


@pytest.mark.asyncio
async def test_parent_abort_marks_teammate_task_cancelled(
    tmp_path: Path,
) -> None:
    captured: dict[str, AbortController] = {}

    async def abortable_runtime(**kwargs: Any) -> None:
        captured["abort"] = kwargs["abort"]
        await kwargs["abort"].signal.wait()

    mgr, _ = _mgr(tmp_path, runtime_runner=abortable_runtime)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    await asyncio.sleep(0)

    captured["abort"].abort("parent_abort")

    record = await _wait_for_teammate_terminal(mgr)
    assert record.status == "cancelled"


@pytest.mark.asyncio
async def test_add_member_propagates_explicit_model_to_task_and_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr, _ = _mgr(tmp_path)
    spawn = MagicMock(wraps=mgr._factory.spawn)
    monkeypatch.setattr(mgr._factory, "spawn", spawn)
    mgr.create_team("alpha")

    mgr.add_member("alice", model_name="openai:gpt-4o")
    await asyncio.sleep(0)

    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.model_spec == "openai:gpt-4o"
    assert spawn.call_args.kwargs["model_spec"] == "openai:gpt-4o"


@pytest.mark.asyncio
async def test_add_member_records_inherited_model_without_spawn_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr, _ = _mgr(tmp_path)
    spawn = MagicMock(wraps=mgr._factory.spawn)
    monkeypatch.setattr(mgr._factory, "spawn", spawn)
    mgr.create_team("alpha")

    mgr.add_member("alice")
    await asyncio.sleep(0)

    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.model_spec == mgr._factory.parent_model_spec
    assert spawn.call_args.kwargs.get("model_spec") is None


@pytest.mark.asyncio
async def test_add_member_teammate_uses_subagent_permission_contract(
    tmp_path: Path,
) -> None:
    parent_ruleset = RuleSet(rules=(Rule(tool="team_allowed_tool", content=None),))
    mgr, _ = _mgr(tmp_path, factory=_factory(parent_ruleset=parent_ruleset))
    mgr.create_team("alpha")

    mgr.add_member("alice")
    child = mgr._member_agents["alice"]
    try:
        from aura.schemas.permissions import Allow, Replace

        allowed = await child._hooks.run_pre_tool(
            tool=_AllowedTool(),
            args={"value": "x"},
            state=LoopState(),
        )
        assert isinstance(allowed, Allow)
        assert allowed.decision.allow is True
        assert allowed.decision.reason == "rule_allow"

        denied = await child._hooks.run_pre_tool(
            tool=_AskTool(),
            args={"value": "x"},
            state=LoopState(),
        )
        assert isinstance(denied, Replace)
        assert denied.decision.allow is False
        assert denied.decision.reason == "user_deny"
        assert "subagent_auto_deny" in (denied.result.error or "")
    finally:
        mgr.remove_member("alice", force=True)
        await asyncio.sleep(0)


def test_add_member_rejects_invalid_model_without_state_leak(
    tmp_path: Path,
) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")

    with pytest.raises(llm.UnknownModelSpecError):
        mgr.add_member("alice", model_name="ghost-provider:nope")

    assert mgr.list_members() == []
    assert mgr._tasks_store.list(kind="teammate") == []


def test_add_member_rejects_empty_model_without_state_leak(
    tmp_path: Path,
) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")

    with pytest.raises(llm.UnknownModelSpecError):
        mgr.add_member("alice", model_name="")

    assert mgr.list_members() == []
    assert mgr._tasks_store.list(kind="teammate") == []


@pytest.mark.asyncio
async def test_aadd_member_propagates_explicit_model_to_task_and_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aura.core.teams.backends.registry as registry

    monkeypatch.setattr(registry, "get_backend", lambda _backend_type: _FakePaneBackend())
    mgr, _ = _mgr(tmp_path)
    spawn = MagicMock(wraps=mgr._factory.spawn)
    monkeypatch.setattr(mgr._factory, "spawn", spawn)
    mgr.create_team("alpha")

    await mgr.aadd_member("alice", model_name="openai:gpt-4o", backend_type="pane")

    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.model_spec == "openai:gpt-4o"
    assert spawn.call_args.kwargs["model_spec"] == "openai:gpt-4o"


@pytest.mark.asyncio
async def test_aadd_member_records_inherited_model_without_spawn_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aura.core.teams.backends.registry as registry

    monkeypatch.setattr(registry, "get_backend", lambda _backend_type: _FakePaneBackend())
    mgr, _ = _mgr(tmp_path)
    spawn = MagicMock(wraps=mgr._factory.spawn)
    monkeypatch.setattr(mgr._factory, "spawn", spawn)
    mgr.create_team("alpha")

    await mgr.aadd_member("alice", backend_type="pane")

    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.model_spec == mgr._factory.parent_model_spec
    assert spawn.call_args.kwargs.get("model_spec") is None


@pytest.mark.asyncio
async def test_pane_force_remove_marks_teammate_task_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aura.core.teams.backends.registry as registry

    monkeypatch.setattr(registry, "get_backend", lambda _backend_type: _FakePaneBackend())
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    await mgr.aadd_member("alice", backend_type="pane")

    mgr.remove_member("alice", force=True)
    await asyncio.sleep(0)

    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.status == "cancelled"
    assert mgr._member_task_ids == {}
    assert mgr._running_aborts == {}
    assert mgr._stop_events == {}


@pytest.mark.asyncio
async def test_pane_session_cleanup_marks_teammate_task_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aura.core.teams.backends.registry as registry

    monkeypatch.setattr(registry, "get_backend", lambda _backend_type: _FakePaneBackend())
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    await mgr.aadd_member("alice", backend_type="pane")

    await mgr.cleanup_session_teams()

    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.status == "cancelled"


@pytest.mark.asyncio
async def test_aadd_member_rejects_invalid_model_without_state_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aura.core.teams.backends.registry as registry

    monkeypatch.setattr(registry, "get_backend", lambda _backend_type: _FakePaneBackend())
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")

    with pytest.raises(llm.UnknownModelSpecError):
        await mgr.aadd_member(
            "alice",
            model_name="ghost-provider:nope",
            backend_type="pane",
        )

    assert mgr.list_members() == []
    assert mgr._tasks_store.list(kind="teammate") == []


@pytest.mark.asyncio
async def test_aadd_member_rejects_empty_model_without_state_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aura.core.teams.backends.registry as registry

    monkeypatch.setattr(registry, "get_backend", lambda _backend_type: _FakePaneBackend())
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")

    with pytest.raises(llm.UnknownModelSpecError):
        await mgr.aadd_member("alice", model_name="", backend_type="pane")

    assert mgr.list_members() == []
    assert mgr._tasks_store.list(kind="teammate") == []


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
    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.status == "cancelled"


@pytest.mark.asyncio
async def test_session_cleanup_marks_teammate_task_cancelled(
    tmp_path: Path,
) -> None:
    async def parked_runtime(**kwargs: Any) -> None:
        await kwargs["abort"].signal.wait()

    mgr, _ = _mgr(tmp_path, runtime_runner=parked_runtime)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    await asyncio.sleep(0)

    await mgr.cleanup_session_teams()

    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.status == "cancelled"


@pytest.mark.asyncio
async def test_session_cleanup_preserves_unrelated_abort_controllers(
    tmp_path: Path,
) -> None:
    running_aborts = {"unrelated-task": AbortController()}

    async def parked_runtime(**kwargs: Any) -> None:
        await kwargs["abort"].signal.wait()

    mgr, _ = _mgr(
        tmp_path,
        runtime_runner=parked_runtime,
        running_aborts=running_aborts,
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    await asyncio.sleep(0)

    teammate_task_ids = set(mgr._member_task_ids.values())
    assert teammate_task_ids
    assert teammate_task_ids.issubset(running_aborts)

    await mgr.cleanup_session_teams()

    assert set(running_aborts) == {"unrelated-task"}


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
