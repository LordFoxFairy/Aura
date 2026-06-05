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

from aura.application.loop_state import LoopState
from aura.application.session import AgentSession
from aura.application.tasks.spawn import SpawnContext, SubagentSpawner
from aura.application.tasks.store import TasksStore
from aura.application.teams.manager import TeamError, TeamManager
from aura.config.schema import AuraConfig
from aura.domain.abort import AbortController
from aura.domain.permission.rule import Rule
from aura.domain.permission.safety import DEFAULT_SAFETY
from aura.domain.permission.session import RuleSet
from aura.domain.team import TEAM_LEADER_NAME, TeamRecord
from aura.domain.tool import ToolMetadata
from aura.infrastructure import llm
from aura.infrastructure.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _cfg() -> AuraConfig:
    # Without ``teams.enabled=True`` the spawned teammate's ``join_team`` raises.
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


def _factory(*, parent_ruleset: RuleSet | None = None) -> SubagentSpawner[AgentSession]:
    return SubagentSpawner(
        SpawnContext(
            parent_config=_cfg(),
            parent_model_spec="openai:gpt-4o-mini",
            build_child=AgentSession,
            parent_ruleset=parent_ruleset or RuleSet(),
            parent_safety=DEFAULT_SAFETY,
            parent_mode_provider=lambda: "default",
            model_factory=lambda: FakeChatModel(
                turns=[FakeTurn(AIMessage(content="ack"))],
            ),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )


def _leader_stub(storage: SessionStorage, *, session_id: str = "leader-1") -> Any:
    """Lightweight stand-in for an AgentSession with the surface TeamManager touches."""
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
    factory: SubagentSpawner[AgentSession] | None = None,
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
    assert all(m.task_id is None for m in mgr._members.values())
    assert mgr._running_aborts == {}
    assert all(m.stop_event is None for m in mgr._members.values())


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
    child = mgr._members["alice"].agent
    assert child is not None
    try:
        from aura.domain.permission.outcome import Allow, Replace

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
    import aura.application.teams.manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_backend", lambda _backend_type: _FakePaneBackend())
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
    import aura.application.teams.manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_backend", lambda _backend_type: _FakePaneBackend())
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
    import aura.application.teams.manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_backend", lambda _backend_type: _FakePaneBackend())
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    await mgr.aadd_member("alice", backend_type="pane")

    mgr.remove_member("alice", force=True)
    await asyncio.sleep(0)

    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.status == "cancelled"
    assert all(m.task_id is None for m in mgr._members.values())
    assert mgr._running_aborts == {}
    assert all(m.stop_event is None for m in mgr._members.values())


@pytest.mark.asyncio
async def test_pane_session_cleanup_marks_teammate_task_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aura.application.teams.manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_backend", lambda _backend_type: _FakePaneBackend())
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
    import aura.application.teams.manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_backend", lambda _backend_type: _FakePaneBackend())
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
    import aura.application.teams.manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_backend", lambda _backend_type: _FakePaneBackend())
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
    """``remove_member(force=True)`` synchronously aborts the member's controller."""
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

    teammate_task_ids = {
        m.task_id for m in mgr._members.values() if m.task_id is not None
    }
    assert teammate_task_ids
    assert teammate_task_ids.issubset(running_aborts)

    await mgr.cleanup_session_teams()

    assert set(running_aborts) == {"unrelated-task"}


@pytest.mark.asyncio
async def test_send_rejects_oversize_body(tmp_path: Path) -> None:
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    big = "x" * 5_000
    with pytest.raises(TeamError, match="MAX_BODY_CHARS"):
        mgr.send(sender="leader", recipient="alice", body=big)


# --- inert-state guards: every mutating surface must reject "no team" ----------


def test_mailbox_requires_active_team(tmp_path: Path) -> None:
    """Reading the inbox before create_team must fail loudly, never silently empty."""
    mgr, _ = _mgr(tmp_path)
    with pytest.raises(TeamError, match="no team is active"):
        mgr.mailbox()


def test_send_requires_active_team(tmp_path: Path) -> None:
    """Routing a message with no team must raise, not drop the payload on the floor."""
    mgr, _ = _mgr(tmp_path)
    with pytest.raises(TeamError, match="no team is active"):
        mgr.send(sender="leader", recipient="alice", body="hi")


def test_remove_member_requires_active_team(tmp_path: Path) -> None:
    """Teardown with no team is a caller bug, surfaced as TeamError not a no-op."""
    mgr, _ = _mgr(tmp_path)
    with pytest.raises(TeamError, match="no team is active"):
        mgr.remove_member("alice")


@pytest.mark.asyncio
async def test_aremove_member_requires_active_team(tmp_path: Path) -> None:
    """Async teardown shares the leader-state guard with its sync sibling."""
    mgr, _ = _mgr(tmp_path)
    with pytest.raises(TeamError, match="no team is active"):
        await mgr.aremove_member("alice")


def test_remove_absent_member_errors(tmp_path: Path) -> None:
    """Removing a name never added must fail; a phantom teardown would hide bugs."""
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    with pytest.raises(TeamError, match="not found in team"):
        mgr.remove_member("ghost")


# --- create/delete edge cases --------------------------------------------------


def test_create_team_rejects_unslugifiable_name(tmp_path: Path) -> None:
    """A name with zero slug chars cannot key storage; reject before persisting."""
    mgr, _ = _mgr(tmp_path)
    with pytest.raises(TeamError, match="no slugifiable characters"):
        mgr.create_team("///")


def test_create_team_collision_skips_taken_suffixes(tmp_path: Path) -> None:
    """Suffix search must step past every occupied slot, not collide on the first."""
    mgr, storage = _mgr(tmp_path)
    for tid in ("alpha", "alpha-2", "alpha-3"):
        storage.team_root(tid).joinpath("config.json").write_text("{}")
    record = mgr.create_team("alpha")
    assert record.team_id == "alpha-4"


def test_delete_team_without_team_is_noop(tmp_path: Path) -> None:
    """Idempotent teardown: deleting when no team exists must not raise."""
    mgr, _ = _mgr(tmp_path)
    mgr.delete_team()
    assert mgr.team is None


@pytest.mark.asyncio
async def test_delete_team_is_idempotent(tmp_path: Path) -> None:
    """A second delete on an already-cleared team stays a safe no-op."""
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    mgr.delete_team()
    mgr.delete_team()
    assert mgr.team is None
    assert mgr.list_members() == []


# --- member add error / limit paths --------------------------------------------


@pytest.mark.asyncio
async def test_add_member_rejects_at_max_members(tmp_path: Path) -> None:
    """The MAX_MEMBERS cap protects mailbox fan-out; the (n+1)th add must reject."""
    from aura.domain.team import MAX_MEMBERS

    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    for i in range(MAX_MEMBERS):
        mgr.add_member(f"m{i}")
        await asyncio.sleep(0)
    with pytest.raises(TeamError, match="MAX_MEMBERS"):
        mgr.add_member("overflow")


def test_add_member_unavailable_backend_does_not_leak_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend that cannot start must abort before any roster/task mutation."""
    import aura.application.teams.manager as mgr_mod
    from aura.infrastructure.teams.registry import BackendUnavailable

    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")

    def _unavailable(_backend_type: Any) -> Any:
        raise BackendUnavailable("no tmux in this env")

    monkeypatch.setattr(mgr_mod, "get_backend", _unavailable)
    with pytest.raises(TeamError, match="no tmux in this env"):
        mgr.add_member("alice")
    assert mgr.list_members() == []
    assert mgr._tasks_store.list(kind="teammate") == []


def test_sync_add_member_rejects_pane_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sync add_member cannot dispatch a pane; it must steer callers to aadd_member."""
    import aura.application.teams.manager as mgr_mod

    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    monkeypatch.setattr(mgr_mod, "get_backend", lambda _bt: _FakePaneBackend())
    with pytest.raises(TeamError, match="aadd_member"):
        mgr.add_member("alice", backend_type="pane")


@pytest.mark.asyncio
async def test_default_in_process_runner_dispatches_real_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no test runner is injected, add_member must route through the real backend."""
    async def _inert(**_kwargs: Any) -> None:
        return

    monkeypatch.setattr(
        "aura.infrastructure.teams.in_process.run_teammate", _inert,
    )
    mgr, _ = _mgr(tmp_path, runtime_runner=None)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    record = await _wait_for_teammate_terminal(mgr)
    assert record.status == "completed"


@pytest.mark.asyncio
async def test_pane_spawn_stamping_pane_id_triggers_repersist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pane backend that stamps tmux_pane_id during spawn must re-persist the record."""
    import aura.application.teams.manager as mgr_mod

    class _StampingBackend:
        async def spawn(self, **kwargs: Any) -> _FakePaneHandle:
            member: Any = kwargs["member"]
            member.tmux_pane_id = "%42"
            return _FakePaneHandle()

    monkeypatch.setattr(mgr_mod, "get_backend", lambda _bt: _StampingBackend())
    mgr, storage = _mgr(tmp_path)
    mgr.create_team("alpha")
    member = await mgr.aadd_member("alice", backend_type="pane")
    assert member.tmux_pane_id == "%42"
    loaded = TeamRecord.model_validate_json(
        storage.team_config_path("alpha").read_text(),
    )
    assert any(m.tmux_pane_id == "%42" for m in loaded.members)


# --- graceful shutdown (aremove_member) ----------------------------------------


@pytest.mark.asyncio
async def test_graceful_remove_acked_terminates_cleanly(tmp_path: Path) -> None:
    """An acked shutdown returns True and reaches the 'terminated' lifecycle path."""
    async def parked(**kwargs: Any) -> None:
        await kwargs["abort"].signal.wait()

    mgr, _ = _mgr(tmp_path, runtime_runner=parked)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    await asyncio.sleep(0)

    waiter = asyncio.ensure_future(mgr.aremove_member("alice", timeout_sec=1.0))
    await asyncio.sleep(0)
    mgr.confirm_shutdown("alice")
    acked = await asyncio.wait_for(waiter, timeout=2)
    assert acked is True
    assert all(m.name != "alice" for m in mgr.list_members())


@pytest.mark.asyncio
async def test_graceful_remove_times_out_then_force_kills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ack within the grace window must force-kill and return False, never hang."""
    async def parked(**kwargs: Any) -> None:
        await kwargs["abort"].signal.wait()

    mgr, _ = _mgr(tmp_path, runtime_runner=parked)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    await asyncio.sleep(0)

    # Drive the timeout branch deterministically rather than racing a wall clock.
    async def _instant_timeout(_awaitable: Any, *, timeout: float) -> bool:
        del timeout
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", _instant_timeout)
    acked = await mgr.aremove_member("alice", timeout_sec=99.0)
    assert acked is False
    assert all(m.name != "alice" for m in mgr.list_members())
    record = mgr._tasks_store.list(kind="teammate")[0]
    assert record.status == "cancelled"


@pytest.mark.asyncio
async def test_sync_remove_with_loop_schedules_waiter(tmp_path: Path) -> None:
    """Inside a loop, non-force remove must defer to a tracked async waiter task."""
    async def parked(**kwargs: Any) -> None:
        await kwargs["abort"].signal.wait()

    mgr, _ = _mgr(tmp_path, runtime_runner=parked)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    await asyncio.sleep(0)

    # No timeout supplied: confirm the ack so the waiter resolves without a clock race.
    mgr.remove_member("alice", timeout_sec=99.0)
    waiter = mgr._members["alice"].shutdown_waiter
    assert waiter is not None
    await asyncio.sleep(0)
    mgr.confirm_shutdown("alice")
    await waiter
    await asyncio.sleep(0)
    slot = mgr._members.get("alice")
    assert slot is None or slot.shutdown_waiter is None


# --- confirm_shutdown no-op guards ---------------------------------------------


def test_confirm_shutdown_unknown_member_is_noop(tmp_path: Path) -> None:
    """An ack for a name with no slot must be ignored, not crash the leader."""
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    mgr.confirm_shutdown("ghost")


@pytest.mark.asyncio
async def test_confirm_shutdown_without_pending_ack_is_noop(tmp_path: Path) -> None:
    """A member present but not draining has no ack future; confirm must be inert."""
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    mgr.add_member("alice")
    mgr.confirm_shutdown("alice")
    assert any(m.name == "alice" for m in mgr.list_members())


# --- send routing edge cases ---------------------------------------------------


def test_send_to_leader_recipient(tmp_path: Path) -> None:
    """A teammate replying to the leader routes to exactly one leader-addressed msg."""
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    sent = mgr.send(sender="alice", recipient=TEAM_LEADER_NAME, body="done")
    assert len(sent) == 1
    assert sent[0].recipient == TEAM_LEADER_NAME


def test_broadcast_to_empty_team_errors(tmp_path: Path) -> None:
    """Broadcasting with zero members is a caller error, not a silent fan-out to none."""
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    with pytest.raises(TeamError, match="no members"):
        mgr.send(sender="leader", recipient="broadcast", body="anyone?")


def test_send_rejects_whitespace_only_body(tmp_path: Path) -> None:
    """A body of only whitespace carries no signal and must be rejected."""
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    with pytest.raises(TeamError, match="non-whitespace"):
        mgr.send(sender="leader", recipient=TEAM_LEADER_NAME, body="   \t\n")


def test_post_message_lands_in_mailbox(tmp_path: Path) -> None:
    """The public post_message surface must append to the inbox like internal _post."""
    from aura.domain.team import TeamMessage

    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    mgr.post_message(TeamMessage(
        msg_id="m1",
        sender=TEAM_LEADER_NAME,
        recipient=TEAM_LEADER_NAME,
        body="ping",
        kind="text",
    ))
    inbox = mgr.mailbox().read_all(TEAM_LEADER_NAME)
    assert [m.body for m in inbox] == ["ping"]


# --- cleanup_session_teams edge cases ------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_with_no_session_teams_is_noop(tmp_path: Path) -> None:
    """Cleanup before any team was created must short-circuit, not iterate empties."""
    mgr, _ = _mgr(tmp_path)
    await mgr.cleanup_session_teams()
    assert mgr.team is None


@pytest.mark.asyncio
async def test_cleanup_tolerates_missing_team_dir(tmp_path: Path) -> None:
    """A team dir already gone (race with external delete) must not abort cleanup."""
    import shutil as _shutil

    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    _shutil.rmtree(mgr.storage.team_root("alpha"))
    await mgr.cleanup_session_teams()
    assert mgr.team is None
    assert mgr._session_created_teams == set()


@pytest.mark.asyncio
async def test_cleanup_oserror_journals_and_keeps_session_team(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An rmtree OSError must be journaled and the team kept, not silently dropped."""
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    events: list[str] = []
    monkeypatch.setattr(
        "aura.application.teams.manager.journal.write",
        lambda kind, **_fields: events.append(kind),
    )

    def _rmtree_boom(_path: Any, *_args: Any, **_kwargs: Any) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(
        "aura.application.teams.manager.shutil.rmtree", _rmtree_boom,
    )
    await mgr.cleanup_session_teams()
    assert "team_session_cleanup_error" in events
    assert mgr._session_created_teams == {"alpha"}


# --- _persist failure path -----------------------------------------------------


def test_persist_oserror_is_journaled_not_raised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient FS write failure must journal and keep the agent alive, not crash."""
    mgr, _ = _mgr(tmp_path)
    mgr.create_team("alpha")
    events: list[str] = []
    monkeypatch.setattr(
        "aura.application.teams.manager.journal.write",
        lambda kind, **_fields: events.append(kind),
    )
    original_open = Path.open

    def _open_boom(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.suffix == ".tmp":
            raise OSError("disk full")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _open_boom)
    mgr._persist()
    assert "team_persist_failed" in events
