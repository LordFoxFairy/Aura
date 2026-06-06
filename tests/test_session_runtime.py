"""Phase 1 §5 — :class:`SessionRuntime` lifecycle in isolation.

The whole point of extracting :class:`SessionRuntime` from the AgentSession
god object is that lifecycle behaviour can be exercised WITHOUT
constructing a full AgentSession (no LangChain model, no HookChain, no
Context). These tests assert that contract directly: every case here
constructs only :class:`SessionStorage` + :class:`SessionRuntime`.

Tests cover the six lifecycle entry points called out in the Phase 1
plan: init, save, load, resume, clear, close — plus the streaming
buffer + notification queue helpers that ride alongside.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.application.hooks import HookChain
from aura.application.runtime.session import SessionRuntime
from aura.application.session import AgentSession
from aura.config.schema import AuraConfig
from aura.domain.abort import AbortController, AbortException
from aura.domain.events import AgentEvent, Final
from aura.domain.permission.rule import Rule
from aura.domain.permission.session import SessionRuleSet
from aura.domain.state_values import ReadCarryover, ReadRecord
from aura.domain.task import TaskNotification, TaskRecord
from aura.domain.team import TeamMessage, TeamMessageKind, TeamRecord
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.wire.events import CoordinationEvent, WireEvent
from aura.tools.base import build_tool
from aura.tools.web_fetch import set_default_model_factory
from tests.conftest import FakeChatModel, FakeTurn


@pytest.fixture
def storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "aura.db", cwd=tmp_path)


def test_init_minimal_args(storage: SessionStorage) -> None:
    """Bare SessionRuntime — no log dir, no rules, no inherited reads.

    Defaults: empty buffer, empty notification queue, no log path,
    SessionStart not yet fired."""
    rt = SessionRuntime(storage=storage, session_id="s-init")
    assert rt.session_id == "s-init"
    assert rt.storage is storage
    assert rt.session_log_path is None
    assert rt.session_rules is None
    assert rt.session_start_fired is False
    assert rt.partial_assistant_text == ""
    assert rt.pending_notifications == ()
    assert rt.carryover is None


def test_init_with_session_log_dir_creates_path(
    storage: SessionStorage, tmp_path: Path,
) -> None:
    """``session_log_dir`` is mkdir'd and the per-session JSONL path
    is computed as ``<dir>/<session_id>.jsonl``."""
    log_dir = tmp_path / "logs" / "nested"
    rt = SessionRuntime(
        storage=storage, session_id="s-1", session_log_dir=log_dir,
    )
    assert log_dir.is_dir()
    assert rt.session_log_path == log_dir / "s-1.jsonl"


def test_init_holds_session_rules_reference(
    storage: SessionStorage,
) -> None:
    """The rules object is held by reference — :meth:`clear` calls
    ``.clear()`` on the same instance the caller passed in."""
    rules = SessionRuleSet()
    rules.add(Rule(tool="read_file", content=None))
    rt = SessionRuntime(
        storage=storage, session_id="s-r", session_rules=rules,
    )
    assert rt.session_rules is rules
    assert len(rules.rules()) == 1


def test_save_then_load_roundtrips_history(storage: SessionStorage) -> None:
    """Round-trip a couple of messages through the runtime — proves
    the storage delegation actually persists + reloads correctly
    without going through the AgentSession layer."""
    rt = SessionRuntime(storage=storage, session_id="s-save")
    history = [
        HumanMessage(content="hello"),
        AIMessage(content="hi back"),
    ]
    rt.save_history(history)
    loaded = rt.load_history()
    assert len(loaded) == 2
    assert loaded[0].content == "hello"
    assert loaded[1].content == "hi back"


def test_load_empty_session_returns_empty_list(
    storage: SessionStorage,
) -> None:
    rt = SessionRuntime(storage=storage, session_id="s-empty")
    assert rt.load_history() == []


def test_resume_swaps_session_id_and_returns_message_count(
    storage: SessionStorage,
) -> None:
    """:meth:`resume` flips the live session_id, returns the row count,
    and drops the SessionStart re-arm flag."""
    # Seed two distinct sessions.
    rt = SessionRuntime(storage=storage, session_id="s-a")
    rt.save_history([HumanMessage(content="a-only")])
    rt_b = SessionRuntime(storage=storage, session_id="s-b")
    rt_b.save_history([
        HumanMessage(content="b-1"),
        AIMessage(content="b-2"),
    ])

    # Mark start fired + buffer some text on rt — resume should clear
    # both so the resumed session feels fresh.
    rt.mark_session_start_fired()
    rt.buffer_partial_assistant_text("partial")
    rt.enqueue_task_notification(TaskNotification(
        task_id="t1", status="completed", summary=None, description="x",
    ))

    count = rt.resume("s-b")
    assert count == 2
    assert rt.session_id == "s-b"
    assert rt.session_start_fired is False
    assert rt.partial_assistant_text == ""
    assert rt.pending_notifications == ()
    # And the loaded rows are the b session, not a's leftovers.
    loaded = rt.load_history()
    assert [m.content for m in loaded] == ["b-1", "b-2"]


def test_resume_unknown_session_raises_keyerror(
    storage: SessionStorage,
) -> None:
    rt = SessionRuntime(storage=storage, session_id="s-x")
    with pytest.raises(KeyError):
        rt.resume("does-not-exist")


def test_resume_retargets_session_log_path(
    storage: SessionStorage, tmp_path: Path,
) -> None:
    """When a log dir was wired, resume re-points the JSONL path to
    the resumed session_id so journal scope routes correctly."""
    log_dir = tmp_path / "logs"
    rt = SessionRuntime(
        storage=storage, session_id="s-1", session_log_dir=log_dir,
    )
    other = SessionRuntime(storage=storage, session_id="s-2")
    other.save_history([HumanMessage(content="hi")])

    rt.resume("s-2")
    assert rt.session_log_path == log_dir / "s-2.jsonl"


def test_clear_drops_history_buffers_notifications_and_rules(
    storage: SessionStorage,
) -> None:
    """:meth:`clear` is the runtime side of /clear — drops persisted
    history, clears partial buffer + notification queue, re-arms
    SessionStart, and wipes session rules."""
    rules = SessionRuleSet()
    rules.add(Rule(tool="read_file", content=None))
    rt = SessionRuntime(
        storage=storage, session_id="s-c", session_rules=rules,
    )
    rt.save_history([HumanMessage(content="will-be-wiped")])
    rt.mark_session_start_fired()
    rt.buffer_partial_assistant_text("buffered")
    rt.enqueue_task_notification(TaskNotification(
        task_id="t", status="completed", summary=None, description="d",
    ))
    # Sanity — preconditions hold.
    assert len(rt.load_history()) == 1
    assert rt.session_start_fired is True

    rt.clear()

    assert rt.load_history() == []
    assert rt.session_start_fired is False
    assert rt.partial_assistant_text == ""
    assert rt.pending_notifications == ()
    assert rules.rules() == ()  # session rules were cleared in place
    assert rt.carryover is None


def test_clear_without_session_rules_is_no_op_on_rules(
    storage: SessionStorage,
) -> None:
    """No rules wired → clear must still succeed, not crash."""
    rt = SessionRuntime(storage=storage, session_id="s-norules")
    rt.save_history([HumanMessage(content="x")])
    rt.clear()  # should not raise
    assert rt.load_history() == []


def test_close_storage_is_idempotent(storage: SessionStorage) -> None:
    """:meth:`close_storage` closes the SQLite handle; calling twice
    must not raise (lifecycle is best-effort and may run on every
    teardown path)."""
    rt = SessionRuntime(storage=storage, session_id="s-close")
    rt.close_storage()
    rt.close_storage()  # idempotent — must not raise


def test_buffer_partial_assistant_text_accumulates(
    storage: SessionStorage,
) -> None:
    rt = SessionRuntime(storage=storage, session_id="s-buf")
    rt.buffer_partial_assistant_text("hello ")
    rt.buffer_partial_assistant_text("world")
    assert rt.partial_assistant_text == "hello world"


def test_take_partial_assistant_text_returns_and_clears(
    storage: SessionStorage,
) -> None:
    rt = SessionRuntime(storage=storage, session_id="s-take")
    rt.buffer_partial_assistant_text("flush-me")
    text = rt.take_partial_assistant_text()
    assert text == "flush-me"
    assert rt.partial_assistant_text == ""


def test_drain_task_notifications_returns_oldest_first(
    storage: SessionStorage,
) -> None:
    rt = SessionRuntime(storage=storage, session_id="s-drain")
    a = TaskNotification(
        task_id="a", status="completed", summary=None, description="A",
    )
    b = TaskNotification(
        task_id="b", status="failed", summary="boom", description="B",
    )
    rt.enqueue_task_notification(a)
    rt.enqueue_task_notification(b)
    drained = rt.drain_task_notifications()
    assert drained == [a, b]
    assert rt.pending_notifications == ()


def _carryover_with_one_record(path: Path) -> ReadCarryover:
    return ReadCarryover(
        records={
            path: ReadRecord(
                path=path,
                mtime_at_read=0.0,
                size_at_read=0,
                read_at_turn=1,
            ),
        },
        source_session_id="parent-1",
        generated_at_turn=1,
    )


def test_carryover_held_for_first_context_build(
    storage: SessionStorage, tmp_path: Path,
) -> None:
    """Subagent path: parent's :class:`ReadCarryover` flows in via
    constructor so the FIRST Context build can pick it up."""
    carry = _carryover_with_one_record(tmp_path / "f.py")
    rt = SessionRuntime(
        storage=storage, session_id="s-inh", carryover=carry,
    )
    assert rt.carryover is carry


def test_clear_drops_carryover(
    storage: SessionStorage, tmp_path: Path,
) -> None:
    """:meth:`clear` drops the carryover — fresh session must NOT
    resurrect a long-gone parent's fingerprints."""
    carry = _carryover_with_one_record(tmp_path / "f.py")
    rt = SessionRuntime(
        storage=storage, session_id="s-cinh", carryover=carry,
    )
    rt.clear()
    assert rt.carryover is None


# ---------------------------------------------------------------------------
# AgentSession residual-branch coverage (target: aura/application/session.py).
#
# SessionRuntime above is the extracted lifecycle leaf; the AgentSession
# controller still owns model switching, memory/cwd reload, compaction hooks,
# teardown, team wiring, and token estimation. These append-only cases drive
# the residual uncovered branches in session.py using the same FakeChatModel
# harness as tests/test_agent.py — no real model, network, or subprocess.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_web_fetch_factory() -> Iterator[None]:
    # AgentSession.__init__ writes the module-global default model factory;
    # reset after each case so it can never leak into unrelated suites.
    yield
    set_default_model_factory(None)


def _agent_cfg(
    *, enabled: list[str] | None = None, teams: bool = False,
) -> AuraConfig:
    # enabled=None omits tools entirely → shipped default allowlist (UNPINNED),
    # which is what the team auto-enable path needs. A list pins the allowlist.
    payload: dict[str, Any] = {
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
    }
    if enabled is not None:
        payload["tools"] = {"enabled": enabled}
    if teams:
        payload["teams"] = {"enabled": True}
    return AuraConfig.model_validate(payload)


def _make_agent(
    tmp_path: Path,
    *,
    turns: list[FakeTurn] | None = None,
    config: AuraConfig | None = None,
    model: BaseChatModel | None = None,
) -> AgentSession:
    cfg = config or _agent_cfg()
    used_model = model if model is not None else FakeChatModel(turns=turns or [])
    return AgentSession(
        config=cfg,
        model=used_model,
        storage=SessionStorage(tmp_path / "agent.db"),
    )


async def _drain(agent: AgentSession, prompt: str) -> list[AgentEvent | dict[str, object]]:
    events: list[AgentEvent | dict[str, object]] = []
    async for event in agent.astream(prompt):
        events.append(event)
    return events


def _lifecycle_event(state: str) -> CoordinationEvent:
    return CoordinationEvent(
        event="coordination",
        family="team",
        action="team_lifecycle",
        payload={"state": state},
    )


class _FakeTeamPort:
    """Full TeamPort surface so AgentSession's ``_team`` slot type-checks.

    Records ``cleanup_session_teams`` calls and surfaces one scripted
    coordination event so the protocol-event merge paths can be asserted.
    """

    def __init__(self) -> None:
        self.cleanup_calls: int = 0
        self._pending: list[CoordinationEvent] = [_lifecycle_event("running")]

    @property
    def is_active(self) -> bool:
        return True

    @property
    def team(self) -> TeamRecord | None:
        return None

    @property
    def storage(self) -> SessionStorage:
        raise NotImplementedError("storage not exercised by these tests")

    def post_message(self, msg: TeamMessage) -> None:
        return None

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        body: str,
        kind: TeamMessageKind = "text",
    ) -> list[TeamMessage]:
        return []

    def confirm_shutdown(self, member_name: str, *, body: str = "") -> None:
        return None

    @property
    def pending_protocol_events(self) -> tuple[CoordinationEvent, ...]:
        return tuple(self._pending)

    def drain_protocol_events(self) -> list[CoordinationEvent]:
        drained = list(self._pending)
        self._pending.clear()
        return drained

    async def cleanup_session_teams(self) -> None:
        self.cleanup_calls += 1


def test_definition_snapshots_live_model_mode_and_tools(tmp_path: Path) -> None:
    """``definition`` is the per-turn static snapshot the loop binds against;
    it must mirror the live model spec, mode, and the enabled-tool allowlist —
    a drift here silently mis-routes a turn to the wrong model/mode."""
    agent = _make_agent(
        tmp_path, config=_agent_cfg(enabled=["read_file", "bash"]),
    )
    try:
        defn = agent.definition
        assert defn.model_spec == "openai:gpt-4o-mini"
        assert defn.permission_mode == "default"
        assert defn.tool_names == frozenset({"read_file", "bash"})
    finally:
        agent.close()


def test_session_id_setter_retargets_runtime(tmp_path: Path) -> None:
    """The private ``_session_id`` setter is the legitimate writer resume uses;
    writing it must flip the SessionRuntime's live id so subsequent saves land
    under the new key, not the old one."""
    agent = _make_agent(tmp_path)
    try:
        agent._session_id = "relabelled"
        assert agent.session_id == "relabelled"
        assert agent._session_runtime.session_id == "relabelled"
    finally:
        agent.close()


def test_pending_notifications_live_list_reaches_runtime_queue(
    tmp_path: Path,
) -> None:
    """The ``_pending_notifications`` forward exposes the runtime's LIVE queue
    so terminal-task listeners can ``.append`` straight onto it; a copy would
    silently drop task-completion notifications."""
    agent = _make_agent(tmp_path)
    try:
        live = agent._pending_notifications
        live.append(TaskNotification(
            task_id="n1", status="completed", summary=None, description="d",
        ))
        assert agent.pending_notifications[0].task_id == "n1"
    finally:
        agent.close()


def test_teammate_binding_exposed_after_join_with_task(tmp_path: Path) -> None:
    """When a session runs AS a team member with a task id, ``teammate`` must
    expose the binding so progress reports route to the right task record;
    None otherwise."""
    agent = _make_agent(tmp_path, config=_agent_cfg(enabled=[], teams=True))
    try:
        assert agent.teammate is None
        store = agent.tasks_store
        rec: TaskRecord = store.create(description="member-task", prompt="x")
        agent.join_team(
            manager=_FakeTeamPort(),
            member_name="worker",
            task_id=rec.id,
            tasks_store=store,
        )
        binding = agent.teammate
        assert binding is not None
        assert binding.task_id == rec.id
    finally:
        agent.close()


def test_pending_protocol_events_merges_bound_team(tmp_path: Path) -> None:
    """A bound team's coordination events must surface through the session's
    ``pending_protocol_events`` so a transport drains member + leader signals
    in one place, never losing the team's lifecycle ticks."""
    agent = _make_agent(tmp_path, config=_agent_cfg(enabled=[], teams=True))
    try:
        agent.join_team(manager=_FakeTeamPort())
        events = agent.pending_protocol_events
        assert any(
            e.get("action") == "team_lifecycle" for e in events
        )
    finally:
        agent.close()


def test_drain_protocol_events_includes_team_events(tmp_path: Path) -> None:
    """``drain_protocol_events`` must also pop the bound team's queue, then
    leave it empty — otherwise a transport re-emits the same team event every
    poll."""
    agent = _make_agent(tmp_path, config=_agent_cfg(enabled=[], teams=True))
    try:
        team = _FakeTeamPort()
        agent.join_team(manager=team)
        drained: list[WireEvent] = agent.drain_protocol_events()
        assert any(e.get("action") == "team_lifecycle" for e in drained)
        # Second drain is empty — the team's queue was consumed, not copied.
        assert agent.drain_protocol_events() == []
    finally:
        agent.close()


async def test_cascade_abort_skips_already_aborted_child(tmp_path: Path) -> None:
    """A Ctrl+C cascade must be idempotent across children: a controller that
    a sibling already fired keeps its original reason and is not re-aborted,
    while a still-live child gets the parent's reason."""
    agent = _make_agent(tmp_path)
    try:
        already = AbortController()
        already.abort("first_reason")
        fresh = AbortController()
        agent._running_aborts["a"] = already
        agent._running_aborts["b"] = fresh

        await agent._cascade_abort_to_children("parent_aborted")

        # Pre-aborted child keeps its original reason (idempotent abort).
        assert already.reason == "first_reason"
        # Fresh child inherits the parent's cascade reason.
        assert fresh.aborted is True
        assert fresh.reason == "parent_aborted"
    finally:
        agent.close()


def test_set_cwd_same_path_is_noop_no_hook_fire(tmp_path: Path) -> None:
    """``set_cwd`` to the resolved-equal current dir must short-circuit before
    firing CwdChanged consumers — re-running rules/memory reload on every
    no-op move would be wasteful churn."""

    async def _run() -> None:
        agent = _make_agent(tmp_path, config=_agent_cfg(enabled=[]))
        try:
            fired: list[Path] = []

            async def _spy(
                *, old_cwd: Path, new_cwd: Path, state: Any, **_: Any,
            ) -> None:
                fired.append(new_cwd)

            agent._hooks.cwd_changed.append(_spy)
            await agent.set_cwd(agent.cwd)
            assert fired == []  # same path → no hook fire
        finally:
            agent.close()

    asyncio.run(_run())


def test_estimate_history_tokens_includes_pinned_prefix(tmp_path: Path) -> None:
    """The token fallback must count the pinned prefix on TOP of history;
    history-only would under-report by the whole system prompt, mis-sizing the
    status bar right before an overflow."""
    agent = _make_agent(tmp_path, config=_agent_cfg(enabled=[]))
    try:
        empty = agent.estimate_history_tokens([])
        with_msg = agent.estimate_history_tokens(
            [HumanMessage(content="some user words here")],
        )
        # Empty history still costs the pinned prefix; a message strictly adds.
        assert empty == agent.pinned_tokens_estimate
        assert with_msg > empty
    finally:
        agent.close()


def test_estimate_pinned_tokens_survives_unserializable_tool_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool whose ``args`` schema can't be JSON-dumped must NOT crash the
    pinned-token estimate — the except branch falls back to an empty schema so
    the status bar stays alive even with an exotic tool."""

    class _Bad:
        # json.dumps(default=str) still raises if __str__ itself blows up.
        def __repr__(self) -> str:
            raise ValueError("unserializable repr")

    class _P(BaseModel):
        pass

    tool: BaseTool = build_tool(
        name="exotic",
        description="exotic tool",
        args_schema=_P,
        func=lambda: {},
        is_read_only=True,
    )
    cfg = _agent_cfg(enabled=["exotic"])
    agent = AgentSession(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "agent.db"),
        available_tools={"exotic": tool},
    )
    try:
        # Force the tool to advertise an un-dumpable args mapping; the estimate
        # must swallow the TypeError/ValueError and keep counting.
        monkeypatch.setattr(
            type(tool), "args",
            property(lambda _self: {"x": _Bad()}),
            raising=False,
        )
        est = agent._estimate_pinned_tokens()
        assert est > 0  # name + description still counted; schema skipped
    finally:
        agent.close()


def test_snapshot_read_carryover_rebases_records_to_current_turn(
    tmp_path: Path,
) -> None:
    """A subagent spawn snapshots the parent's read fingerprints, re-stamped at
    the parent's CURRENT turn — so a child's must-read gate trusts files the
    parent already read, without copying a stale read_at_turn."""
    agent = _make_agent(tmp_path, config=_agent_cfg(enabled=["read_file"]))
    try:
        target = tmp_path / "seen.txt"
        target.write_text("body\n", encoding="utf-8")
        agent._context.record_read(target)
        agent.state.turn_count = 7

        carry = agent._snapshot_read_carryover()

        assert carry.source_session_id == agent.session_id
        assert carry.generated_at_turn == 7
        assert target.resolve() in carry.records
        assert carry.records[target.resolve()].read_at_turn == 7
    finally:
        agent.close()


async def test_teardown_kills_lingering_shell_with_no_returncode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shell subprocess still running at teardown (returncode is None) must
    be SIGKILL'd belt-and-braces; a finished one (returncode set) is left
    alone and merely dropped from the map."""
    agent = _make_agent(tmp_path, config=_agent_cfg(enabled=[]))
    try:
        killed: list[str] = []

        class _FakeProc:
            def __init__(self, returncode: int | None, label: str) -> None:
                self.returncode = returncode
                self._label = label

            def kill(self) -> None:
                killed.append(self._label)

        # The real map is typed for asyncio subprocess handles; swap in a
        # duck-typed map exposing only the returncode + kill members teardown
        # touches, so no real process is ever spawned.
        fake_shells: dict[str, Any] = {
            "live": _FakeProc(None, "live"),
            "done": _FakeProc(0, "done"),
        }
        monkeypatch.setattr(agent, "_running_shells", fake_shells)

        agent._teardown_local_tasks()

        assert killed == ["live"]  # only the unfinished proc is killed
        assert fake_shells == {}  # both dropped from the map
    finally:
        with contextlib.suppress(Exception):
            agent._session_runtime.close_storage()


async def test_aclose_fires_team_cleanup_only_for_leader(tmp_path: Path) -> None:
    """Only the team LEADER may run ``cleanup_session_teams`` on close — a
    teammate shares the leader's team set and would cancel siblings mid-flight,
    so a member (``_team_member_name`` set) must skip the cleanup call."""
    leader = AgentSession(
        config=_agent_cfg(enabled=[], teams=True),
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "leader.db"),
    )
    leader_team = _FakeTeamPort()
    leader.join_team(manager=leader_team)  # no member_name → leader
    await leader.aclose()
    assert leader_team.cleanup_calls == 1

    member = AgentSession(
        config=_agent_cfg(enabled=[], teams=True),
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "member.db"),
    )
    member_team = _FakeTeamPort()
    member.join_team(manager=member_team, member_name="worker")
    await member.aclose()
    assert member_team.cleanup_calls == 0


async def test_reactive_compact_callback_refreshes_history_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On context overflow the loop hands its working ``history`` list to the
    reactive callback, which must mutate it IN PLACE to the post-compact
    transcript — re-binding the local would leave the loop on stale history."""
    agent = _make_agent(tmp_path, config=_agent_cfg(enabled=[]))
    try:
        compact_sources: list[str] = []

        async def _fake_compact(*, source: str = "manual") -> None:
            compact_sources.append(source)
            agent.storage.save(
                agent.session_id, [AIMessage(content="compacted-summary")],
            )

        # Replace the real summariser; we only assert the in-place refresh.
        monkeypatch.setattr(agent, "compact", _fake_compact)
        working: list[BaseMessage] = [HumanMessage(content="stale")]
        await agent._reactive_compact_callback(working)

        assert compact_sources == ["reactive"]
        assert [m.content for m in working] == ["compacted-summary"]
    finally:
        agent.close()


async def test_astream_drains_compact_events_after_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-compact runs AFTER the stream closes; any compact wire events it
    enqueues must still be yielded to the caller in the same astream call, not
    stranded until the next turn."""
    agent = _make_agent(
        tmp_path,
        config=_agent_cfg(enabled=[]),
        turns=[FakeTurn(AIMessage(content="hi"))],
    )
    try:
        sentinel: dict[str, object] = {"type": "compact", "marker": "post-stream"}
        queue = agent._pending_compact_events

        async def _fake_auto(
            history: list[BaseMessage], slots: Any, *, model: str,
            trigger: Any = None,
        ) -> None:
            # Mimic the real auto-compact enqueuing a wire event at the seam.
            queue.append(sentinel)

        monkeypatch.setattr(agent._compactor, "auto", _fake_auto)
        events = await _drain(agent, "go")
        assert sentinel in events
        assert any(isinstance(e, Final) for e in events)
    finally:
        await agent.aclose()


async def test_astream_self_abort_yields_cancelled_final_and_drops_turn(
    tmp_path: Path,
) -> None:
    """A self-abort with no AIMessage landed must yield a ``(cancelled)`` Final,
    drop the unanswered user turn from storage, and NOT re-raise (own abort,
    no parent) — the REPL keeps running on the next prompt."""

    class _AbortingModel(FakeChatModel):
        def __init__(self, ctrl: AbortController) -> None:
            super().__init__(turns=[])
            self.__dict__["_ctrl"] = ctrl

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **_: Any,
        ) -> Any:
            ctrl: AbortController = self.__dict__["_ctrl"]
            ctrl.abort("user_ctrl_c")
            raise AbortException("user_ctrl_c")

    ctrl = AbortController()
    agent = _make_agent(
        tmp_path,
        config=_agent_cfg(enabled=[]),
        model=_AbortingModel(ctrl),
    )
    try:
        events: list[AgentEvent | dict[str, object]] = []
        async for event in agent.astream("orphan prompt", abort=ctrl):
            events.append(event)

        finals = [e for e in events if isinstance(e, Final)]
        assert finals and finals[-1].message == "(cancelled)"
        assert finals[-1].reason == "aborted"
        # Unanswered user turn was rolled back — storage stays empty.
        assert agent.storage.load(agent.session_id) == []
    finally:
        await agent.aclose()


async def test_astream_cancellederror_with_aborted_controller_reraises(
    tmp_path: Path,
) -> None:
    """A CancelledError raised while the local controller is already aborted is
    a real abort, not a stray cancel: astream must yield ``(cancelled)`` AND
    re-raise the CancelledError so a subagent's run_task records a terminal
    status instead of swallowing it as a clean return."""
    ctrl = AbortController()

    async def _abort_then_cancel(
        *, history: list[BaseMessage], state: Any, **_: Any,
    ) -> None:
        # Mark the controller aborted, then raise a raw CancelledError so the
        # is_abort + CancelledError re-raise branch (not the parent branch) fires.
        ctrl.abort("ctrl_c")
        raise asyncio.CancelledError

    hooks = HookChain(pre_model=[_abort_then_cancel])
    agent = AgentSession(
        config=_agent_cfg(enabled=[]),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="never"))]),
        storage=SessionStorage(tmp_path / "agent.db"),
        hooks=hooks,
    )
    try:
        events: list[AgentEvent | dict[str, object]] = []
        with pytest.raises(asyncio.CancelledError):
            async for event in agent.astream("orphan", abort=ctrl):
                events.append(event)
        # The terminal Final was still surfaced before the re-raise.
        assert any(
            isinstance(e, Final) and e.message == "(cancelled)" for e in events
        )
    finally:
        await agent.aclose()


def test_switch_model_rebuilds_loop_with_new_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``switch_model`` swaps the live model and rebuilds the loop so the next
    turn binds the NEW model; ``current_model`` must reflect the new spec while
    ``config.router`` stays immutable."""
    from aura.infrastructure import llm

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini", "big": "openai:gpt-4o"},
        "tools": {"enabled": []},
    })
    model_b = FakeChatModel(turns=[])
    agent = AgentSession(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "agent.db"),
    )
    try:
        monkeypatch.setattr(llm, "create", lambda provider, name: model_b)
        old_loop = agent._loop
        agent.switch_model("big")
        # The live spec stores the alias verbatim; resolve() expands it under
        # the hood to pick the model, but current_model echoes what was asked.
        assert agent.current_model == "big"
        assert agent.model is model_b
        assert agent._loop is not old_loop  # loop rebuilt around the new model
        assert cfg.router["default"] == "openai:gpt-4o-mini"  # config untouched
    finally:
        agent.close()


def test_change_cwd_and_reload_refreshes_memory_from_new_dir(
    tmp_path: Path,
) -> None:
    """``change_cwd_and_reload`` retargets ``cwd`` and re-reads AURA.md from the
    NEW directory — a memory file that only exists under the new cwd must show
    up after the move."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "AURA.md").write_text("MOVED-MEMORY", encoding="utf-8")

    agent = _make_agent(tmp_path, config=_agent_cfg(enabled=[]))
    try:
        assert "MOVED-MEMORY" not in agent._primary_memory
        agent.change_cwd_and_reload(other)
        assert agent.cwd == other
        assert "MOVED-MEMORY" in agent._primary_memory
    finally:
        agent.close()


def test_auto_enable_short_circuits_when_already_registered(
    tmp_path: Path,
) -> None:
    """Joining a SECOND distinct team must not re-register send_message — the
    registry rejects duplicate names, so the already-wired guard has to fire on
    the team switch, not just the same-manager fast path."""
    # Unpinned allowlist (no tools.enabled) so auto-enable actually wires it.
    agent = _make_agent(tmp_path, config=_agent_cfg(teams=True))
    try:
        first = _FakeTeamPort()
        second = _FakeTeamPort()
        agent.join_team(manager=first)
        assert "send_message" in agent._registry
        # Different manager → join_team's same-manager fast path is skipped,
        # so _auto_enable runs again and must short-circuit on "already wired".
        agent.join_team(manager=second)
        assert "send_message" in agent._registry
        assert agent.team is second
    finally:
        agent.close()


def test_auto_enable_is_inert_when_teams_disabled(tmp_path: Path) -> None:
    """The auto-enable helper guards on teams.enabled FIRST — even if invoked
    on a teams-disabled session it must be a silent no-op, never wiring
    send_message into a swarm-less session."""
    agent = _make_agent(tmp_path, config=_agent_cfg())  # teams disabled
    try:
        assert agent.config.teams.enabled is False
        agent._auto_enable_send_message_for_team()  # teams-off early return
        assert "send_message" not in agent._registry
    finally:
        agent.close()


def test_leave_team_without_send_message_is_clean_noop(tmp_path: Path) -> None:
    """leave_team on an unpinned session that never auto-added send_message
    must take the ``not in registry`` early return — unregistering a tool that
    was never added would raise and break a clean /team teardown."""
    # Unpinned allowlist + teams enabled, but we never join, so send_message is
    # never auto-added; leave_team must short-circuit on the missing tool.
    agent = _make_agent(tmp_path, config=_agent_cfg(teams=True))
    try:
        assert "send_message" not in agent._registry
        agent.leave_team()  # not-in-registry early return; must not raise
        assert "send_message" not in agent._registry
    finally:
        agent.close()


def test_leave_team_respects_user_pinned_allowlist(tmp_path: Path) -> None:
    """A user-pinned allowlist suppresses auto-enable, so leave_team must take
    the ``user pinned`` early return — a pinned (non-send_message) allowlist
    means the tool was never auto-added and must not be stripped."""
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["read_file"]},  # pinned, excludes send_message
        "teams": {"enabled": True},
    })
    agent = _make_agent(tmp_path, config=cfg)
    try:
        agent.join_team(manager=_FakeTeamPort())
        assert "send_message" not in agent._registry  # pin suppressed auto-add
        agent.leave_team()  # user-pinned early return; must not raise
        assert "send_message" not in agent._registry
    finally:
        agent.close()
