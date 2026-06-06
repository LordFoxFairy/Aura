"""LocalAgentTask — the only runner topology task_create dispatches to.

Exercises spawn / abort / idempotent start via the
``start() / wait_for_terminal() / abort()`` contract.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aura.application.hooks import HookChain
from aura.application.session import AgentSession
from aura.application.tasks.runners import LocalAgentTask, local_agent_io
from aura.application.tasks.runners.local_agent import run_local_agent
from aura.application.tasks.spawn import SpawnContext, SubagentSpawner
from aura.application.tasks.spawn_port import SpawnedAgent
from aura.application.tasks.store import TasksStore
from aura.config.schema import AuraConfig
from aura.domain.events import AgentEvent, Final, ToolCallStarted
from aura.domain.task import TaskRecord
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def _factory_with_reply(text: str) -> SubagentSpawner[AgentSession]:
    return SubagentSpawner(
        SpawnContext(
            parent_config=_cfg(),
            parent_model_spec="openai:gpt-4o-mini",
            build_child=AgentSession,
            model_factory=lambda: FakeChatModel(
                turns=[FakeTurn(AIMessage(content=text))],
            ),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )


@pytest.mark.asyncio
async def test_local_agent_task_runs_to_completion() -> None:
    store = TasksStore()
    factory = _factory_with_reply("child done")
    rec = store.create(description="probe", prompt="go")
    task = LocalAgentTask(store=store, factory=factory, task_id=rec.id)
    task.start()
    await task.wait_for_terminal()
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.final_result == "child done"


@pytest.mark.asyncio
async def test_local_agent_task_abort_cancels_task() -> None:
    class _Slow(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            # Long enough that the abort definitely lands before
            # natural completion. CancelledError will short-circuit it.
            await asyncio.sleep(5.0)
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="!"))],
            )

    factory = SubagentSpawner(
        SpawnContext(
            parent_config=_cfg(),
            parent_model_spec="openai:gpt-4o-mini",
            build_child=AgentSession,
            model_factory=lambda: _Slow(),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    store = TasksStore()
    rec = store.create(description="slow", prompt="hang")
    task = LocalAgentTask(store=store, factory=factory, task_id=rec.id)
    task.start()
    # Give the child a moment to enter ``_agenerate`` so the cancel
    # has something to race against.
    await asyncio.sleep(0.05)
    task.abort()
    await task.wait_for_terminal()
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "cancelled"


@pytest.mark.asyncio
async def test_local_agent_task_start_is_idempotent() -> None:
    factory = _factory_with_reply("ok")
    store = TasksStore()
    rec = store.create(description="probe", prompt="go")
    task = LocalAgentTask(store=store, factory=factory, task_id=rec.id)
    t1 = task.start()
    t2 = task.start()
    # Same handle returned — no double-scheduling.
    assert t1 is t2
    await task.wait_for_terminal()


# --- local_agent_io: transcript / metadata persistence + child-message capture ---


def _disk_storage(tmp_path: Path) -> SessionStorage:
    """On-disk store so subagent_transcript_path resolves under tmp, not :memory:."""
    return SessionStorage(tmp_path / "store.db", cwd=tmp_path)


def _spawn_real_agent(task_id: str) -> SpawnedAgent:
    """A genuine SpawnedAgent (FakeChatModel, :memory: storage) — no network."""
    return _factory_with_reply("child").spawn("go", task_id=task_id)


class _CleanupFlagAgent:
    """Delegates to a real child but forces the opt-in cleanup flag on its config."""

    def __init__(self, inner: SpawnedAgent) -> None:
        self._inner = inner
        self._config = inner.config.model_copy(update={
            "tools": inner.config.tools.model_copy(
                update={"cleanup_completed_subagent_transcripts": True},
            ),
        })

    @property
    def config(self) -> AuraConfig:
        return self._config

    @property
    def model(self) -> BaseChatModel:
        return self._inner.model

    @property
    def storage(self) -> SessionStorage:
        return self._inner.storage

    @property
    def session_id(self) -> str:
        return self._inner.session_id

    @property
    def hooks(self) -> HookChain:
        return self._inner.hooks

    def astream(self, prompt: str) -> AsyncIterator[AgentEvent | dict[str, Any]]:
        return self._inner.astream(prompt)

    async def aclose(self) -> None:
        await self._inner.aclose()


def _spawn_cleanup_agent(task_id: str) -> SpawnedAgent:
    """A SpawnedAgent whose config opts into completed-transcript cleanup."""
    return _CleanupFlagAgent(_spawn_real_agent(task_id))


class _JournalSink:
    """Captures journal.write events so error paths can be asserted without disk."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, event: str, /, **fields: Any) -> None:
        self.events.append((event, dict(fields)))


@pytest.mark.asyncio
async def test_flush_transcript_writes_file_and_records_path(tmp_path: Path) -> None:
    """A child's transcript must persist to disk and back-reference into the store."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    storage = _disk_storage(tmp_path)
    msgs: list[BaseMessage] = [HumanMessage(content="hi"), AIMessage(content="bye")]
    path = local_agent_io.flush_transcript(
        transcript_storage=storage,
        task_id=rec.id,
        messages=msgs,
        store=store,
        parent_session_id="parent-123",
        cwd=str(tmp_path),
    )
    assert path is not None
    assert path.exists()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.transcript_path == path


@pytest.mark.asyncio
async def test_flush_transcript_empty_cwd_and_parent_use_flat_bucket(
    tmp_path: Path,
) -> None:
    """Empty parent/cwd are the ad-hoc-bucket sentinels, not literal '' path parts."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    storage = _disk_storage(tmp_path)
    path = local_agent_io.flush_transcript(
        transcript_storage=storage,
        task_id=rec.id,
        messages=[AIMessage(content="x")],
        store=store,
        parent_session_id="",
        cwd="",
    )
    assert path is not None
    # Flat bucket lives directly under the store root — no encoded-cwd project dir.
    assert path.parent.name == "subagents"


@pytest.mark.asyncio
async def test_flush_transcript_swallows_persistence_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistence is best-effort: a write failure logs and returns None, never raises."""
    store = TasksStore()
    storage = _disk_storage(tmp_path)
    sink = _JournalSink()
    monkeypatch.setattr(journal, "write", sink)
    # An empty task_id trips validate_task_id deep inside the write seam.
    path = local_agent_io.flush_transcript(
        transcript_storage=storage,
        task_id="",
        messages=[AIMessage(content="x")],
        store=store,
    )
    assert path is None
    assert sink.events[0][0] == "subagent_transcript_flush_error"


@pytest.mark.asyncio
async def test_flush_metadata_writes_companion_with_ended_iso(tmp_path: Path) -> None:
    """A finished task records both started_at and an ended_at ISO timestamp."""
    storage = _disk_storage(tmp_path)
    rec = TaskRecord(id="t1", description="desc", prompt="p", status="completed")
    rec.started_at = 1000.0
    rec.finished_at = 2000.0
    rec.agent_type = "explorer"
    rec.model_spec = "openai:gpt-4o-mini"
    rec.progress.input_tokens = 7
    rec.progress.output_tokens = 11
    meta_path = local_agent_io.flush_metadata(
        transcript_storage=storage,
        record=rec,
        parent_session_id="parent",
        cwd=str(tmp_path),
    )
    assert meta_path is not None

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["agent_type"] == "explorer"
    assert payload["ended_at"] == 2000.0
    assert payload["ended_at_iso"] is not None
    assert payload["input_tokens"] == 7
    assert payload["output_tokens"] == 11
    # Atomic write must leave no .tmp sibling behind.
    assert not meta_path.with_suffix(meta_path.suffix + ".tmp").exists()


@pytest.mark.asyncio
async def test_flush_metadata_running_task_has_null_ended_fields(
    tmp_path: Path,
) -> None:
    """An unfinished task must serialise ended_at/ended_at_iso as null, not 0/epoch."""
    storage = _disk_storage(tmp_path)
    rec = TaskRecord(id="t2", description="d", prompt="p")
    rec.started_at = 5.0
    rec.finished_at = None
    rec.agent_type = None
    rec.model_spec = ""
    meta_path = local_agent_io.flush_metadata(
        transcript_storage=storage,
        record=rec,
        parent_session_id="",
        cwd="",
    )
    assert meta_path is not None

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["ended_at"] is None
    assert payload["ended_at_iso"] is None
    # Empty agent_type/model_spec/parent/cwd fall back to documented defaults.
    assert payload["agent_type"] == "general-purpose"
    assert payload["model_spec"] == ""
    assert payload["parent_session_id"] == ""
    assert payload["cwd"] == ""


@pytest.mark.asyncio
async def test_flush_metadata_is_idempotent(tmp_path: Path) -> None:
    """Re-flushing the same record overwrites atomically with identical content."""
    storage = _disk_storage(tmp_path)
    rec = TaskRecord(id="t3", description="d", prompt="p", status="completed")
    rec.started_at = 1.0
    rec.finished_at = 2.0
    first = local_agent_io.flush_metadata(
        transcript_storage=storage, record=rec, parent_session_id="par", cwd="",
    )
    second = local_agent_io.flush_metadata(
        transcript_storage=storage, record=rec, parent_session_id="par", cwd="",
    )
    assert first is not None
    assert second == first
    assert first.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_flush_metadata_swallows_path_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A metadata path/write blow-up logs and returns None instead of crashing."""
    storage = _disk_storage(tmp_path)
    rec = TaskRecord(id="t4", description="d", prompt="p")
    sink = _JournalSink()
    monkeypatch.setattr(journal, "write", sink)

    def _boom(*_a: object, **_k: object) -> Path:
        raise OSError("disk gone")

    monkeypatch.setattr(storage, "subagent_metadata_path", _boom)
    result = local_agent_io.flush_metadata(
        transcript_storage=storage, record=rec, parent_session_id="p", cwd="",
    )
    assert result is None
    assert sink.events[0][0] == "subagent_metadata_flush_error"


@pytest.mark.asyncio
async def test_maybe_cleanup_noop_when_flag_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default config keeps transcripts; cleanup must not touch disk or journal."""
    agent = _spawn_real_agent("ct1")
    assert not agent.config.tools.cleanup_completed_subagent_transcripts
    storage = _disk_storage(tmp_path)
    sink = _JournalSink()
    monkeypatch.setattr(journal, "write", sink)
    local_agent_io.maybe_cleanup_completed_transcript(
        agent=agent,
        transcript_storage=storage,
        task_id="ct1",
        parent_session_id="par",
        cwd=str(tmp_path),
    )
    assert sink.events == []


@pytest.mark.asyncio
async def test_maybe_cleanup_unlinks_transcript_and_meta(tmp_path: Path) -> None:
    """Opt-in cleanup deletes both the transcript and its meta companion."""
    agent = _spawn_cleanup_agent("ct2")
    assert agent.config.tools.cleanup_completed_subagent_transcripts
    storage = _disk_storage(tmp_path)
    tpath = storage.subagent_transcript_path("ct2", parent_session_id="par")
    mpath = storage.subagent_metadata_path("ct2", parent_session_id="par")
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tpath.write_text("{}\n", encoding="utf-8")
    mpath.write_text("{}\n", encoding="utf-8")
    local_agent_io.maybe_cleanup_completed_transcript(
        agent=agent,
        transcript_storage=storage,
        task_id="ct2",
        parent_session_id="par",
        cwd="",
    )
    assert not tpath.exists()
    assert not mpath.exists()


@pytest.mark.asyncio
async def test_maybe_cleanup_missing_files_is_idempotent(tmp_path: Path) -> None:
    """Cleaning an already-absent transcript twice must stay silent (missing_ok)."""
    agent = _spawn_cleanup_agent("ct3")
    storage = _disk_storage(tmp_path)
    for _ in range(2):
        local_agent_io.maybe_cleanup_completed_transcript(
            agent=agent,
            transcript_storage=storage,
            task_id="ct3",
            parent_session_id="par",
            cwd="",
        )


@pytest.mark.asyncio
async def test_maybe_cleanup_logs_per_file_unlink_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-file unlink OSError is logged but never aborts the second deletion."""
    agent = _spawn_cleanup_agent("ct4")
    storage = _disk_storage(tmp_path)
    tpath = storage.subagent_transcript_path("ct4", parent_session_id="par")
    mpath = storage.subagent_metadata_path("ct4", parent_session_id="par")
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tpath.write_text("{}\n", encoding="utf-8")
    mpath.write_text("{}\n", encoding="utf-8")
    sink = _JournalSink()
    monkeypatch.setattr(journal, "write", sink)
    real_unlink = Path.unlink

    def _flaky_unlink(self: Path, *, missing_ok: bool = False) -> None:
        if self.suffix == ".jsonl":
            raise OSError("locked")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _flaky_unlink)
    local_agent_io.maybe_cleanup_completed_transcript(
        agent=agent,
        transcript_storage=storage,
        task_id="ct4",
        parent_session_id="par",
        cwd="",
    )
    names = [e[0] for e in sink.events]
    assert "subagent_transcript_cleanup_error" in names
    assert "subagent_transcript_cleaned" in names


@pytest.mark.asyncio
async def test_maybe_cleanup_outer_exception_is_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path-resolution blow-up is logged once and never propagates to the caller."""
    agent = _spawn_cleanup_agent("ct5")
    storage = _disk_storage(tmp_path)
    sink = _JournalSink()
    monkeypatch.setattr(journal, "write", sink)

    def _boom(*_a: object, **_k: object) -> Path:
        raise RuntimeError("resolver dead")

    monkeypatch.setattr(storage, "subagent_transcript_path", _boom)
    local_agent_io.maybe_cleanup_completed_transcript(
        agent=agent,
        transcript_storage=storage,
        task_id="ct5",
        parent_session_id="par",
        cwd="",
    )
    assert sink.events[-1][0] == "subagent_transcript_cleanup_error"


@pytest.mark.asyncio
async def test_capture_child_messages_none_agent_is_noop() -> None:
    """No spawned child means nothing to capture — must return without error."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    await local_agent_io.capture_child_messages(None, store, rec.id)
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.messages == []


@pytest.mark.asyncio
async def test_capture_child_messages_missing_record_is_noop() -> None:
    """A vanished TaskRecord must short-circuit, not raise on attribute access."""
    store = TasksStore()
    agent = _spawn_real_agent("cap1")
    await local_agent_io.capture_child_messages(agent, store, "no-such-task")


@pytest.mark.asyncio
async def test_capture_child_messages_copies_loaded_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured messages land on the record as an independent list copy."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    agent = _spawn_real_agent(rec.id)
    loaded: list[BaseMessage] = [HumanMessage(content="q"), AIMessage(content="a")]
    monkeypatch.setattr(agent.storage, "load", lambda _sid: loaded)
    await local_agent_io.capture_child_messages(agent, store, rec.id)
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert [m.content for m in refreshed.messages] == ["q", "a"]
    assert refreshed.messages is not loaded


@pytest.mark.asyncio
async def test_capture_child_messages_swallows_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A storage read failure during capture logs and never propagates."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    agent = _spawn_real_agent(rec.id)
    sink = _JournalSink()
    monkeypatch.setattr(journal, "write", sink)

    def _boom(_sid: str) -> list[BaseMessage]:
        raise RuntimeError("db read failed")

    monkeypatch.setattr(agent.storage, "load", _boom)
    await local_agent_io.capture_child_messages(agent, store, rec.id)
    assert sink.events[0][0] == "subagent_capture_messages_error"


@pytest.mark.asyncio
async def test_load_child_messages_prefers_record_messages() -> None:
    """Already-captured record messages win over re-reading the child storage."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    rec.messages = [AIMessage(content="cached")]
    out = local_agent_io.load_child_messages(None, store, rec.id)
    assert [m.content for m in out] == ["cached"]
    assert out is not rec.messages


@pytest.mark.asyncio
async def test_load_child_messages_falls_back_to_agent_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With an empty record, the child's own transcript is the source of truth."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    agent = _spawn_real_agent(rec.id)
    loaded: list[BaseMessage] = [AIMessage(content="fromdisk")]
    monkeypatch.setattr(agent.storage, "load", lambda _sid: loaded)
    out = local_agent_io.load_child_messages(agent, store, rec.id)
    assert [m.content for m in out] == ["fromdisk"]


@pytest.mark.asyncio
async def test_load_child_messages_empty_when_no_record_no_agent() -> None:
    """No record and no agent yields an empty list — never None, never a raise."""
    store = TasksStore()
    out = local_agent_io.load_child_messages(None, store, "missing")
    assert out == []


@pytest.mark.asyncio
async def test_load_child_messages_swallows_storage_error_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing fallback read is suppressed, yielding an empty list not a crash."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    agent = _spawn_real_agent(rec.id)

    def _boom(_sid: str) -> list[BaseMessage]:
        raise RuntimeError("io error")

    monkeypatch.setattr(agent.storage, "load", _boom)
    out = local_agent_io.load_child_messages(agent, store, rec.id)
    assert out == []


# --- run_local_agent: timeout / cancel / driver-error / abort-cascade branches ---


class _FakeSpawnedAgent:
    """Scripted child whose event stream + close behaviour drive the runner branches."""

    def __init__(
        self,
        *,
        events: list[AgentEvent] | None = None,
        stream_error: BaseException | None = None,
        yield_ticks: int = 0,
    ) -> None:
        # A real :memory: SessionStorage so capture_child_messages' load() works.
        self._storage = SessionStorage(Path(":memory:"))
        self._hooks = HookChain()
        self._events = events or []
        # stream_error raises from inside astream after the scripted events drain.
        # Feeding TimeoutError / CancelledError here drives the runner's matching
        # except-branches deterministically — identical to a real asyncio.timeout
        # deadline or parent-abort cancel, but with zero wallclock-race flakiness.
        self._stream_error = stream_error
        # yield_ticks > 0 hands control back via sleep(0) ticks (no wallclock) so a
        # pre-armed abort watcher gets scheduled and lands its cancel mid-stream.
        self._yield_ticks = yield_ticks
        self.closed = False

    @property
    def config(self) -> AuraConfig:
        return _cfg()

    @property
    def model(self) -> BaseChatModel:
        return FakeChatModel()

    @property
    def storage(self) -> SessionStorage:
        return self._storage

    @property
    def session_id(self) -> str:
        return "fake-child"

    @property
    def hooks(self) -> HookChain:
        return self._hooks

    async def astream(
        self, prompt: str,  # noqa: ARG002  # matches SpawnedAgent.astream signature
    ) -> AsyncIterator[AgentEvent | dict[str, Any]]:
        for event in self._events:
            yield event
        for _ in range(self._yield_ticks):
            await asyncio.sleep(0)
        if self._stream_error is not None:
            raise self._stream_error

    async def aclose(self) -> None:
        self.closed = True


class _FakeSpawnPort:
    """Minimal SpawnPort feeding one scripted child; spawn may itself raise."""

    def __init__(
        self,
        *,
        agent: _FakeSpawnedAgent | None = None,
        abort_event: asyncio.Event | None = None,
        spawn_error: Exception | None = None,
    ) -> None:
        self._agent = agent
        self._abort_event = abort_event
        self._spawn_error = spawn_error

    @property
    def parent_model_spec(self) -> str:
        return "openai:gpt-4o-mini"

    @property
    def abort_event(self) -> asyncio.Event | None:
        return self._abort_event

    def validate_model_spec(self, spec: str) -> None:  # noqa: ARG002  # Protocol noop
        return

    def spawn(
        self,
        prompt: str,  # noqa: ARG002  # matches SpawnPort.spawn signature
        allowed_tools: list[str] | None = None,  # noqa: ARG002  # ditto
        *,
        agent_type: str = "general-purpose",  # noqa: ARG002  # ditto
        task_id: str | None = None,  # noqa: ARG002  # ditto
        model_spec: str | None = None,  # noqa: ARG002  # ditto
    ) -> SpawnedAgent:
        if self._spawn_error is not None:
            raise self._spawn_error
        assert self._agent is not None
        return self._agent


@pytest.mark.asyncio
async def test_wait_for_terminal_noop_when_never_started() -> None:
    """An unstarted task must await cleanly — no AttributeError on the null handle."""
    store = TasksStore()
    factory = _factory_with_reply("x")
    rec = store.create(description="d", prompt="p")
    task = LocalAgentTask(store=store, factory=factory, task_id=rec.id)
    await task.wait_for_terminal()
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "running"


@pytest.mark.asyncio
async def test_run_local_agent_missing_record_short_circuits() -> None:
    """A vanished task_id must return before spawning — no child, no store mutation."""
    store = TasksStore()
    agent = _FakeSpawnedAgent(events=[Final(message="never")])
    factory = _FakeSpawnPort(agent=agent)
    await run_local_agent(
        store=store,
        factory=factory,
        task_id="ghost",
        timeout_sec=None,
        transcript_storage=None,
        parent_session_id=None,
        cwd=None,
    )
    assert not agent.closed
    assert store.get("ghost") is None


@pytest.mark.asyncio
async def test_run_local_agent_records_tool_activity_and_final() -> None:
    """A ToolCallStarted bumps tool_count; a Final supplies the completed result text."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    agent = _FakeSpawnedAgent(events=[
        ToolCallStarted(name="Bash", input={}),
        Final(message="all done"),
    ])
    factory = _FakeSpawnPort(agent=agent)
    await run_local_agent(
        store=store,
        factory=factory,
        task_id=rec.id,
        timeout_sec=None,
        transcript_storage=None,
        parent_session_id=None,
        cwd=None,
    )
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.final_result == "all done"
    assert refreshed.progress.tool_count == 1
    assert agent.closed


@pytest.mark.asyncio
async def test_run_local_agent_timeout_marks_failed_with_ceiling_message() -> None:
    """A deadline TimeoutError fails the task with the override-hint ceiling message."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    # TimeoutError from the stream is indistinguishable to the runner from an
    # asyncio.timeout deadline — both land in the same except-branch.
    agent = _FakeSpawnedAgent(stream_error=TimeoutError())
    factory = _FakeSpawnPort(agent=agent)
    await run_local_agent(
        store=store,
        factory=factory,
        task_id=rec.id,
        timeout_sec=12.5,
        transcript_storage=None,
        parent_session_id=None,
        cwd=None,
    )
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.error is not None
    assert "subagent_timeout" in refreshed.error
    assert "12.5s" in refreshed.error
    assert agent.closed


@pytest.mark.asyncio
async def test_run_local_agent_driver_error_marks_failed_and_closes() -> None:
    """A child stream blow-up is captured as a typed failure, not a propagated crash."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    agent = _FakeSpawnedAgent(
        events=[ToolCallStarted(name="Read", input={})],
        stream_error=RuntimeError("driver exploded"),
    )
    factory = _FakeSpawnPort(agent=agent)
    await run_local_agent(
        store=store,
        factory=factory,
        task_id=rec.id,
        timeout_sec=None,
        transcript_storage=None,
        parent_session_id=None,
        cwd=None,
    )
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.error == "RuntimeError: driver exploded"
    assert agent.closed


@pytest.mark.asyncio
async def test_run_local_agent_spawn_failure_marks_failed_without_agent() -> None:
    """spawn() raising before assignment must still fail-mark the task, never abort cleanup."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    factory = _FakeSpawnPort(spawn_error=ValueError("no model"))
    await run_local_agent(
        store=store,
        factory=factory,
        task_id=rec.id,
        timeout_sec=None,
        transcript_storage=None,
        parent_session_id=None,
        cwd=None,
    )
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.error == "ValueError: no model"


@pytest.mark.asyncio
async def test_run_local_agent_cancelled_marks_cancelled_and_reraises() -> None:
    """A cancel mid-stream marks the task cancelled, flushes, then re-raises to caller."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    # CancelledError from the stream mirrors a parent-abort / Ctrl+C cancellation.
    agent = _FakeSpawnedAgent(stream_error=asyncio.CancelledError())
    factory = _FakeSpawnPort(agent=agent)
    with pytest.raises(asyncio.CancelledError):
        await run_local_agent(
            store=store,
            factory=factory,
            task_id=rec.id,
            timeout_sec=None,
            transcript_storage=None,
            parent_session_id=None,
            cwd=None,
        )
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "cancelled"
    assert agent.closed


@pytest.mark.asyncio
async def test_run_local_agent_arms_abort_watcher_and_tears_it_down() -> None:
    """A non-null parent abort Event arms a watcher that is torn down on clean exit."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    # A never-fired abort Event still forces the watcher-arming branch; the finally
    # block must cancel that watcher so the unfired Event leaves no live waiter.
    abort = asyncio.Event()
    agent = _FakeSpawnedAgent(events=[Final(message="done")])
    factory = _FakeSpawnPort(agent=agent, abort_event=abort)
    await run_local_agent(
        store=store,
        factory=factory,
        task_id=rec.id,
        timeout_sec=None,
        transcript_storage=None,
        parent_session_id=None,
        cwd=None,
    )
    # Yield once so the cancelled watcher is reaped before we inspect the Event.
    await asyncio.sleep(0)
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert not abort.is_set()
    leaked = [
        t for t in asyncio.all_tasks()
        if (t.get_name() or "").startswith("aura-subagent-abort-watch")
    ]
    assert leaked == []


@pytest.mark.asyncio
async def test_run_local_agent_fired_abort_watcher_cancels_in_flight_run() -> None:
    """A pre-armed parent abort Event makes the watcher cancel the running child."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    # Pre-set: the watcher's wait() resolves on its first scheduled tick; yield_ticks
    # hands control to it so its cancel() lands while the stream is still in flight.
    abort = asyncio.Event()
    abort.set()
    agent = _FakeSpawnedAgent(events=[Final(message="interrupted")], yield_ticks=50)
    factory = _FakeSpawnPort(agent=agent, abort_event=abort)
    runner = asyncio.create_task(
        run_local_agent(
            store=store,
            factory=factory,
            task_id=rec.id,
            timeout_sec=None,
            transcript_storage=None,
            parent_session_id=None,
            cwd=None,
        )
    )
    # gather(return_exceptions) captures the re-raised CancelledError as a value so
    # cross-task cancellation cannot leak past this await under any scheduler timing.
    (outcome,) = await asyncio.gather(runner, return_exceptions=True)
    # The watcher firing its cancel() is what this test pins down; the exact final
    # status depends on which await point the cancel lands at, so accept either a
    # mid-stream cancel (marked) or a pre-stream cancel (still running) — both prove
    # the watcher ran. The mark-cancelled bookkeeping is pinned deterministically by
    # test_run_local_agent_cancelled_marks_cancelled_and_reraises.
    assert isinstance(outcome, asyncio.CancelledError)
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status in {"cancelled", "running"}


@pytest.mark.asyncio
async def test_run_local_agent_completed_flushes_transcript_and_metadata(
    tmp_path: Path,
) -> None:
    """A completed run with transcript_storage must persist both transcript + meta files."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    storage = _disk_storage(tmp_path)
    agent = _FakeSpawnedAgent(events=[Final(message="done")])
    factory = _FakeSpawnPort(agent=agent)
    await run_local_agent(
        store=store,
        factory=factory,
        task_id=rec.id,
        timeout_sec=None,
        transcript_storage=storage,
        parent_session_id="parent-9",
        cwd=str(tmp_path),
    )
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.transcript_path is not None
    assert refreshed.transcript_path.exists()
    meta_path = storage.subagent_metadata_path(rec.id, parent_session_id="parent-9")
    assert meta_path.exists()


@pytest.mark.asyncio
async def test_run_local_agent_timeout_zero_ceiling_is_disabled(tmp_path: Path) -> None:
    """timeout_sec<=0 disables the ceiling: resolve_timeout yields None, run completes."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    agent = _FakeSpawnedAgent(events=[Final(message="ok")])
    factory = _FakeSpawnPort(agent=agent)
    await run_local_agent(
        store=store,
        factory=factory,
        task_id=rec.id,
        timeout_sec=0.0,
        transcript_storage=None,
        parent_session_id=None,
        cwd=str(tmp_path),
    )
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.final_result == "ok"
