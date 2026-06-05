"""task_stop — cancels a running subagent.

Covers the happy path (running task cancels, record flips to cancelled),
error paths (unknown id, already-completed task), and the
"we do await the cancellation" invariant — the tool must not return
until the child has unwound, so the next task_get reflects the
terminal state.
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess
import inspect
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import ValidationError

from aura.application.session import AgentSession
from aura.application.tasks.run import run_task
from aura.application.tasks.spawn import SpawnContext, SubagentSpawner
from aura.application.tasks.store import TasksStore
from aura.config.schema import AuraConfig
from aura.domain.tool import ToolError
from aura.infrastructure.persistence.storage import SessionStorage
from aura.tools.task_stop import TaskStop, TaskStopParams, _preview
from tests.conftest import FakeChatModel


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


class _HangingFake(FakeChatModel):
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        await asyncio.sleep(10)
        raise RuntimeError("should not reach here")


def _hanging_factory() -> SubagentSpawner[AgentSession]:
    return SubagentSpawner(
        SpawnContext(
            parent_config=_cfg(),
            parent_model_spec="openai:gpt-4o-mini",
            build_child=AgentSession,
            model_factory=lambda: _HangingFake(),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )


@pytest.mark.asyncio
async def test_task_stop_cancels_running_task() -> None:
    store = TasksStore()
    factory = _hanging_factory()
    running: dict[str, asyncio.Task[None]] = {}
    rec = store.create(description="slow", prompt="go")
    bg = asyncio.create_task(run_task(store, factory, rec.id))
    running[rec.id] = bg
    # Let the child actually enter _agenerate so the cancel has something
    # to race against.
    await asyncio.sleep(0.05)
    tool = TaskStop(store=store, running=running)
    out = await tool.ainvoke({"task_id": rec.id})
    assert out["task_id"] == rec.id
    assert out["status"] == "cancelled"
    # run_task's CancelledError branch flipped the store.
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "cancelled"
    assert r.finished_at is not None


@pytest.mark.asyncio
async def test_task_stop_raises_on_unknown_id() -> None:
    store = TasksStore()
    tool = TaskStop(store=store, running={})
    with pytest.raises(ToolError, match="unknown task_id"):
        await tool.ainvoke({"task_id": "no-such"})


@pytest.mark.asyncio
async def test_task_stop_raises_on_already_completed_task() -> None:
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    store.mark_completed(rec.id, "done")
    tool = TaskStop(store=store, running={})
    with pytest.raises(ToolError, match="already in terminal state"):
        await tool.ainvoke({"task_id": rec.id})


@pytest.mark.asyncio
async def test_task_stop_awaits_cancellation_so_record_is_terminal() -> None:
    # Key invariant: after task_stop returns, the record MUST be in a
    # terminal state. If the tool returned before the child unwound, a
    # follow-up task_get could still see "running" and the LLM would
    # loop forever. We prove it by requiring status != "running"
    # synchronously in the same event-loop tick as the tool return.
    store = TasksStore()
    factory = _hanging_factory()
    running: dict[str, asyncio.Task[None]] = {}
    rec = store.create(description="slow", prompt="go")
    bg = asyncio.create_task(run_task(store, factory, rec.id))
    running[rec.id] = bg
    await asyncio.sleep(0.05)
    tool = TaskStop(store=store, running=running)
    await tool.ainvoke({"task_id": rec.id})
    # No extra await: the next line runs in the same tick as the return.
    r = store.get(rec.id)
    assert r is not None
    assert r.status != "running"
    assert r.status == "cancelled"


@pytest.mark.asyncio
async def test_task_stop_handles_missing_handle_via_direct_mark() -> None:
    # Race: store says running but the handle map has nothing (e.g. the
    # done-callback popped it right before task_stop looked). The tool
    # must still flip the record to cancelled rather than leave it
    # stuck in a half-running state.
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    tool = TaskStop(store=store, running={})  # no handle, still "running"
    out = await tool.ainvoke({"task_id": rec.id})
    assert out["status"] == "cancelled"
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "cancelled"


class _FakeShellProc(asyncio.subprocess.Process):
    """Stand-in for a live child process; records signal calls, no real fork.

    ``never_dies`` keeps ``returncode`` None even after ``terminate`` so the
    tool is forced down the SIGKILL escalation branch.
    """

    def __init__(
        self,
        *,
        already_exited: bool = False,
        never_dies: bool = False,
        terminate_raises: BaseException | None = None,
    ) -> None:
        self._rc: int | None = 0 if already_exited else None
        self._never_dies = never_dies
        self._terminate_raises = terminate_raises
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    @property
    def returncode(self) -> int | None:
        return self._rc

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self._terminate_raises is not None:
            raise self._terminate_raises
        if not self._never_dies:
            self._rc = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self._rc = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        return self._rc if self._rc is not None else 0


def _shell_rec(store: TasksStore, *, description: str = "sh") -> str:
    rec = store.create(description=description, prompt="p", kind="shell")
    return rec.id


async def _raise_timeout(awaitable: object, timeout: float) -> object:
    """wait_for replacement: simulate the grace window elapsing with no real sleep."""
    if inspect.iscoroutine(awaitable):
        coro: Coroutine[Any, Any, object] = awaitable
        coro.close()
    raise TimeoutError


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({}, "task_stop: ?"),
        ({"task_id": "abc"}, "task_stop: abc"),
        ({"task_id": "0123456789abcdef"}, "task_stop: 01234567"),
        ({"task_id": ""}, "task_stop: "),
    ],
)
def test_preview_truncates_id_and_tolerates_missing(
    args: dict[str, Any], expected: str
) -> None:
    """args_preview feeds the permission UI; it must never KeyError on a bad arg map."""
    assert _preview(args) == expected


@pytest.mark.parametrize("bad", ["", None, 123])
def test_params_reject_empty_or_nonstring_id(bad: object) -> None:
    """task_id is the only handle to the target; an empty/typed-wrong id must be rejected."""
    with pytest.raises(ValidationError):
        TaskStopParams.model_validate({"task_id": bad})


def test_running_maps_share_identity_with_session() -> None:
    """The tool must mutate the SAME dicts the session owns, not private copies."""
    store = TasksStore()
    running: dict[str, asyncio.Task[None]] = {}
    shells: dict[str, asyncio.subprocess.Process] = {}
    tool = TaskStop(store=store, running=running, running_shells=shells)
    assert tool.running is running
    assert tool.running_shells is shells


def test_running_shells_defaults_to_empty_dict() -> None:
    """Omitting running_shells must yield a usable empty map, not None."""
    tool = TaskStop(store=TasksStore(), running={})
    assert tool.running_shells == {}


def test_sync_run_is_async_only() -> None:
    """Cancellation awaits process unwind; a sync entrypoint cannot honor that contract."""
    tool = TaskStop(store=TasksStore(), running={})
    with pytest.raises(NotImplementedError, match="async-only"):
        tool._run("x")


@pytest.mark.asyncio
async def test_stop_shell_with_absent_proc_marks_cancelled() -> None:
    """Store says running but the shell map is empty (proc reaped early) -> still flip."""
    store = TasksStore()
    tid = _shell_rec(store)
    tool = TaskStop(store=store, running={}, running_shells={})
    out = await tool.ainvoke({"task_id": tid})
    assert out["status"] == "cancelled"
    r = store.get(tid)
    assert r is not None
    assert r.status == "cancelled"


@pytest.mark.asyncio
async def test_stop_shell_already_exited_skips_signals() -> None:
    """A process that already has a returncode must not be re-signalled, just marked."""
    store = TasksStore()
    tid = _shell_rec(store)
    proc = _FakeShellProc(already_exited=True)
    tool = TaskStop(store=store, running={}, running_shells={tid: proc})
    out = await tool.ainvoke({"task_id": tid})
    assert out["status"] == "cancelled"
    assert proc.terminate_calls == 0
    assert proc.kill_calls == 0


@pytest.mark.asyncio
async def test_stop_shell_graceful_terminate_no_kill() -> None:
    """SIGTERM that ends the child within the grace window must not escalate to SIGKILL."""
    store = TasksStore()
    tid = _shell_rec(store)
    proc = _FakeShellProc()
    tool = TaskStop(store=store, running={}, running_shells={tid: proc})
    out = await tool.ainvoke({"task_id": tid})
    assert out["status"] == "cancelled"
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 0
    r = store.get(tid)
    assert r is not None
    assert r.status == "cancelled"


@pytest.mark.asyncio
async def test_stop_shell_unresponsive_escalates_to_kill() -> None:
    """A child that ignores SIGTERM must be SIGKILLed so the slot is never left running."""
    store = TasksStore()
    tid = _shell_rec(store)
    proc = _FakeShellProc(never_dies=True)
    tool = TaskStop(store=store, running={}, running_shells={tid: proc})
    out = await tool.ainvoke({"task_id": tid})
    assert out["status"] == "cancelled"
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 1
    r = store.get(tid)
    assert r is not None
    assert r.status == "cancelled"


@pytest.mark.asyncio
async def test_stop_shell_terminate_raising_is_suppressed() -> None:
    """A racing reap can make terminate() raise; the error must not abort cancellation."""
    store = TasksStore()
    tid = _shell_rec(store)
    proc = _FakeShellProc(
        never_dies=True, terminate_raises=ProcessLookupError("gone")
    )
    tool = TaskStop(store=store, running={}, running_shells={tid: proc})
    out = await tool.ainvoke({"task_id": tid})
    assert out["status"] == "cancelled"
    # terminate raised before setting rc; escalation still runs and marks done.
    assert proc.kill_calls == 1
    r = store.get(tid)
    assert r is not None
    assert r.status == "cancelled"


@pytest.mark.asyncio
async def test_stop_shell_wait_timeout_then_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the post-SIGTERM wait times out, the tool must still SIGKILL and finish fast."""
    store = TasksStore()
    tid = _shell_rec(store)
    proc = _FakeShellProc(never_dies=True)
    monkeypatch.setattr(asyncio, "wait_for", _raise_timeout)
    tool = TaskStop(store=store, running={}, running_shells={tid: proc})
    out = await tool.ainvoke({"task_id": tid})
    assert out["status"] == "cancelled"
    assert proc.kill_calls == 1


@pytest.mark.asyncio
async def test_stop_subagent_timeout_marks_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the cancelled child overruns the grace window, the tool must force the flip itself."""
    store = TasksStore()
    rec = store.create(description="slow", prompt="go")

    async def _never() -> None:
        await asyncio.Event().wait()

    handle: asyncio.Task[None] = asyncio.create_task(_never())
    running: dict[str, asyncio.Task[None]] = {rec.id: handle}
    monkeypatch.setattr(asyncio, "wait_for", _raise_timeout)
    tool = TaskStop(store=store, running=running)
    out = await tool.ainvoke({"task_id": rec.id})
    assert out["status"] == "cancelled"
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "cancelled"
    handle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle


@pytest.mark.asyncio
async def test_stop_subagent_with_done_handle_marks_directly() -> None:
    """A handle already finished (slow done-callback) must still mark the record cancelled."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")

    async def _noop() -> None:
        return None

    handle: asyncio.Task[None] = asyncio.create_task(_noop())
    await handle
    tool = TaskStop(store=store, running={rec.id: handle})
    out = await tool.ainvoke({"task_id": rec.id})
    assert out["status"] == "cancelled"
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "cancelled"


@pytest.mark.asyncio
async def test_stop_shell_is_idempotent_second_call_raises() -> None:
    """Double-stop is a no-op then a loud error; a second SIGKILL on a dead pid is a bug."""
    store = TasksStore()
    tid = _shell_rec(store)
    proc = _FakeShellProc()
    tool = TaskStop(store=store, running={}, running_shells={tid: proc})
    first = await tool.ainvoke({"task_id": tid})
    assert first["status"] == "cancelled"
    with pytest.raises(ToolError, match="already in terminal state"):
        await tool.ainvoke({"task_id": tid})
