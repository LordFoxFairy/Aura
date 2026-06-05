"""bash_background — fire-and-forget long-running shell as a TaskRecord.

The tool spawns a shell subprocess DETACHED from the tool invocation,
returns a ``task_id`` immediately, and writes rolling output into the
parent AgentSession's ``TasksStore`` so ``task_get`` / ``task_stop`` /
``task_list`` can observe / kill it. These tests exercise the lifecycle
axes: fast completion, long-running polling, timeout kill, task_stop
kill, stream prefixes, safety rejection, and ring-buffer boundedness.

All tests use real subprocesses (``/bin/sh -c``) because mocking asyncio
subprocess plumbing is worse than just running short commands. Tests
are gated on POSIX semantics (sh available, SIGTERM delivery) — same
target platforms as the blocking ``bash`` tool.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable
from typing import Any

import pytest

from aura.application.hooks.bash_safety import make_bash_safety_hook
from aura.application.loop_state import LoopState
from aura.application.tasks.store import TasksStore
from aura.domain.task import SHELL_RECENT_ACTIVITIES_CAP
from aura.domain.tool import ToolResult
from aura.tools import bash_background as bg_mod
from aura.tools.bash_background import (
    BashBackground,
    _drain_stream,
    _preview,
    _shutdown,
    _truncate_line,
)
from aura.tools.task_get import TaskGet
from aura.tools.task_stop import TaskStop


def _sc(outcome: object) -> ToolResult | None:
    """Extract the short-circuit result from Replace Outcome."""
    from aura.domain.permission.outcome import Replace
    if isinstance(outcome, Replace):
        return outcome.result
    return getattr(outcome, "short_circuit", None)



def _make_tool() -> (
    tuple[
        BashBackground,
        TasksStore,
        dict[str, asyncio.subprocess.Process],
        dict[str, asyncio.Task[None]],
    ]
):
    store = TasksStore()
    running_shells: dict[str, asyncio.subprocess.Process] = {}
    running_tasks: dict[str, asyncio.Task[None]] = {}
    tool = BashBackground(
        store=store,
        running_shells=running_shells,
        running_tasks=running_tasks,
    )
    return tool, store, running_shells, running_tasks


async def _wait_for_terminal(
    store: TasksStore, task_id: str, timeout: float = 10.0,
) -> Any:
    """Poll until record.status != 'running'; return the record."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        rec = store.get(task_id)
        if rec is not None and rec.status != "running":
            return rec
        await asyncio.sleep(0.02)
    last = store.get(task_id)
    raise AssertionError(
        f"task {task_id[:8]} never reached terminal; status="
        f"{last.status if last is not None else 'missing'}",
    )


@pytest.mark.asyncio
async def test_short_command_completes_with_stdout_captured() -> None:
    tool, store, _, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "echo hello-aura"})
    assert out["status"] == "running"
    assert out["command"] == "echo hello-aura"
    task_id = out["task_id"]
    # Wait for the watcher to fully flip the record.
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id)
    assert rec.kind == "shell"
    assert rec.status == "completed"
    # final_result holds exit_code + tail.
    assert "exit_code=0" in rec.final_result
    assert "hello-aura" in rec.final_result
    assert any("[out] hello-aura" in a for a in rec.progress.recent_activities)


@pytest.mark.asyncio
async def test_long_running_returns_immediately_and_status_progresses() -> None:
    # sleep 3 then echo — tool must return while the child is still
    # running; a polled task_get must surface the running status.
    tool, store, _, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "sleep 2; echo ok"})
    task_id = out["task_id"]
    # Immediately after the tool returns: still running.
    rec = store.get(task_id)
    assert rec is not None
    assert rec.status == "running"
    # Let it finish.
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id, timeout=10.0)
    assert rec.status == "completed"
    assert any("[out] ok" in a for a in rec.progress.recent_activities)


@pytest.mark.asyncio
async def test_timeout_kills_child_and_marks_failed() -> None:
    tool, store, _, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "sleep 60", "timeout_sec": 1})
    task_id = out["task_id"]
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id, timeout=10.0)
    assert rec.status == "failed"
    assert rec.error is not None
    assert "timed out after 1s" in rec.error


@pytest.mark.asyncio
async def test_task_stop_kills_shell_subprocess() -> None:
    tool, store, running_shells, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "sleep 60"})
    task_id = out["task_id"]
    # Give the spawn a tick so the process is registered.
    await asyncio.sleep(0.1)
    assert task_id in running_shells
    stop = TaskStop(
        store=store,
        running=running_tasks,
        running_shells=running_shells,
    )
    stop_out = await stop.ainvoke({"task_id": task_id})
    assert stop_out["status"] == "cancelled"
    rec = store.get(task_id)
    assert rec is not None
    assert rec.status == "cancelled"
    # Let the watcher finish so the test doesn't leak pending tasks. The
    # watcher's finally block may raise CancelledError once SIGKILL
    # closes the pipes and its awaits unwind — that's expected.
    for t in list(running_tasks.values()):
        with contextlib.suppress(asyncio.CancelledError):
            await t


@pytest.mark.asyncio
async def test_stderr_captured_with_err_prefix() -> None:
    tool, store, _, running_tasks = _make_tool()
    # ``>&2`` redirects to stderr in /bin/sh.
    out = await tool.ainvoke(
        {"command": "echo fail-msg >&2"},
    )
    task_id = out["task_id"]
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id)
    assert rec.status == "completed"
    assert any(
        a.startswith("[err] ") and "fail-msg" in a
        for a in rec.progress.recent_activities
    ), rec.progress.recent_activities


@pytest.mark.asyncio
async def test_stdout_captured_with_out_prefix() -> None:
    tool, store, _, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "echo greet"})
    task_id = out["task_id"]
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id)
    assert rec.status == "completed"
    assert any(
        a.startswith("[out] ") and "greet" in a
        for a in rec.progress.recent_activities
    ), rec.progress.recent_activities


@pytest.mark.asyncio
async def test_safety_rejects_command_substitution() -> None:
    """Safety is a hook-chain concern: dangerous commands routed through
    the bash safety hook are short-circuited with a ToolResult and never
    reach the tool's ``_arun``. The tool itself is a dumb executor after
    the Option-A refactor (see ``test_bash_background_permission.py``
    for the per-channel parity assertions)."""
    tool, store, _, _ = _make_tool()
    hook = make_bash_safety_hook()
    for bad in ("echo $(whoami)", "echo `whoami`", "bash -c 'echo hi'"):
        outcome = await hook(
            tool=tool,
            args={"command": bad},
            state=LoopState(),
        )
        sc = _sc(outcome)
        assert sc is not None, bad
        assert sc.ok is False
        assert sc.error is not None
        assert "command substitution" in sc.error
    # No task records should have been created — the hook blocks the
    # call before the tool runs.
    assert store.list() == []


@pytest.mark.asyncio
async def test_progress_ring_bounded_at_shell_cap() -> None:
    tool, store, _, running_tasks = _make_tool()
    # Emit 40 lines of stdout — well past the 20-line cap. ``seq`` is
    # POSIX and produces deterministic numbered lines without needing
    # $((...)) arithmetic (which trips the bash-safety rule).
    out = await tool.ainvoke({
        "command": "seq 1 40",
        "timeout_sec": 30,
    })
    task_id = out["task_id"]
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id, timeout=15.0)
    assert rec.status == "completed"
    # Ring is bounded to 20 entries. line_count tracks the monotonic total.
    assert len(rec.progress.recent_activities) <= SHELL_RECENT_ACTIVITIES_CAP
    assert rec.progress.line_count >= 40


@pytest.mark.asyncio
async def test_nonzero_exit_marks_failed_with_exit_code() -> None:
    tool, store, _, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "exit 7"})
    task_id = out["task_id"]
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id)
    assert rec.status == "failed"
    assert "exit_code=7" in (rec.error or "")


@pytest.mark.asyncio
async def test_task_list_filters_by_shell_kind() -> None:
    # Cross-check: task_list's new ``kind`` filter surfaces only shell
    # tasks when requested. Mixes a subagent (via store.create) and a
    # shell (via bash_background) in the same store.
    from aura.tools.task_list import TaskList

    tool, store, _, running_tasks = _make_tool()
    # Inject a fake subagent record directly into the store so we don't
    # need to spin up a real SubagentSpawner here.
    store.create(description="sub", prompt="p")
    out = await tool.ainvoke({"command": "echo sh"})
    await asyncio.gather(*running_tasks.values())
    listing = await TaskList(store=store).ainvoke({"kind": "shell"})
    kinds = {t["kind"] for t in listing["tasks"]}
    assert kinds == {"shell"}
    descs = [t["description"] for t in listing["tasks"]]
    assert any(d.startswith("bg: ") for d in descs)
    # And the subagent filter sees only the injected one.
    listing2 = await TaskList(store=store).ainvoke({"kind": "subagent"})
    assert {t["kind"] for t in listing2["tasks"]} == {"subagent"}
    assert out["task_id"] not in {t["id"] for t in listing2["tasks"]}


@pytest.mark.asyncio
async def test_task_get_surfaces_shell_kind_and_line_count() -> None:
    tool, store, _, running_tasks = _make_tool()
    out = await tool.ainvoke({"command": "echo a; echo b; echo c"})
    task_id = out["task_id"]
    await asyncio.gather(*running_tasks.values())
    rec = await _wait_for_terminal(store, task_id)
    assert rec.status == "completed"
    get_out = await TaskGet(store=store).ainvoke({"task_id": task_id})
    assert get_out["kind"] == "shell"
    assert get_out["progress"]["line_count"] >= 3
    assert get_out["progress"]["tool_count"] == 0  # shell never fires tool events


# --------------------------------------------------------------------------- #
# Mocked-seam boundary tests. The block above shells out for real; below we    #
# inject a fake process at ``create_subprocess_exec`` to drive the watcher's   #
# error / timeout / shutdown branches deterministically with no real sleeps,   #
# subprocesses, or signals.                                                    #
# --------------------------------------------------------------------------- #


class _FakeTransport:
    """Minimal ``SubprocessTransport`` surface read by ``Process``.

    ``Process.returncode`` / ``terminate`` / ``kill`` all delegate here, so a
    fake transport is the seam that lets ``_FakeProc`` be a genuine
    ``Process`` subclass (mypy-clean) without a real child.
    """

    def __init__(self, exit_code: int, *, started_exited: bool) -> None:
        self._exit_code = exit_code
        self._returncode: int | None = exit_code if started_exited else None
        self.terminated = False
        self.killed = False

    def get_returncode(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def exit_now(self) -> None:
        self._returncode = self._exit_code


class _FakeProc(asyncio.subprocess.Process):
    """A real ``Process`` backed by a fake transport (no fork, no signals).

    ``stdout`` / ``stderr`` are live :class:`asyncio.StreamReader` instances so
    ``_drain_stream`` runs its true readline path; ``returncode`` and the
    signal methods route through :class:`_FakeTransport`.
    """

    def __init__(
        self,
        *,
        stdout_lines: list[bytes],
        stderr_lines: list[bytes],
        exit_code: int,
        hang: bool = False,
    ) -> None:
        self._fake = _FakeTransport(exit_code, started_exited=not hang)
        self.stdout: asyncio.StreamReader = asyncio.StreamReader()
        self.stderr: asyncio.StreamReader = asyncio.StreamReader()
        for line in stdout_lines:
            self.stdout.feed_data(line)
        for line in stderr_lines:
            self.stderr.feed_data(line)
        if not hang:
            self.stdout.feed_eof()
            self.stderr.feed_eof()

    @property
    def returncode(self) -> int | None:
        return self._fake.get_returncode()

    @property
    def terminated(self) -> bool:
        return self._fake.terminated

    @property
    def killed(self) -> bool:
        return self._fake.killed

    async def wait(self) -> int:
        self._fake.exit_now()
        return self._fake.get_returncode() or 0

    def terminate(self) -> None:
        self._fake.terminate()
        self._fake.exit_now()
        self.stdout.feed_eof()
        self.stderr.feed_eof()

    def kill(self) -> None:
        self._fake.kill()
        self._fake.exit_now()


def _patch_spawn(
    monkeypatch: pytest.MonkeyPatch, proc: _FakeProc,
) -> None:
    """Route ``create_subprocess_exec`` to yield ``proc`` (no real fork)."""

    async def _fake_spawn(*_args: object, **_kwargs: object) -> _FakeProc:
        return proc

    monkeypatch.setattr(
        "aura.tools.bash_background.asyncio.create_subprocess_exec",
        _fake_spawn,
    )


@pytest.mark.asyncio
async def test_sync_run_raises_async_only() -> None:
    """The sync ``_run`` entrypoint must hard-fail: this tool is await-only,
    so a synchronous LangChain invocation must not silently no-op."""
    tool, _, _, _ = _make_tool()
    with pytest.raises(NotImplementedError, match="async-only"):
        tool._run("echo hi")


@pytest.mark.asyncio
async def test_spawn_oserror_marks_failed_without_leaking_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kernel-level spawn failure (e.g. ENOENT / EMFILE) must surface as a
    ``failed`` record, not a crashed watcher or an orphaned running_shells
    entry — the user still needs a terminal status to poll."""
    store = TasksStore()
    running_shells: dict[str, asyncio.subprocess.Process] = {}

    async def _boom(*_args: object, **_kwargs: object) -> object:
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(
        "aura.tools.bash_background.asyncio.create_subprocess_exec", _boom,
    )
    rec = store.create(description="bg", prompt="x", kind="shell")
    await bg_mod._run_shell_task(
        store=store,
        task_id=rec.id,
        command="echo x",
        timeout_sec=5,
        cwd=None,
        running_shells=running_shells,
    )
    done = store.get(rec.id)
    assert done is not None
    assert done.status == "failed"
    assert done.error is not None
    assert "failed to spawn subprocess" in done.error
    assert rec.id not in running_shells


@pytest.mark.asyncio
async def test_nonzero_exit_via_fake_proc_marks_failed_with_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean-exiting child with a non-zero code must be ``failed`` and the
    summary must carry both the code and the captured output tail so the
    poller sees *why* it failed without a second round-trip."""
    proc = _FakeProc(
        stdout_lines=[b"line-1\n"],
        stderr_lines=[b"oops\n"],
        exit_code=3,
    )
    _patch_spawn(monkeypatch, proc)
    store = TasksStore()
    running_shells: dict[str, asyncio.subprocess.Process] = {}
    rec = store.create(description="bg", prompt="x", kind="shell")
    await bg_mod._run_shell_task(
        store=store,
        task_id=rec.id,
        command="x",
        timeout_sec=5,
        cwd=None,
        running_shells=running_shells,
    )
    done = store.get(rec.id)
    assert done is not None
    assert done.status == "failed"
    assert "exit_code=3" in (done.error or "")
    assert "[out] line-1" in (done.error or "")
    assert "[err] oops" in (done.error or "")
    assert rec.id not in running_shells


@pytest.mark.asyncio
async def test_zero_exit_via_fake_proc_marks_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path with a fake child: exit 0 must mark ``completed`` and
    register/deregister the live shell handle so task_stop can find it."""
    proc = _FakeProc(
        stdout_lines=[b"ok\n"], stderr_lines=[], exit_code=0,
    )
    _patch_spawn(monkeypatch, proc)
    store = TasksStore()
    running_shells: dict[str, asyncio.subprocess.Process] = {}
    rec = store.create(description="bg", prompt="x", kind="shell")
    await bg_mod._run_shell_task(
        store=store,
        task_id=rec.id,
        command="x",
        timeout_sec=5,
        cwd=None,
        running_shells=running_shells,
    )
    done = store.get(rec.id)
    assert done is not None
    assert done.status == "completed"
    assert "exit_code=0" in (done.final_result or "")
    assert proc.terminated is False  # clean exit never escalates to SIGTERM


@pytest.mark.asyncio
async def test_timeout_path_shuts_down_child_and_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that never closes its pipes must hit the ``wait_for`` deadline,
    get escalated through ``_shutdown`` (SIGTERM), and land as ``failed`` with
    a timeout message — never hang the watcher forever."""
    proc = _FakeProc(
        stdout_lines=[b"partial\n"],
        stderr_lines=[],
        exit_code=0,
        hang=True,
    )
    _patch_spawn(monkeypatch, proc)

    real_wait_for = asyncio.wait_for

    async def _fast_wait_for(
        awaitable: Awaitable[object], timeout: float | None,
    ) -> object:
        # The drain gather is the only call with the user timeout (≥5): force a
        # miss. Inner proc.wait() calls (timeout 2.0 / grace) stay real.
        if timeout is not None and timeout >= 5:
            raise TimeoutError
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(
        "aura.tools.bash_background.asyncio.wait_for", _fast_wait_for,
    )
    store = TasksStore()
    running_shells: dict[str, asyncio.subprocess.Process] = {}
    rec = store.create(description="bg", prompt="x", kind="shell")
    await bg_mod._run_shell_task(
        store=store,
        task_id=rec.id,
        command="x",
        timeout_sec=5,
        cwd=None,
        running_shells=running_shells,
    )
    done = store.get(rec.id)
    assert done is not None
    assert done.status == "failed"
    assert "timed out after 5s" in (done.error or "")
    assert proc.terminated is True
    assert rec.id not in running_shells


@pytest.mark.asyncio
async def test_cancellation_marks_cancelled_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """task_stop cancels the watcher task: the watcher must shut the child
    down, mark the record ``cancelled``, and re-raise CancelledError so the
    asyncio machinery still sees the cancellation."""
    proc = _FakeProc(
        stdout_lines=[], stderr_lines=[], exit_code=0, hang=True,
    )
    _patch_spawn(monkeypatch, proc)
    store = TasksStore()
    running_shells: dict[str, asyncio.subprocess.Process] = {}
    rec = store.create(description="bg", prompt="x", kind="shell")
    task = asyncio.create_task(
        bg_mod._run_shell_task(
            store=store,
            task_id=rec.id,
            command="x",
            timeout_sec=3600,
            cwd=None,
            running_shells=running_shells,
        ),
    )
    # Let the watcher reach its wait_for(drain) suspension point.
    for _ in range(5):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    done = store.get(rec.id)
    assert done is not None
    assert done.status == "cancelled"
    assert proc.terminated is True
    assert rec.id not in running_shells


@pytest.mark.asyncio
async def test_first_terminal_mark_wins_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idempotency: if the record is already terminal (a prior task_stop won
    the race), the watcher's finally block must NOT overwrite it with an
    exit-code verdict — the first terminal mark is authoritative."""
    proc = _FakeProc(
        stdout_lines=[b"late\n"], stderr_lines=[], exit_code=1,
    )
    _patch_spawn(monkeypatch, proc)
    store = TasksStore()
    running_shells: dict[str, asyncio.subprocess.Process] = {}
    rec = store.create(description="bg", prompt="x", kind="shell")
    store.mark_cancelled(rec.id)  # someone already terminated it
    await bg_mod._run_shell_task(
        store=store,
        task_id=rec.id,
        command="x",
        timeout_sec=5,
        cwd=None,
        running_shells=running_shells,
    )
    done = store.get(rec.id)
    assert done is not None
    # Stays cancelled — the exit_code=1 failure verdict must NOT win.
    assert done.status == "cancelled"
    assert done.error is None


def test_truncate_line_caps_overlong_output() -> None:
    """A pathological single line (no newline, megabytes wide) must be capped
    so one runaway ``yes``-style line cannot blow up the progress ring."""
    raw = ("A" * 5000).encode("utf-8")
    out = _truncate_line(raw)
    assert out.endswith("… (line truncated)")
    assert len(out) <= 2_000 + len("… (line truncated)")


def test_truncate_line_strips_eol_and_decodes_invalid_bytes() -> None:
    """Trailing CRLF must be stripped and invalid UTF-8 must degrade via the
    replacement char — a binary-spewing child must never crash the drainer."""
    assert _truncate_line(b"hello\r\n") == "hello"
    assert _truncate_line(b"") == ""
    assert "�" in _truncate_line(b"\xff\xfe")


def test_preview_truncates_long_command() -> None:
    """The args preview (shown in the permission prompt) must clamp the
    command to 80 chars so a giant one-liner cannot flood the prompt UI."""
    assert _preview({}) == "bg: "
    long = "echo " + "x" * 200
    preview = _preview({"command": long})
    assert preview.startswith("bg: echo ")
    assert len(preview) == len("bg: ") + 80


@pytest.mark.asyncio
async def test_drain_stream_records_each_line_then_stops_on_eof() -> None:
    """The drainer must emit one prefixed activity per line and return cleanly
    on EOF — never spin, never drop the last partial-less line."""
    store = TasksStore()
    rec = store.create(description="bg", prompt="x", kind="shell")
    reader = asyncio.StreamReader()
    reader.feed_data(b"first\n")
    reader.feed_data(b"second\n")
    reader.feed_eof()
    await _drain_stream(store, rec.id, reader, "[out] ")
    got = store.get(rec.id)
    assert got is not None
    assert "[out] first" in got.progress.recent_activities
    assert "[out] second" in got.progress.recent_activities
    assert got.progress.line_count == 2


@pytest.mark.asyncio
async def test_drain_stream_noop_when_task_missing() -> None:
    """A line arriving for an already-evicted task_id must be a silent no-op,
    not a KeyError — store.record_shell_line guards the missing record."""
    store = TasksStore()
    reader = asyncio.StreamReader()
    reader.feed_data(b"orphan\n")
    reader.feed_eof()
    await _drain_stream(store, "no-such-id", reader, "[out] ")
    assert store.list() == []


@pytest.mark.asyncio
async def test_stall_watcher_marks_once_when_idle_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long-idle child must get exactly one ``[stalled?]`` marker (not one
    per poll) so the user is nudged once, not spammed every second."""
    proc = _FakeProc(
        stdout_lines=[], stderr_lines=[], exit_code=0, hang=True,
    )
    store = TasksStore()
    rec = store.create(description="bg", prompt="x", kind="shell")
    store.record_shell_line(rec.id, "[out] tick")  # sets last_activity_at
    last = rec.progress.last_activity_at
    assert last is not None

    real_sleep = asyncio.sleep

    async def _yield_sleep(_seconds: float) -> None:
        await real_sleep(0)  # advance one loop turn with zero delay

    monkeypatch.setattr(
        "aura.tools.bash_background.asyncio.sleep", _yield_sleep,
    )
    monkeypatch.setattr(
        "aura.tools.bash_background.time.time", lambda: last + 60.0,
    )  # 60s past last_activity_at → idle_for >> 30s threshold
    watcher = asyncio.create_task(bg_mod._stall_watcher(store, rec.id, proc))
    for _ in range(5):
        await real_sleep(0)
    watcher.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await watcher
    got = store.get(rec.id)
    assert got is not None
    # Marked exactly once despite many idle polls — ``already_marked`` guard.
    assert got.progress.recent_activities.count("[stalled?]") == 1


@pytest.mark.asyncio
async def test_stall_watcher_returns_on_missing_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the record vanishes mid-watch the stall loop must return, not raise —
    a dropped task must not leave a wedged background coroutine."""
    proc = _FakeProc(
        stdout_lines=[], stderr_lines=[], exit_code=0, hang=True,
    )
    store = TasksStore()

    real_sleep = asyncio.sleep

    async def _yield_sleep(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(
        "aura.tools.bash_background.asyncio.sleep", _yield_sleep,
    )
    # No record created → store.get returns None on the first poll → return.
    await asyncio.wait_for(
        bg_mod._stall_watcher(store, "ghost-id", proc), timeout=1.0,
    )


@pytest.mark.asyncio
async def test_shutdown_noop_when_already_exited() -> None:
    """If the child already exited, ``_shutdown`` must short-circuit and never
    signal a reaped PID (which would raise ProcessLookupError)."""
    proc = _FakeProc(  # hang=False ⇒ already exited (returncode == 0)
        stdout_lines=[], stderr_lines=[], exit_code=0,
    )
    assert proc.returncode == 0
    await _shutdown(proc, grace=0.05)
    assert proc.terminated is False
    assert proc.killed is False


@pytest.mark.asyncio
async def test_shutdown_terminate_then_kill_when_sigterm_ignored() -> None:
    """A child that ignores SIGTERM must be escalated to SIGKILL after the
    grace window — the watcher must not leak an un-reapable process."""

    class _SigtermIgnorer(_FakeProc):
        def terminate(self) -> None:
            self._fake.terminate()  # swallow SIGTERM: returncode stays None

        async def wait(self) -> int:
            # Resolve only after kill(); before that, time out the grace wait.
            if self._fake.killed:
                self._fake.exit_now()
                return 0
            await asyncio.Event().wait()  # never resolves → grace times out
            return 0

    proc = _SigtermIgnorer(
        stdout_lines=[], stderr_lines=[], exit_code=0, hang=True,
    )
    await _shutdown(proc, grace=0.02)
    assert proc.terminated is True
    assert proc.killed is True
