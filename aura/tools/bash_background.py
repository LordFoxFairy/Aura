"""bash_background — long-running bash as a fire-and-forget task.

Counterpart to :mod:`aura.tools.bash`: the blocking ``bash`` tool waits
for the subprocess to exit (≤600s), which freezes the agent turn for
``npm test`` / ``pytest`` / ``make check``. ``bash_background`` spawns
the child DETACHED, returns a ``task_id`` immediately, and streams
stdout/stderr line-by-line into a :class:`TaskRecord` owned by the same
:class:`TasksStore` the subagent task tools use. Progress is queryable
via ``task_get`` / ``task_list``; the subprocess is killable via
``task_stop`` (SIGTERM → 3s → SIGKILL).

Semantics mirror claude-code's ``LocalShellTask``:

- stdout/stderr are prefixed with ``[out] `` / ``[err] `` in the rolling
  ring (last 20 lines), so polling task_get shows interleaved output
  in the order it arrived.
- If no output for 30s and the process is still alive, we append a
  single ``[stalled?]`` marker — useful signal for "blocked on a prompt"
  without killing. Reset on new output.
- On timeout the child is taken down via SIGTERM → 3s → SIGKILL; the
  record is flipped to ``failed`` with an ``error`` noting the timeout.
- On normal exit, status becomes ``completed`` if exit_code == 0,
  ``failed`` otherwise; ``final_result`` holds the exit_code + a
  truncated tail of recent output so the LLM has something to read.

Safety: reuses :func:`check_bash_safety` inline. We don't route through
the global ``make_bash_safety_hook`` because that hook's closure matches
on ``tool.name == 'bash'`` specifically; the policy is pure (no I/O,
no state) so calling it here is cheap and keeps the two tools decoupled.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.core.permissions.bash_safety import check_bash_safety
from aura.core.permissions.matchers import exact_match_on
from aura.core.persistence import journal
from aura.core.tasks.store import TasksStore
from aura.schemas.tool import ToolError, tool_metadata
from aura.tools.bash import _is_bash_destructive

# Hard ceiling on the caller-supplied timeout. 24h — a single agent
# session should never need to park a background shell longer than
# that, and leaving it unbounded invites zombie processes outliving
# the owning Agent.
_MAX_TIMEOUT_SECONDS = 86_400
_DEFAULT_TIMEOUT_SECONDS = 3_600
_TERM_GRACE_SECONDS = 3.0
# Stall detector: if no new stdout/stderr for this long and the child is
# still alive and some output has already been seen, post a single marker.
_STALL_THRESHOLD_SECONDS = 30.0
# Poll period inside the stall detector. Small enough to notice fresh
# output quickly without busy-looping.
_STALL_POLL_SECONDS = 1.0
# Soft cap on a single captured line — pathological producers can emit
# megabyte-long lines. Truncate at this point so the ring stays bounded.
_MAX_LINE_LENGTH = 2_000
# Tail of lines retained in final_result so the LLM has something to
# grep after completion without re-polling task_get.
_FINAL_TAIL_LINES = 50


class BashBackgroundParams(BaseModel):
    command: str = Field(
        min_length=1,
        description="Shell command to run via /bin/sh -c. Same safety "
        "rules as 'bash' apply ($(), backticks, -c flag rejected).",
    )
    timeout_sec: int = Field(
        default=_DEFAULT_TIMEOUT_SECONDS,
        ge=1,
        le=_MAX_TIMEOUT_SECONDS,
        description=(
            "Seconds after which the child is killed (SIGTERM → 3s → "
            "SIGKILL). Default 3600 (1h); hard ceiling 86400 (24h)."
        ),
    )
    cwd: str | None = Field(
        default=None,
        description="Working directory for the child process. Defaults to the agent's cwd.",
    )


def _preview(args: dict[str, Any]) -> str:
    cmd = args.get("command", "")
    return f"bg: {cmd[:80]}"


def _truncate_line(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
    if len(text) > _MAX_LINE_LENGTH:
        text = text[:_MAX_LINE_LENGTH] + "… (line truncated)"
    return text


class BashBackground(BaseTool):
    """Spawn a shell command as a detached task; return its task_id immediately."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "bash_background"
    description: str = (
        "Run a shell command in the background. Returns a task_id "
        "immediately; the command keeps running after your turn ends. "
        "Use task_get(task_id) to poll status and output, task_stop(task_id) "
        "to kill. Subject to the same safety rules as 'bash'. "
        "Default timeout 3600s; hard ceiling 86400s (24h)."
    )
    args_schema: type[BaseModel] = BashBackgroundParams
    metadata: dict[str, Any] | None = tool_metadata(
        # Input-aware — shared classifier with ``bash``. A long-running
        # ``tail -f access.log`` resolves False; ``sudo systemctl stop``
        # resolves True. See ``aura.tools.bash._is_bash_destructive``.
        is_destructive=_is_bash_destructive,
        # Spawning processes and mutating the store + running-shells map
        # is not safe to batch with siblings.
        is_concurrency_safe=False,
        rule_matcher=exact_match_on("command"),
        max_result_size_chars=1000,
        args_preview=_preview,
    )
    store: TasksStore
    _running_shells: dict[str, asyncio.subprocess.Process] = PrivateAttr()
    _running_tasks: dict[str, asyncio.Task[None]] = PrivateAttr()

    def __init__(
        self,
        *,
        store: TasksStore,
        running_shells: dict[str, asyncio.subprocess.Process],
        running_tasks: dict[str, asyncio.Task[None]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(store=store, **kwargs)
        self._running_shells = running_shells
        self._running_tasks = running_tasks if running_tasks is not None else {}

    @property
    def running_shells(self) -> dict[str, asyncio.subprocess.Process]:
        return self._running_shells

    def _run(
        self,
        command: str,
        timeout_sec: int = _DEFAULT_TIMEOUT_SECONDS,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "bash_background is async-only; use `await tool.ainvoke(...)`"
        )

    async def _arun(
        self,
        command: str,
        timeout_sec: int = _DEFAULT_TIMEOUT_SECONDS,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        # Safety: identical policy to the blocking bash tool. Inlined
        # rather than routed through the hook chain because the hook
        # matches on tool.name=='bash' and we want a single source of
        # truth for the policy — ``check_bash_safety`` is pure.
        violation = check_bash_safety(command)
        if violation is not None:
            journal.write(
                "bash_background_safety_blocked",
                reason=violation.reason,
                detail=violation.detail,
                command=command,
            )
            raise ToolError(
                f"bash safety blocked: {violation.detail} "
                f"(reason={violation.reason})"
            )

        rec = self.store.create(
            description=f"bg: {command[:80]}",
            prompt=command,
            kind="shell",
            metadata={"kind": "shell", "command": command},
        )
        started_at = rec.started_at

        task = asyncio.create_task(
            _run_shell_task(
                store=self.store,
                task_id=rec.id,
                command=command,
                timeout_sec=timeout_sec,
                cwd=cwd,
                running_shells=self._running_shells,
            ),
            name=f"aura-bg-shell-{rec.id[:8]}",
        )
        self._running_tasks[rec.id] = task

        def _cleanup(t: asyncio.Task[None]) -> None:
            self._running_tasks.pop(rec.id, None)
            self._running_shells.pop(rec.id, None)
            if not t.cancelled():
                # Consume the exception so asyncio doesn't warn — the
                # watcher already wrote any failure into the record.
                t.exception()

        task.add_done_callback(_cleanup)
        return {
            "task_id": rec.id,
            "command": command,
            "status": "running",
            "started_at": started_at,
        }


async def _run_shell_task(
    *,
    store: TasksStore,
    task_id: str,
    command: str,
    timeout_sec: int,
    cwd: str | None,
    running_shells: dict[str, asyncio.subprocess.Process],
) -> None:
    """Spawn the child, stream its output, enforce timeout + stall detector.

    On any terminal path (natural exit, timeout, cancel, spawn failure),
    writes the final state into ``store`` and removes the proc handle
    from ``running_shells``. The record is only flipped if it's still in
    the ``running`` state — so a pre-emptive ``task_stop`` that already
    marked cancelled wins the race cleanly.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
    except OSError as exc:
        store.mark_failed(task_id, f"failed to spawn subprocess: {exc}")
        journal.write(
            "bash_background_spawn_failed",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return

    running_shells[task_id] = proc
    assert proc.stdout is not None and proc.stderr is not None

    out_task = asyncio.create_task(
        _drain_stream(store, task_id, proc.stdout, "[out] "),
    )
    err_task = asyncio.create_task(
        _drain_stream(store, task_id, proc.stderr, "[err] "),
    )
    stall_task = asyncio.create_task(_stall_watcher(store, task_id, proc))
    drain_fut = asyncio.gather(out_task, err_task)

    timed_out = False
    cancelled = False
    try:
        try:
            await asyncio.wait_for(drain_fut, timeout=timeout_sec)
        except TimeoutError:
            timed_out = True
            await _shutdown(proc)
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await drain_fut
        except asyncio.CancelledError:
            cancelled = True
            await _shutdown(proc)
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await drain_fut
            raise
    finally:
        stall_task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await stall_task
        # Reap so returncode is populated.
        with contextlib.suppress(TimeoutError, Exception):
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        running_shells.pop(task_id, None)
        # Only the first terminal mark wins — task_stop may have
        # already flipped the record to cancelled.
        rec = store.get(task_id)
        if rec is not None and rec.status == "running":
            if cancelled:
                store.mark_cancelled(task_id)
            elif timed_out:
                store.mark_failed(
                    task_id, f"timed out after {timeout_sec}s",
                )
            else:
                exit_code = proc.returncode
                tail_lines = list(rec.progress.recent_activities)[-_FINAL_TAIL_LINES:]
                tail = "\n".join(tail_lines)
                summary = f"exit_code={exit_code}"
                if tail:
                    summary = f"{summary}\n{tail}"
                if exit_code == 0:
                    store.mark_completed(task_id, summary)
                else:
                    store.mark_failed(task_id, summary)
        final_rec = store.get(task_id)
        journal.write(
            "bash_background_finished",
            task_id=task_id,
            status=(final_rec.status if final_rec is not None else "unknown"),
            exit_code=proc.returncode,
        )


async def _drain_stream(
    store: TasksStore,
    task_id: str,
    stream: asyncio.StreamReader,
    prefix: str,
) -> None:
    """Read ``stream`` line-by-line to EOF, recording each into the task's ring.

    Uses ``readline`` which returns up to and including the trailing ``\\n``;
    if the producer never emits a newline (binary output, abruptly-killed
    tail) ``readline`` will still return on buffer full / EOF. Exceptions
    are suppressed so a broken pipe doesn't poison the sibling drain or
    the watcher's finally block.
    """
    while True:
        try:
            raw = await stream.readline()
        except Exception:  # pragma: no cover - defensive; stream error
            return
        if not raw:
            return
        store.record_shell_line(task_id, prefix + _truncate_line(raw))


async def _stall_watcher(
    store: TasksStore,
    task_id: str,
    proc: asyncio.subprocess.Process,
) -> None:
    """Append a single ``[stalled?]`` marker if the task goes 30s+ idle.

    Re-armed on every new output line (record_shell_line bumps
    ``last_activity_at``). If the record has no activity yet — the task
    is stuck even producing its first byte — we don't fire; this matches
    claude-code's behaviour of only flagging a stall once the task has
    proven it's reached a steady state.
    """
    already_marked = False
    try:
        while proc.returncode is None:
            await asyncio.sleep(_STALL_POLL_SECONDS)
            rec = store.get(task_id)
            if rec is None:
                return
            last = rec.progress.last_activity_at
            if last is None:
                # Never produced output; not a stall in claude-code's sense.
                continue
            idle_for = time.time() - last
            if idle_for >= _STALL_THRESHOLD_SECONDS and not already_marked:
                store.record_shell_marker(task_id, "[stalled?]")
                already_marked = True
            elif idle_for < _STALL_THRESHOLD_SECONDS:
                already_marked = False
    except asyncio.CancelledError:
        return


async def _shutdown(
    proc: asyncio.subprocess.Process,
    grace: float = _TERM_GRACE_SECONDS,
) -> None:
    """SIGTERM → grace → SIGKILL. All errors swallowed (cleanup path)."""
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError, Exception):
        proc.terminate()
    with contextlib.suppress(TimeoutError, Exception):
        await asyncio.wait_for(proc.wait(), timeout=grace)
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError, Exception):
        proc.kill()
    with contextlib.suppress(TimeoutError, Exception):
        await asyncio.wait_for(proc.wait(), timeout=grace)


