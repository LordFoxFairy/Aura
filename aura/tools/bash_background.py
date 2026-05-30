"""bash_background — detached shell task with task_id polling."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Literal, TypedDict

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.application.tasks.store import TasksStore
from aura.domain.permission.matchers import exact_match_on
from aura.domain.tool import ToolMetadata
from aura.infrastructure.persistence import journal
from aura.tools.bash import is_bash_destructive

_MAX_TIMEOUT_SECONDS = 86_400
_DEFAULT_TIMEOUT_SECONDS = 3_600
_TERM_GRACE_SECONDS = 3.0
_STALL_THRESHOLD_SECONDS = 30.0
_STALL_POLL_SECONDS = 1.0
_MAX_LINE_LENGTH = 2_000
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


class BashBackgroundResult(TypedDict):
    task_id: str
    command: str
    status: Literal["running"]
    started_at: float


def _preview(args: dict[str, Any]) -> str:
    cmd = args.get("command", "")
    return f"bg: {cmd[:80]}"


def _truncate_line(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
    if len(text) > _MAX_LINE_LENGTH:
        text = text[:_MAX_LINE_LENGTH] + "… (line truncated)"
    return text


class BashBackground(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "bash_background"
    description: str = (
        "Run a shell command in the background. Returns a task_id "
        "immediately; the command keeps running after your turn ends. "
        "Use task_get(task_id) to poll status and output, task_stop(task_id) "
        "to kill. Subject to the same safety rules as 'bash'. "
        "Default timeout 3600s; hard ceiling 86400s (24h)."
    )
    args_schema: type[BaseModel] = BashBackgroundParams  # pyright: ignore[reportIncompatibleVariableOverride]  # langchain BaseTool declares args_schema as mutable ArgsSchema|None; subclass narrows widely on purpose.
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=is_bash_destructive,
        is_concurrency_safe=False,
        rule_matcher=exact_match_on("command"),
        args_preview=_preview,
        timeout_sec=None,
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
    ) -> BashBackgroundResult:
        raise NotImplementedError(
            "bash_background is async-only; use `await tool.ainvoke(...)`"
        )

    async def _arun(
        self,
        command: str,
        timeout_sec: int = _DEFAULT_TIMEOUT_SECONDS,
        cwd: str | None = None,
    ) -> BashBackgroundResult:
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
                # Consume the exception — watcher already recorded failure.
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
        with contextlib.suppress(TimeoutError, Exception):
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        running_shells.pop(task_id, None)
        # Only the first terminal mark wins.
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
    already_marked = False
    try:
        while proc.returncode is None:
            await asyncio.sleep(_STALL_POLL_SECONDS)
            rec = store.get(task_id)
            if rec is None:
                return
            last = rec.progress.last_activity_at
            if last is None:
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


