"""bash tool — run a shell command with a timeout."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
import subprocess
import sys
from asyncio.subprocess import Process
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from aura.domain.permission.matchers import exact_match_on
from aura.domain.tool import ToolError, ToolMetadata, ValidationResult
from aura.tools.base import Tool
from aura.tools.progress import get_progress_callback

_DEFAULT_TIMEOUT = 30
_MAX_OUTPUT_BYTES = 30_000
_HARD_CEILING_BYTES = 100 * 1024 * 1024
_STREAM_CHUNK = 8192
_SHUTDOWN_GRACE = 0.5
_REAP_TIMEOUT = 2.0


class BashParams(BaseModel):
    command: str = Field(description="Shell command to run via /bin/sh -c.")
    timeout: int = Field(
        default=_DEFAULT_TIMEOUT,
        ge=1,
        le=600,
        description="Timeout in seconds (1-600).",
    )


class BashResult(TypedDict):
    stdout: str
    stderr: str
    exit_code: int | None
    truncated: bool
    killed_at_hard_ceiling: bool


def _preview(args: dict[str, Any]) -> str:
    return f"command: {args.get('command', '')}"


_DANGEROUS_COMMAND_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|[;&|\s])rm\s+(?:-[a-zA-Z]*[rRf][a-zA-Z]*)"),
    re.compile(r"(?:^|[;&|\s])sudo\b"),
    re.compile(r"\b(?:curl|wget)\b[^|]*\|\s*(?:sh|bash|zsh|ksh|sudo)\b"),
    re.compile(r"\bchmod\s+(?:-R\s+)?0?777\b"),
    re.compile(r">{1,2}\s*/(?:etc|usr|bin|sbin|boot|sys|proc)(?:/|\b)"),
    re.compile(r"(?:>{1,2}\s*|\bof=)\s*/dev/(?:sd[a-z]|nvme|hd[a-z]|vd[a-z])"),
    re.compile(r"\brm\b[^;&\n]*&(?!&)"),
    re.compile(
        r"\$\(\s*"
        r"(?:rm\s+(?:-[a-zA-Z]*[rRf][a-zA-Z]*)"
        r"|dd\s+[^)]*\bof=/dev/"
        r"|mkfs(?:\.\w+)?\b"
        r"|chown\s+-[a-zA-Z]*R"
        r"|chmod\s+(?:-R\s+)?0?777)"
    ),
    re.compile(
        r"`\s*"
        r"(?:rm\s+(?:-[a-zA-Z]*[rRf][a-zA-Z]*)"
        r"|dd\s+[^`]*\bof=/dev/"
        r"|mkfs(?:\.\w+)?\b"
        r"|chown\s+-[a-zA-Z]*R"
        r"|chmod\s+(?:-R\s+)?0?777)"
    ),
    re.compile(r"\bfind\b[^|;&\n]*\s-delete\b"),
    re.compile(
        r"\bfind\b[^|;&\n]*\s-exec(?:dir)?\s+"
        r"(?:rm|chown|chmod|dd|mv|truncate)\b"
    ),
    re.compile(r"\bdd\b[^|;&\n]*\bof=/dev/(?:sd[a-z]|nvme|hd[a-z]|vd[a-z])"),
    re.compile(r"\bmkfs(?:\.\w+)?\s+/dev/"),
)


def is_bash_destructive(args: dict[str, Any]) -> bool:
    command = args.get("command", "")
    if not isinstance(command, str) or not command:
        return False
    return any(pat.search(command) for pat in _DANGEROUS_COMMAND_PATTERNS)


def _format_streamed(tail_bytes: bytes, total: int) -> tuple[str, bool]:
    if total <= _MAX_OUTPUT_BYTES:
        return tail_bytes.decode("utf-8", errors="replace"), False
    dropped = total - len(tail_bytes)
    tail = tail_bytes.decode("utf-8", errors="replace")
    marker = (
        f"… ({dropped} bytes truncated; showing last {_MAX_OUTPUT_BYTES} "
        f"of {total})\n"
    )
    return marker + tail, True


async def _stream_capped(
    stream: asyncio.StreamReader | None,
    proc: Process,
    label: Literal["stdout", "stderr"] = "stdout",
) -> tuple[bytes, int, bool]:
    if stream is None:
        return b"", 0, False
    buf = bytearray()
    total = 0
    killed = False
    cb = get_progress_callback()
    while True:
        try:
            chunk = await stream.read(_STREAM_CHUNK)
        except Exception:  # pragma: no cover - defensive; stream error
            break
        if not chunk:
            break
        total += len(chunk)
        buf.extend(chunk)
        if len(buf) > _MAX_OUTPUT_BYTES:
            del buf[: len(buf) - _MAX_OUTPUT_BYTES]
        if cb is not None:
            with contextlib.suppress(Exception):
                cb(label, chunk.decode("utf-8", errors="replace"))
        if total >= _HARD_CEILING_BYTES and not killed:
            killed = True
            await _shutdown(proc)
    return bytes(buf), total, killed


def _signal_group(proc: Process, sig: int) -> bool:
    if proc.returncode is not None:
        return True
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return False
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return True


async def _shutdown(proc: Process, grace: float = _SHUTDOWN_GRACE) -> None:
    if proc.returncode is not None:
        return
    if not _signal_group(proc, signal.SIGTERM):
        with contextlib.suppress(ProcessLookupError, Exception):
            proc.terminate()
    with contextlib.suppress(TimeoutError, Exception):
        await asyncio.wait_for(proc.wait(), timeout=grace)
        if proc.returncode is not None:
            return
    if proc.returncode is not None:
        return
    if not _signal_group(proc, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError, Exception):
            proc.kill()
    with contextlib.suppress(TimeoutError, Exception):
        await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT)


async def _drain_pipes(proc: Process) -> None:
    # Close transport on the current loop so GC doesn't finalize it after loop shutdown.
    async def _read_eof(stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            await stream.read()

    with contextlib.suppress(TimeoutError, Exception):
        await asyncio.wait_for(
            asyncio.gather(
                _read_eof(proc.stdout),
                _read_eof(proc.stderr),
                return_exceptions=True,
            ),
            timeout=_REAP_TIMEOUT,
        )

    # Private CPython asyncio attr; absent on some transports, so feature-detect.
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            transport.close()


class Bash(Tool):
    name: str = "bash"
    description: str = (
        "Run a shell command with a timeout. Returns stdout, stderr, exit_code, "
        "truncated (True when stdout or stderr exceeded the per-stream byte cap; "
        "tail preserved, head replaced with a marker), and killed_at_hard_ceiling "
        "(True when the child was terminated for dumping more than the per-stream "
        "100 MB hard ceiling into Python memory)."
    )
    args_schema: type[BaseModel] = BashParams  # pyright: ignore[reportIncompatibleVariableOverride]  # langchain BaseTool declares args_schema as mutable ArgsSchema|None; subclass narrows widely on purpose.
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=is_bash_destructive,
        is_concurrency_safe=False,
        rule_matcher=exact_match_on("command"),
        args_preview=_preview,
        timeout_sec=None,
    )

    def validate_input(self, args: dict[str, Any]) -> ValidationResult:
        command = args.get("command", "")
        if not isinstance(command, str) or command.strip() == "":
            return ValidationResult(
                invalid=True,
                reason="bash requires a non-empty command",
            )
        return ValidationResult(invalid=False)

    def _run(self, command: str, timeout: int = _DEFAULT_TIMEOUT) -> BashResult:
        raise NotImplementedError("bash is async-only; use `await bash.ainvoke(...)`")

    async def _arun(
        self, command: str, timeout: int = _DEFAULT_TIMEOUT
    ) -> BashResult:
        # New session/group: isolate Ctrl-C from agent TTY and enable killpg of the whole group.
        # Heterogeneous by-platform: creationflags(int) | start_new_session(bool).
        spawn_kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            # Feature-detection: CREATE_NEW_PROCESS_GROUP is Windows-only on subprocess.
            spawn_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            spawn_kwargs["start_new_session"] = True
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **spawn_kwargs,
            )
        except OSError as exc:
            raise ToolError(f"failed to spawn subprocess: {exc}") from exc

        stdout_task = asyncio.create_task(
            _stream_capped(proc.stdout, proc, "stdout"),
        )
        stderr_task = asyncio.create_task(
            _stream_capped(proc.stderr, proc, "stderr"),
        )
        # Local ref so wait_for's cancel path doesn't leave the gather's exception unretrieved.
        gather_fut = asyncio.gather(stdout_task, stderr_task)

        async def _cleanup() -> None:
            for t in (stdout_task, stderr_task):
                if not t.done():
                    t.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await gather_fut

        try:
            stdout_result, stderr_result = await asyncio.wait_for(
                gather_fut, timeout=timeout
            )
        except TimeoutError as exc:
            await _cleanup()
            await _shutdown(proc)
            await _drain_pipes(proc)
            raise ToolError(f"timeout after {timeout}s: {exc}") from exc
        except asyncio.CancelledError:
            await _cleanup()
            await _shutdown(proc)
            await _drain_pipes(proc)
            raise
        except BaseException:
            await _cleanup()
            await _shutdown(proc)
            await _drain_pipes(proc)
            raise

        with contextlib.suppress(TimeoutError, Exception):
            await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT)

        stdout_tail, stdout_total, stdout_killed = stdout_result
        stderr_tail, stderr_total, stderr_killed = stderr_result
        stdout, stdout_truncated = _format_streamed(stdout_tail, stdout_total)
        stderr, stderr_truncated = _format_streamed(stderr_tail, stderr_total)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": proc.returncode,
            "truncated": stdout_truncated or stderr_truncated,
            "killed_at_hard_ceiling": stdout_killed or stderr_killed,
        }


bash: Bash = Bash()
