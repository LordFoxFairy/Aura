"""bash tool — run a shell command with a timeout.

Native async implementation: uses ``asyncio.create_subprocess_shell`` so the
subprocess is owned by the event loop. On timeout *or* cancellation, the child
is taken down via a SIGTERM→SIGKILL ladder; cancellation is re-raised to
preserve structured-concurrency semantics.

Targets macOS + Linux. Windows is not supported (POSIX signal ladder).
"""

from __future__ import annotations

import asyncio
import contextlib
from asyncio.subprocess import Process
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.core.permissions.matchers import exact_match_on
from aura.schemas.tool import ToolError, tool_metadata
from aura.tools.progress import get_progress_callback

_DEFAULT_TIMEOUT = 30
_MAX_OUTPUT_BYTES = 30_000  # what the model sees — kept small to protect context
_HARD_CEILING_BYTES = 100 * 1024 * 1024  # 100 MB per stream before we kill the proc
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


def _preview(args: dict[str, Any]) -> str:
    return f"command: {args.get('command', '')}"


def _format_streamed(tail_bytes: bytes, total: int) -> tuple[str, bool]:
    """Turn the streamed-tail bytes + total count into the model-facing string.

    Mirrors the old in-memory ``_cap_stream`` return shape: if the total
    exceeded the per-stream cap, prepend a marker noting the dropped byte
    count and return ``truncated=True``. Decodes with ``errors='replace'``
    so a mid-multibyte tail slice yields U+FFFD at worst — safe for logs.
    """
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
    """Read ``stream`` to EOF while keeping only the last ``_MAX_OUTPUT_BYTES``
    in memory. If total bytes read for this stream crosses the hard ceiling,
    fire ``_shutdown(proc)`` so the child cannot keep pumping gigabytes into
    Python RSS.

    Returns ``(tail_bytes, total_bytes, killed_at_hard_ceiling)``.

    While reading, each chunk is also forwarded to the ambient progress
    callback (if one is installed — see :mod:`aura.tools.progress`) so the
    renderer can stream output to the user in near-realtime rather than
    waiting for the process to exit. Decode errors fall back to
    ``errors='replace'`` — matches the final-output decode so partial
    multibyte sequences surface as U+FFFD on both paths.

    The tail buffer is bounded to ``_MAX_OUTPUT_BYTES`` + one chunk — we
    trim after each append. This keeps a 10 GB output producer pinned at
    ~30 KB of Python-side memory per stream, plus ripple through the OS
    pipe buffer (usually 64 KB).
    """
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
            # Trim in-place to the tail window.
            del buf[: len(buf) - _MAX_OUTPUT_BYTES]
        if cb is not None:
            # Fire-and-forget: a misbehaving renderer must NOT break the
            # capture path — swallow any callback exception.
            with contextlib.suppress(Exception):
                cb(label, chunk.decode("utf-8", errors="replace"))
        if total >= _HARD_CEILING_BYTES and not killed:
            killed = True
            # Kill the producer; the stream will EOF shortly. Keep looping
            # to drain any already-buffered bytes so the final tail is
            # stable rather than abruptly chopped.
            await _shutdown(proc)
    return bytes(buf), total, killed


async def _shutdown(proc: Process, grace: float = _SHUTDOWN_GRACE) -> None:
    """Terminate then kill. Used on timeout and on cancel.

    SIGTERM first, wait up to ``grace`` seconds; if still alive, SIGKILL and
    briefly reap. Exceptions from terminate/kill are swallowed — we're in a
    cleanup path and must not raise over the original timeout/cancel.
    """
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    except Exception:  # pragma: no cover - defensive; platform-level oddity
        pass
    with contextlib.suppress(TimeoutError, Exception):
        await asyncio.wait_for(proc.wait(), timeout=grace)
        if proc.returncode is not None:
            return
    if proc.returncode is not None:
        return
    try:
        proc.kill()
    except ProcessLookupError:
        return
    except Exception:  # pragma: no cover - defensive
        pass
    with contextlib.suppress(TimeoutError, Exception):
        await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT)


async def _drain_pipes(proc: Process) -> None:
    """Drain stdout/stderr to EOF and close the subprocess transport on the
    current loop. Without this, GC may finalize the transport after the loop
    has shut down (in pytest, across tests), producing
    'Event loop is closed' unraisable warnings.
    """

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

    # Best-effort: close the private transport now, while the loop is alive.
    # asyncio.Process exposes no public transport-close API; this mirrors
    # what CPython's __del__ would do — only now rather than later.
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            transport.close()


class Bash(BaseTool):
    name: str = "bash"
    description: str = (
        "Run a shell command with a timeout. Returns stdout, stderr, exit_code, "
        "truncated (True when stdout or stderr exceeded the per-stream byte cap; "
        "tail preserved, head replaced with a marker), and killed_at_hard_ceiling "
        "(True when the child was terminated for dumping more than the per-stream "
        "100 MB hard ceiling into Python memory)."
    )
    args_schema: type[BaseModel] = BashParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_destructive=True,
        rule_matcher=exact_match_on("command"),
        args_preview=_preview,
    )

    def _run(self, command: str, timeout: int = _DEFAULT_TIMEOUT) -> dict[str, Any]:
        # Bash is async-only: cancellation must be able to kill the child
        # process, which requires the event loop to own it. Sync callers are
        # steered to ``ainvoke`` by raising here.
        raise NotImplementedError("bash is async-only; use `await bash.ainvoke(...)`")

    async def _arun(
        self, command: str, timeout: int = _DEFAULT_TIMEOUT
    ) -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise ToolError(f"failed to spawn subprocess: {exc}") from exc

        # Stream both pipes concurrently with ring-buffer tails + per-stream
        # hard-ceiling enforcement. This replaces proc.communicate(), which
        # would buffer the entire output before trimming — for a runaway
        # producer (`yes`, `cat /dev/zero`, pathological build output) that
        # buffer is RSS-unbounded. Streaming caps Python memory at ~60 KB
        # (2 × _MAX_OUTPUT_BYTES) regardless of how much the child emits.
        stdout_task = asyncio.create_task(
            _stream_capped(proc.stdout, proc, "stdout"),
        )
        stderr_task = asyncio.create_task(
            _stream_capped(proc.stderr, proc, "stderr"),
        )
        # Hold the gather-future in a local so we can explicitly consume
        # its exception on the error path. Without this, asyncio logs an
        # "exception was never retrieved" warning when wait_for cancels it.
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

        # Both streams have EOF'd — reap the process to capture returncode.
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


bash: BaseTool = Bash()
