"""bash tool — run a shell command with a timeout.

Native async implementation: uses ``asyncio.create_subprocess_shell`` so the
subprocess is owned by the event loop. On timeout *or* cancellation, the child
is taken down via a SIGTERM→SIGKILL ladder; cancellation is re-raised to
preserve structured-concurrency semantics.

Regex residual gap: the ``_DANGEROUS_COMMAND_PATTERNS`` table catches the
common destructive idioms (rm -rf, $(rm), find -delete, etc.) but cannot
defeat shell-AST-only obfuscations — alias indirection, eval of decoded
strings, glob/brace expansion that constructs the literal at runtime, or
heredoc-fed scripts. claude-code's bashCommandSafety.ts uses a real AST
for those cases; the bash_safety policy layer covers some via shlex, but
true defense-in-depth is the permission hook + workspace sandbox.

Targets macOS + Linux. Windows is not supported (POSIX signal ladder).

Input-aware destructiveness
---------------------------

``bash`` carries an input-aware ``is_destructive`` classifier on its
metadata (``_is_bash_destructive``) rather than a static ``True``. The
permission hook resolves it per-invocation via
:func:`aura.schemas.tool.resolve_is_destructive`, so ``bash("ls /tmp")``
resolves False (safety layer treats it as a read) while ``bash("rm -rf
/tmp")`` resolves True. This matches claude-code's ``isDestructive(input)``
method pattern — the same tool object classifies differently depending
on what the LLM asked it to do. Static flags can't express that.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
import subprocess
import sys
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


# Dangerous-command pattern table. Kept in sync with the widget-side table
# in ``aura.cli.permission_bash._DANGEROUS_PATTERNS`` — same regexes, same
# semantics. Duplicated rather than imported to avoid a tools → cli layer
# dependency (the CLI layer imports from tools, not the other way round).
# If a new dangerous pattern is added to the CLI widget, mirror it here so
# the permission hook's safety-direction classification stays consistent
# with the UI's warning banner.
_DANGEROUS_COMMAND_PATTERNS: tuple[re.Pattern[str], ...] = (
    # rm -r / rm -f / rm -rf (and combined short flags like -rfv).
    re.compile(r"(?:^|[;&|\s])rm\s+(?:-[a-zA-Z]*[rRf][a-zA-Z]*)"),
    # sudo — any elevation is a destructive intent signal.
    re.compile(r"(?:^|[;&|\s])sudo\b"),
    # curl|sh / wget|sh (pipe-to-shell install).
    re.compile(r"\b(?:curl|wget)\b[^|]*\|\s*(?:sh|bash|zsh|ksh|sudo)\b"),
    # chmod 777 / chmod -R 777 / chmod 0777.
    re.compile(r"\bchmod\s+(?:-R\s+)?0?777\b"),
    # Redirects to system paths: > /etc/..., >> /usr/..., > /bin/...
    re.compile(r">{1,2}\s*/(?:etc|usr|bin|sbin|boot|sys|proc)(?:/|\b)"),
    # Writes to raw block devices: > /dev/sda, dd of=/dev/sd*.
    re.compile(r"(?:>{1,2}\s*|\bof=)\s*/dev/(?:sd[a-z]|nvme|hd[a-z]|vd[a-z])"),
    # rm combined with background operator: rm ... & (not &&).
    re.compile(r"\brm\b[^;&\n]*&(?!&)"),
    # $(rm ...) command substitution wrapping a destructive command.
    re.compile(r"\$\(\s*rm\s+(?:-[a-zA-Z]*[rRf][a-zA-Z]*)"),
    # `rm ...` backtick substitution wrapping a destructive command.
    re.compile(r"`\s*rm\s+(?:-[a-zA-Z]*[rRf][a-zA-Z]*)"),
    # find ... -delete (recursive delete via find action).
    re.compile(r"\bfind\b[^|;&\n]*\s-delete\b"),
    # find ... -exec rm / -execdir rm (per-match rm exec).
    re.compile(r"\bfind\b[^|;&\n]*\s-exec(?:dir)?\s+rm\b"),
)


def _is_bash_destructive(args: dict[str, Any]) -> bool:
    """Input-aware ``is_destructive`` classifier for bash.

    Mirrors claude-code's ``isDestructive(input)`` method pattern. The
    same ``bash`` tool is safe for ``ls /tmp`` and destructive for
    ``rm -rf /`` — the classification has to look at the command string,
    not at a tool-wide flag.

    Returns True iff the command matches any entry in
    ``_DANGEROUS_COMMAND_PATTERNS``. A non-string or missing command
    falls through to False — the tool's own arg-schema validation
    catches malformed inputs before dispatch, and treating a missing
    command as destructive would noise-up the safety layer.

    Consumed by :func:`aura.schemas.tool.resolve_is_destructive` from
    the permission hook; never called directly. Raising from here would
    fail-safe to True via the resolver (ambiguity ≙ destructive), so
    the implementation stays defensive — no external calls, no I/O,
    just regex scans.
    """
    command = args.get("command", "")
    if not isinstance(command, str) or not command:
        return False
    return any(pat.search(command) for pat in _DANGEROUS_COMMAND_PATTERNS)


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


def _signal_group(proc: Process, sig: int) -> bool:
    """Send ``sig`` to the subprocess's process group. Returns True if the
    signal was delivered to the group, False if the group lookup failed (in
    which case the caller should fall back to per-process signalling).

    Using killpg ensures grandchildren spawned by the shell (``sleep 30 &``)
    get the signal too — single-process terminate would leave them orphaned.
    """
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
    """Terminate then kill the subprocess group. Used on timeout and on cancel.

    SIGTERM first to the whole process group (so backgrounded children die
    too), wait up to ``grace`` seconds; if still alive, SIGKILL the group
    and briefly reap. Exceptions from terminate/kill are swallowed — we're
    in a cleanup path and must not raise over the original timeout/cancel.
    """
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
        # Input-aware: a command like ``ls`` resolves to False (read-like,
        # safety layer picks the protected_reads list); ``rm -rf`` resolves
        # to True (destructive, protected_writes list). See
        # ``_is_bash_destructive`` and ``resolve_is_destructive``.
        is_destructive=_is_bash_destructive,
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
        # Detach the child into its own session/process group so a Ctrl-C
        # delivered to the agent's TTY doesn't propagate to it, and so that
        # _shutdown can killpg() the entire group (children of the shell
        # included) rather than just the immediate proc.
        spawn_kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
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
