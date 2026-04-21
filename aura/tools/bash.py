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
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.core.permissions.matchers import exact_match_on
from aura.schemas.tool import ToolError, tool_metadata

_DEFAULT_TIMEOUT = 30
_MAX_OUTPUT_BYTES = 30_000
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


def _cap_stream(data: str) -> tuple[str, bool]:
    """Tail-preserving byte cap; returns (possibly-truncated text, was_truncated)."""
    encoded = data.encode("utf-8")
    total = len(encoded)
    if total <= _MAX_OUTPUT_BYTES:
        return data, False
    kept = encoded[-_MAX_OUTPUT_BYTES:]
    dropped = total - len(kept)
    tail = kept.decode("utf-8", errors="replace")
    marker = f"… ({dropped} bytes truncated; showing last {_MAX_OUTPUT_BYTES} of {total})\n"
    return marker + tail, True


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
        "and truncated (True when stdout or stderr exceeded the per-stream byte cap; "
        "truncation keeps the tail and prepends a marker noting the dropped byte count)."
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

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError as exc:
            await _shutdown(proc)
            await _drain_pipes(proc)
            raise ToolError(f"timeout after {timeout}s: {exc}") from exc
        except asyncio.CancelledError:
            await _shutdown(proc)
            await _drain_pipes(proc)
            raise
        except BaseException:
            await _shutdown(proc)
            await _drain_pipes(proc)
            raise

        stdout_text = stdout_b.decode("utf-8", errors="replace")
        stderr_text = stderr_b.decode("utf-8", errors="replace")
        stdout, stdout_truncated = _cap_stream(stdout_text)
        stderr, stderr_truncated = _cap_stream(stderr_text)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": proc.returncode,
            "truncated": stdout_truncated or stderr_truncated,
        }


bash: BaseTool = Bash()
