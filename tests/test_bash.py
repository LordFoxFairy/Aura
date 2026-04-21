"""Tests for aura.tools.bash."""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import time

import pytest
from pydantic import ValidationError

from aura.schemas.tool import ToolError
from aura.tools.bash import BashParams, bash


def _pid_alive(pid: int) -> bool:
    """Return True if a process with `pid` is still alive (POSIX)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _wait_pid_dead(pid: int, timeout: float = 5.0) -> bool:
    """Poll until the PID no longer exists or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        await asyncio.sleep(0.05)
    return not _pid_alive(pid)


async def test_bash_success_echo() -> None:
    out = await bash.ainvoke({"command": "echo hello"})
    assert out["stdout"] == "hello\n"
    assert out["exit_code"] == 0
    assert out["stderr"] == ""


async def test_bash_nonzero_exit_returns_ok_true() -> None:
    out = await bash.ainvoke({"command": "exit 42"})
    assert out["exit_code"] == 42


async def test_bash_nonzero_exit_captures_stderr() -> None:
    out = await bash.ainvoke({"command": "echo err >&2; exit 1"})
    assert "err" in out["stderr"]
    assert out["exit_code"] == 1


async def test_bash_pipe_works() -> None:
    out = await bash.ainvoke({"command": "echo hello | tr a-z A-Z"})
    assert "HELLO" in out["stdout"]


async def test_bash_timeout() -> None:
    with pytest.raises(ToolError, match="timeout"):
        await bash.ainvoke({"command": "sleep 5", "timeout": 1})


def test_bash_capability_flags() -> None:
    meta = bash.metadata or {}
    assert meta.get("is_read_only") is False
    assert meta.get("is_destructive") is True
    assert meta.get("is_concurrency_safe") is False


def test_bash_no_check_permissions_method() -> None:
    assert not hasattr(bash, "check_permissions")


def test_bash_timeout_validation_rejects_too_large() -> None:
    with pytest.raises(ValidationError):
        BashParams(command="x", timeout=601)


def test_bash_timeout_validation_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        BashParams(command="x", timeout=0)


def test_bash_default_timeout() -> None:
    assert BashParams(command="x").timeout == 30


def test_bash_metadata_includes_matcher_and_preview() -> None:
    meta = bash.metadata or {}
    matcher = meta.get("rule_matcher")
    assert callable(matcher)
    # Matcher is exact-match on command.
    assert matcher({"command": "npm test"}, "npm test") is True
    assert matcher({"command": "rm -rf /"}, "npm test") is False

    preview = meta.get("args_preview")
    assert callable(preview)
    assert preview({"command": "ls"}) == "command: ls"


async def test_bash_stdout_capped_at_30k() -> None:
    # Produce 50_000 bytes of 'x' on stdout ending with a recognizable tail.
    out = await bash.ainvoke(
        {"command": "printf 'x%.0s' $(seq 1 49990); printf 'TAILMARKER'"}
    )
    stdout_bytes = out["stdout"].encode("utf-8")
    # The marker itself adds a bounded number of bytes; 200 is generous.
    assert len(stdout_bytes) <= 30_000 + 200
    assert out["stdout"].startswith("… (")
    assert out["truncated"] is True
    # Tail preserved: the final sentinel (well under 1000 bytes from end) must remain.
    assert "TAILMARKER" in out["stdout"]


async def test_bash_stderr_capped_independently() -> None:
    out = await bash.ainvoke(
        {
            "command": (
                "printf 'y%.0s' $(seq 1 49990) 1>&2; printf 'ERRTAIL' 1>&2"
            )
        }
    )
    stderr_bytes = out["stderr"].encode("utf-8")
    assert len(stderr_bytes) <= 30_000 + 200
    assert out["stderr"].startswith("… (")
    assert out["truncated"] is True
    assert out["stdout"] == ""
    assert "ERRTAIL" in out["stderr"]


async def test_bash_small_output_not_truncated() -> None:
    out = await bash.ainvoke({"command": "echo hello"})
    assert out["truncated"] is False
    assert "… (" not in out["stdout"]


async def test_bash_exactly_at_limit_not_truncated() -> None:
    # Exactly 30_000 bytes of 'x' (no trailing newline).
    out = await bash.ainvoke({"command": "printf 'x%.0s' $(seq 1 30000)"})
    assert len(out["stdout"].encode("utf-8")) == 30_000
    assert out["truncated"] is False
    assert not out["stdout"].startswith("… (")


async def test_bash_tail_preserved_in_truncation() -> None:
    out = await bash.ainvoke(
        {"command": "printf 'x%.0s' $(seq 1 50000); echo SENTINEL"}
    )
    assert out["truncated"] is True
    assert "SENTINEL" in out["stdout"]


async def test_bash_cancellation_kills_subprocess() -> None:
    """When the awaiting task is cancelled, the child subprocess must die."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pid") as pf:
        pid_path = pf.name
    try:
        # Write the shell's own PID, then sleep for a long time.
        cmd = f"echo $$ > {pid_path}; sleep 30"
        task = asyncio.create_task(
            bash.ainvoke({"command": cmd, "timeout": 60})
        )
        # Wait until the PID file has content (subprocess actually started).
        deadline = time.monotonic() + 5.0
        pid_str = ""
        while time.monotonic() < deadline:
            try:
                with open(pid_path) as fh:
                    pid_str = fh.read().strip()
                if pid_str:
                    break
            except FileNotFoundError:
                pass
            await asyncio.sleep(0.05)
        assert pid_str, "subprocess did not start / PID not captured"
        pid = int(pid_str)
        assert _pid_alive(pid), "subprocess should be alive before cancel"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # After cancellation, the child must be reaped.
        assert await _wait_pid_dead(pid), f"orphan process pid={pid} still alive"
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(pid_path)


async def test_bash_cancellation_propagates_not_swallowed() -> None:
    """CancelledError must reach the awaiter — not be swallowed by the tool."""
    task = asyncio.create_task(
        bash.ainvoke({"command": "sleep 10", "timeout": 30})
    )
    # Give the subprocess a moment to actually start.
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_bash_timeout_still_kills_subprocess() -> None:
    """On timeout, the child subprocess must be killed — not orphaned."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pid") as pf:
        pid_path = pf.name
    try:
        cmd = f"echo $$ > {pid_path}; sleep 10"
        with pytest.raises(ToolError, match="timeout"):
            await bash.ainvoke({"command": cmd, "timeout": 1})

        with open(pid_path) as fh:
            pid_str = fh.read().strip()
        assert pid_str, "subprocess did not start"
        pid = int(pid_str)
        assert await _wait_pid_dead(pid, timeout=3.0), (
            f"process pid={pid} still alive after timeout"
        )
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(pid_path)


async def test_bash_sigterm_race_cleaned_up_with_sigkill() -> None:
    """Child ignores SIGTERM — ladder must escalate to SIGKILL."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pid") as pf:
        pid_path = pf.name
    try:
        # Trap & ignore SIGTERM; the background sleep holds the shell alive.
        # `wait` will wake on signals, so we loop the wait to truly ignore TERM.
        cmd = (
            f"trap '' TERM; echo $$ > {pid_path}; "
            "sleep 30 & child=$!; "
            "while kill -0 $child 2>/dev/null; do wait $child; done"
        )
        task = asyncio.create_task(
            bash.ainvoke({"command": cmd, "timeout": 60})
        )
        deadline = time.monotonic() + 5.0
        pid_str = ""
        while time.monotonic() < deadline:
            try:
                with open(pid_path) as fh:
                    pid_str = fh.read().strip()
                if pid_str:
                    break
            except FileNotFoundError:
                pass
            await asyncio.sleep(0.05)
        assert pid_str, "subprocess did not start"
        pid = int(pid_str)
        assert _pid_alive(pid)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The ladder uses 0.5s grace; allow generous buffer for SIGKILL reap.
        assert await _wait_pid_dead(pid, timeout=4.0), (
            f"SIGTERM-ignoring process pid={pid} survived the kill ladder"
        )
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(pid_path)
