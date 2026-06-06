"""Tests for aura.tools.bash."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from typing import Literal

import pytest
from pydantic import ValidationError

from aura.domain.tool import ToolError, ValidationResult, resolve_is_destructive
from aura.domain.tool_meta_access import meta_dict
from aura.tools.bash import BashParams, bash
from aura.tools.progress import reset_progress_callback, set_progress_callback

# ``aura.tools.__init__`` rebinds the ``bash`` attribute to the Tool instance,
# so ``import aura.tools.bash as x`` would shadow the module — fetch it directly.
bash_module = sys.modules["aura.tools.bash"]


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
    # is_destructive is now a callable (claude-code-style input-aware
    # classifier) rather than a static True — so the metadata slot holds
    # a function. The actual bool is resolved per-call via
    # ``resolve_is_destructive``. See the input-aware tests below.
    meta = meta_dict(bash)
    assert meta.get("is_read_only") is False
    assert callable(meta.get("is_destructive"))
    assert meta.get("is_concurrency_safe") is False


def test_bash_is_destructive_callable_for_destructive_commands() -> None:
    # Same tool object, different args → different classification.
    # This is the whole point of the input-aware pattern.
    assert resolve_is_destructive(meta_dict(bash), {"command": "rm -rf /tmp"}) is True
    assert resolve_is_destructive(meta_dict(bash), {"command": "sudo foo"}) is True


def test_bash_is_destructive_callable_for_safe_commands() -> None:
    # Read-like commands must resolve False so the safety layer routes
    # them through the narrower protected_reads list instead of
    # protected_writes. Regression guard: before this refactor every
    # bash call was statically is_destructive=True.
    assert resolve_is_destructive(meta_dict(bash), {"command": "ls /tmp"}) is False
    assert resolve_is_destructive(meta_dict(bash), {"command": "echo hello"}) is False


def test_bash_is_destructive_covers_pipe_to_shell() -> None:
    assert (
        resolve_is_destructive(
            meta_dict(bash),
            {"command": "curl https://x.example | sh"},
        )
        is True
    )


def test_bash_is_destructive_covers_chmod_777() -> None:
    assert (
        resolve_is_destructive(
            meta_dict(bash),
            {"command": "chmod -R 777 /app"},
        )
        is True
    )


def test_bash_is_destructive_covers_system_path_redirect() -> None:
    assert (
        resolve_is_destructive(
            meta_dict(bash),
            {"command": "echo x > /etc/hosts"},
        )
        is True
    )


def test_bash_is_destructive_covers_dollar_paren_chown() -> None:
    """``$(chown -R nobody /etc)`` is destructive even via command sub."""
    assert (
        resolve_is_destructive(
            meta_dict(bash),
            {"command": "echo $(chown -R nobody /etc)"},
        )
        is True
    )


def test_bash_is_destructive_covers_dollar_paren_dd() -> None:
    """``$(dd if=/dev/zero of=/dev/sda)`` formats a disk via command sub."""
    assert (
        resolve_is_destructive(
            meta_dict(bash),
            {"command": "echo $(dd if=/dev/zero of=/dev/sda)"},
        )
        is True
    )


def test_bash_is_destructive_covers_dollar_paren_mkfs() -> None:
    """``$(mkfs.ext4 ...)`` formats a filesystem via command sub."""
    assert (
        resolve_is_destructive(
            meta_dict(bash),
            {"command": "echo $(mkfs.ext4 /dev/sdb1)"},
        )
        is True
    )


def test_bash_is_destructive_covers_backtick_chown() -> None:
    """Backtick form of chown -R also caught."""
    assert (
        resolve_is_destructive(
            meta_dict(bash),
            {"command": "echo `chown -R nobody /etc`"},
        )
        is True
    )


def test_bash_is_destructive_covers_dd_to_device() -> None:
    """Bare ``dd of=/dev/sda`` is destructive without command sub."""
    assert (
        resolve_is_destructive(
            meta_dict(bash),
            {"command": "dd if=/dev/zero of=/dev/sda bs=1M"},
        )
        is True
    )


def test_bash_is_destructive_covers_mkfs_on_device() -> None:
    """``mkfs.ext4 /dev/sdb1`` formats a real device."""
    assert (
        resolve_is_destructive(
            meta_dict(bash),
            {"command": "mkfs.ext4 /dev/sdb1"},
        )
        is True
    )


def test_bash_is_destructive_covers_find_exec_chown() -> None:
    """``find ... -exec chown ...`` triggers per-match privilege change."""
    assert (
        resolve_is_destructive(
            meta_dict(bash),
            {"command": "find /etc -exec chown root:root {} +"},
        )
        is True
    )


def test_bash_is_destructive_missing_command_returns_false() -> None:
    # Defensive: args without a ``command`` key shouldn't throw — the
    # arg-schema layer catches that earlier, but the classifier still
    # has to be safe to call with partial inputs.
    assert resolve_is_destructive(meta_dict(bash), {}) is False


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
    meta = meta_dict(bash)
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
    out = await bash.ainvoke({"command": "printf 'x%.0s' $(seq 1 49990); printf 'TAILMARKER'"})
    stdout_bytes = out["stdout"].encode("utf-8")
    # The marker itself adds a bounded number of bytes; 200 is generous.
    assert len(stdout_bytes) <= 30_000 + 200
    assert out["stdout"].startswith("… (")
    assert out["truncated"] is True
    # Tail preserved: the final sentinel (well under 1000 bytes from end) must remain.
    assert "TAILMARKER" in out["stdout"]


async def test_bash_stderr_capped_independently() -> None:
    out = await bash.ainvoke(
        {"command": ("printf 'y%.0s' $(seq 1 49990) 1>&2; printf 'ERRTAIL' 1>&2")}
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
    out = await bash.ainvoke({"command": "printf 'x%.0s' $(seq 1 50000); echo SENTINEL"})
    assert out["truncated"] is True
    assert "SENTINEL" in out["stdout"]


async def test_bash_cancellation_kills_subprocess() -> None:
    """When the awaiting task is cancelled, the child subprocess must die."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pid") as pf:
        pid_path = pf.name
    try:
        # Write the shell's own PID, then sleep for a long time.
        cmd = f"echo $$ > {pid_path}; sleep 30"
        task = asyncio.create_task(bash.ainvoke({"command": cmd, "timeout": 60}))
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
    task = asyncio.create_task(bash.ainvoke({"command": "sleep 10", "timeout": 30}))
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


async def test_bash_huge_output_hits_hard_ceiling() -> None:
    """A command that would dump 200MB to stdout must be killed at the
    hard ceiling — we must not let the full 200MB reach Python RSS.

    Uses `yes | head -c 200000000` (200MB). Expected shape:
      - ToolResult returned (no ToolError raised)
      - truncated is True (output exceeds _MAX_OUTPUT_BYTES)
      - killed_at_hard_ceiling is True (stream hit _HARD_CEILING_BYTES)
      - stdout byte length bounded by the cap (+ tail-marker overhead)

    Skip note: if this test is flaky on heavily-loaded CI (timing of the
    hard-ceiling trip vs subprocess exit), raise the ceiling or split
    the assertion into two cases; do NOT revert — the regression window
    is explicitly the memory-blowup path.
    """
    # 200MB > 100MB hard ceiling, fits a 5-second timeout on a dev box.
    out = await bash.ainvoke({"command": "yes y | head -c 200000000", "timeout": 30})
    assert out["truncated"] is True
    assert out["killed_at_hard_ceiling"] is True
    # Stdout must be bounded; the cap marker adds a bounded preamble.
    stdout_bytes = out["stdout"].encode("utf-8")
    assert len(stdout_bytes) <= 30_000 + 200


async def test_bash_below_hard_ceiling_not_killed() -> None:
    """Output above the 30KB display cap but below the 100MB hard ceiling:
    truncation marker appears, but the stream is NOT killed at the hard
    ceiling and the process exits normally."""
    out = await bash.ainvoke({"command": "yes y | head -c 100000", "timeout": 10})
    assert out["truncated"] is True
    assert out["killed_at_hard_ceiling"] is False
    # head exits 0 after emitting N bytes; `yes` dies with SIGPIPE, but the
    # pipeline exit status is `head`'s.
    assert out["exit_code"] == 0


def test_is_destructive_blocks_command_sub_rm() -> None:
    # $(rm -rf /) wraps the destructive command in command substitution —
    # the wrapping bash will execute the inner; static check must catch.
    assert resolve_is_destructive(meta_dict(bash), {"command": "echo $(rm -rf /)"}) is True


def test_is_destructive_blocks_backtick_rm() -> None:
    assert resolve_is_destructive(meta_dict(bash), {"command": "echo `rm -rf /`"}) is True


def test_is_destructive_blocks_find_delete() -> None:
    assert resolve_is_destructive(meta_dict(bash), {"command": "find . -delete"}) is True


def test_is_destructive_blocks_find_exec_rm() -> None:
    assert resolve_is_destructive(meta_dict(bash), {"command": "find . -exec rm {} \\;"}) is True


def test_is_destructive_blocks_find_execdir_rm() -> None:
    assert resolve_is_destructive(meta_dict(bash), {"command": "find . -execdir rm {} \\;"}) is True


async def test_bash_timeout_kills_process_group() -> None:
    """Timeout on a shell that backgrounded a sleep must reap the *group*.

    Pre-fix, ``proc.terminate()`` only signalled the immediate /bin/sh —
    the backgrounded ``sleep`` lived on as an orphan. With
    ``start_new_session=True`` + ``killpg`` the whole group dies.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pid") as pf:
        pid_path = pf.name
    try:
        # Background a long sleep, capture ITS pid (the grandchild), then
        # the shell waits forever. On group-kill, the sleep dies too.
        cmd = f"sleep 100 & echo $! > {pid_path}; wait"
        with pytest.raises(ToolError, match="timeout"):
            await bash.ainvoke({"command": cmd, "timeout": 1})

        with open(pid_path) as fh:
            pid_str = fh.read().strip()
        assert pid_str, "background sleep PID not captured"
        sleep_pid = int(pid_str)
        # ~2s budget per spec — group teardown should reap inside that.
        assert await _wait_pid_dead(sleep_pid, timeout=2.0), (
            f"backgrounded sleep pid={sleep_pid} survived group kill"
        )
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(pid_path)


async def test_bash_subprocess_in_separate_session() -> None:
    """The child must NOT share the agent's process group — otherwise a
    Ctrl-C on the agent's TTY would propagate via SIGINT and kill the
    REPL too. Verify the child's pgid differs from the parent's.
    """
    out = await bash.ainvoke({"command": "ps -o pgid= -p $$ | tr -d ' '", "timeout": 5})
    child_pgid = int(out["stdout"].strip())
    parent_pgid = os.getpgid(0)
    assert child_pgid != parent_pgid, (
        f"child pgid={child_pgid} matches parent pgid={parent_pgid} — "
        "start_new_session not in effect"
    )


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
        task = asyncio.create_task(bash.ainvoke({"command": cmd, "timeout": 60}))
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


def test_validate_input_rejects_empty_command() -> None:
    result = bash.validate_input({"command": ""})
    assert isinstance(result, ValidationResult)
    assert result.invalid is True
    assert "non-empty" in result.reason


def test_validate_input_rejects_whitespace_only_command() -> None:
    result = bash.validate_input({"command": "   \t\n"})
    assert result.invalid is True


def test_validate_input_accepts_real_command() -> None:
    result = bash.validate_input({"command": "echo hi"})
    assert result.invalid is False
    assert result.reason == ""


# --------------------------------------------------------------------------- #
# Seam fakes: a controllable stand-in for asyncio.subprocess.Process so the    #
# kill-ladder / drain / spawn-error branches can be driven WITHOUT a real      #
# subprocess, sleep, or signal delivery. Each fake exposes only the surface    #
# aura.tools.bash actually touches.                                            #
# --------------------------------------------------------------------------- #


class _FakeStream:
    """A StreamReader stand-in whose read() yields scripted chunks then EOF."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, _n: int = -1) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _HangingStream:
    """A StreamReader stand-in whose read() never resolves — forces cleanup."""

    async def read(self, _n: int = -1) -> bytes:
        await asyncio.Event().wait()
        return b""


class _FakeProc:
    """Minimal Process double driving the bash kill-ladder deterministically."""

    def __init__(
        self,
        *,
        stdout: object = None,
        stderr: object = None,
        returncode: int | None = None,
        pid: int = 4242,
        wait_hangs: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.pid = pid
        self._wait_hangs = wait_hangs
        self.terminate_calls = 0
        self.kill_calls = 0

    async def wait(self) -> int:
        if self._wait_hangs and self.returncode is None:
            await asyncio.Event().wait()
        return self.returncode if self.returncode is not None else 0

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1


def _spawn_returning(
    proc: object,
    captured: dict[str, object] | None = None,
) -> Callable[..., Awaitable[object]]:
    """Build a create_subprocess_shell replacement that yields `proc`."""

    async def _factory(_command: str, **kwargs: object) -> object:
        if captured is not None:
            captured.update(kwargs)
        return proc

    return _factory


async def test_bash_progress_callback_receives_stdout_chunks() -> None:
    """Streaming progress must fan each captured chunk to the live callback —
    the REPL relies on this to render long-running output incrementally."""
    seen: list[tuple[str, str]] = []

    def _cb(label: Literal["stdout", "stderr"], text: str) -> None:
        seen.append((label, text))

    token = set_progress_callback(_cb)
    try:
        out = await bash.ainvoke({"command": "echo streamed"})
    finally:
        reset_progress_callback(token)

    assert out["exit_code"] == 0
    assert ("stdout", "streamed\n") in seen


async def test_bash_progress_callback_exception_is_swallowed() -> None:
    """A throwing progress callback must not corrupt the captured result —
    UI render bugs may not crash the agent's command execution."""

    def _cb(_label: Literal["stdout", "stderr"], _text: str) -> None:
        raise RuntimeError("render exploded")

    token = set_progress_callback(_cb)
    try:
        out = await bash.ainvoke({"command": "echo ok"})
    finally:
        reset_progress_callback(token)

    assert out["stdout"] == "ok\n"
    assert out["exit_code"] == 0


def test_bash_run_is_async_only() -> None:
    """The sync _run path is a hard contract violation — callers must use the
    async invoke; a silent sync fallback would deadlock the event loop."""
    with pytest.raises(NotImplementedError, match="async-only"):
        bash._run("echo hi")  # noqa: SLF001 — asserting the documented async-only guard


async def test_bash_spawn_oserror_becomes_toolerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed fork (ENOMEM/EMFILE) must surface as a typed ToolError, never
    leak a raw OSError that the tool-runner cannot classify."""

    async def _boom(_command: str, **_kwargs: object) -> object:
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _boom)
    with pytest.raises(ToolError, match="failed to spawn subprocess"):
        await bash.ainvoke({"command": "echo hi"})


async def test_bash_win32_uses_creationflags_not_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Windows there is no POSIX session; spawn must request a new process
    group via creationflags so the kill-ladder can still reap the child."""
    captured: dict[str, object] = {}
    proc = _FakeProc(stdout=_FakeStream([b"win\n"]), stderr=None, returncode=0)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        0x200,
        raising=False,
    )
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_shell",
        _spawn_returning(proc, captured),
    )
    out = await bash.ainvoke({"command": "echo win"})
    assert out["exit_code"] == 0
    assert captured.get("creationflags") == 0x200
    assert "start_new_session" not in captured


def _shorten_reap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the kill-ladder grace windows so fake-hang teardown stays fast."""
    monkeypatch.setattr(bash_module, "_REAP_TIMEOUT", 0.05)
    monkeypatch.setattr(bash_module, "_SHUTDOWN_GRACE", 0.05)


async def test_bash_timeout_with_null_stderr_drains_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child missing one pipe (stderr None) must still cap the live pipe and
    drain the None side on timeout without dereferencing a None stream."""
    _shorten_reap(monkeypatch)
    proc = _FakeProc(
        stdout=_HangingStream(),
        stderr=None,
        returncode=None,
        wait_hangs=True,
    )
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_shell",
        _spawn_returning(proc),
    )
    monkeypatch.setattr(os, "getpgid", lambda _pid: 9999)

    def _killpg(_pgid: int, sig: int) -> None:
        proc.returncode = -sig  # SIGTERM "lands" so the grace wait resolves

    monkeypatch.setattr(os, "killpg", _killpg)
    with pytest.raises(ToolError, match="timeout after 1s"):
        await bash.ainvoke({"command": "anything", "timeout": 1})


async def test_bash_cleanup_cancels_hanging_stream_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the cap-readers never see EOF, the timeout path must cancel them —
    a leaked reader task would keep the dead child's pipes referenced."""
    _shorten_reap(monkeypatch)
    proc = _FakeProc(
        stdout=_HangingStream(),
        stderr=_HangingStream(),
        returncode=None,
        wait_hangs=True,
    )
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_shell",
        _spawn_returning(proc),
    )
    monkeypatch.setattr(os, "getpgid", lambda _pid: 1234)
    killpg_sigs: list[int] = []

    def _killpg(_pgid: int, sig: int) -> None:
        killpg_sigs.append(sig)
        proc.returncode = -sig

    monkeypatch.setattr(os, "killpg", _killpg)
    with pytest.raises(ToolError, match="timeout"):
        await bash.ainvoke({"command": "anything", "timeout": 1})
    assert signal.SIGTERM in killpg_sigs


async def test_bash_shutdown_falls_back_to_terminate_when_getpgid_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the group lookup fails (race: child already reaped by the OS), the
    ladder must still attempt the per-process terminate()/kill() fallback."""
    _shorten_reap(monkeypatch)
    proc = _FakeProc(
        stdout=_HangingStream(),
        stderr=None,
        returncode=None,
        wait_hangs=True,
    )
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_shell",
        _spawn_returning(proc),
    )

    def _no_group(_pid: int) -> int:
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(os, "getpgid", _no_group)
    with pytest.raises(ToolError, match="timeout"):
        await bash.ainvoke({"command": "anything", "timeout": 1})
    # getpgid failed → _signal_group False → per-process terminate then kill.
    assert proc.terminate_calls >= 1
    assert proc.kill_calls >= 1


async def test_bash_shutdown_handles_getpgid_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generic OSError from getpgid (EPERM) must be treated like a failed
    group lookup, not propagated to crash the timeout teardown."""
    _shorten_reap(monkeypatch)
    proc = _FakeProc(
        stdout=_HangingStream(),
        stderr=None,
        returncode=None,
        wait_hangs=True,
    )
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_shell",
        _spawn_returning(proc),
    )

    def _eperm(_pid: int) -> int:
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(os, "getpgid", _eperm)
    with pytest.raises(ToolError, match="timeout"):
        await bash.ainvoke({"command": "anything", "timeout": 1})
    assert proc.terminate_calls >= 1


async def test_bash_shutdown_killpg_oserror_falls_back_to_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """killpg raising a non-lookup OSError means the group signal did not land;
    the ladder must escalate to the per-process kill() so nothing is orphaned."""
    _shorten_reap(monkeypatch)
    proc = _FakeProc(
        stdout=_HangingStream(),
        stderr=None,
        returncode=None,
        wait_hangs=True,
    )
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_shell",
        _spawn_returning(proc),
    )
    monkeypatch.setattr(os, "getpgid", lambda _pid: 7777)

    def _killpg_eperm(_pgid: int, _sig: int) -> None:
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", _killpg_eperm)
    with pytest.raises(ToolError, match="timeout"):
        await bash.ainvoke({"command": "anything", "timeout": 1})
    # killpg never succeeds → both ladder rungs fall back to proc methods.
    assert proc.terminate_calls >= 1
    assert proc.kill_calls >= 1


async def test_bash_shutdown_killpg_processlookup_treated_as_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """killpg raising ProcessLookupError means the group is already gone — the
    ladder treats the signal as delivered and must NOT also call terminate()."""
    _shorten_reap(monkeypatch)
    proc = _FakeProc(
        stdout=_HangingStream(),
        stderr=None,
        returncode=None,
        wait_hangs=True,
    )
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_shell",
        _spawn_returning(proc),
    )
    monkeypatch.setattr(os, "getpgid", lambda _pid: 5555)

    def _killpg_gone(_pgid: int, _sig: int) -> None:
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(os, "killpg", _killpg_gone)
    with pytest.raises(ToolError, match="timeout"):
        await bash.ainvoke({"command": "anything", "timeout": 1})
    # Group reported delivered (returns True) → no per-process fallback fires.
    assert proc.terminate_calls == 0
    assert proc.kill_calls == 0


async def test_bash_timeout_on_already_exited_child_sends_no_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the child already exited by the time the cap-reader times out (its
    pipe stayed open via an inherited fd), teardown must short-circuit and
    never signal a reaped PID — signalling a recycled PID could hit a bystander."""
    _shorten_reap(monkeypatch)
    proc = _FakeProc(stdout=_HangingStream(), stderr=None, returncode=0)
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_shell",
        _spawn_returning(proc),
    )

    def _explode_getpgid(_pid: int) -> int:
        raise AssertionError("dead child must not be signalled")

    monkeypatch.setattr(os, "getpgid", _explode_getpgid)
    with pytest.raises(ToolError, match="timeout"):
        await bash.ainvoke({"command": "anything", "timeout": 1})
    assert proc.terminate_calls == 0
    assert proc.kill_calls == 0
