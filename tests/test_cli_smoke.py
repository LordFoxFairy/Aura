"""Black-box subprocess smoke tests for the installed ``aura`` entry point.

Every other test in the suite imports Aura modules directly — none of
them actually invoke the ``aura`` binary the user types at the shell.
That gap shipped nine release tags (0.7.0 through 0.9.0) without any
CI catching:

- a broken ``project.scripts`` entry point,
- an import-time crash at startup,
- argparse wiring that writes to stderr on ``--help``,
- a REPL that ignores ``/exit`` or EOF,
- a ``--version`` fast-path that accidentally depends on config.

These tests close that gap by spawning a real subprocess and driving
it the way a human would. Nothing is mocked; the agent binary under
test is the one ``pip install .`` just installed.
"""

from __future__ import annotations

import contextlib
import errno
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

# pty is POSIX-only. On Windows CI we skip the whole pty test section.
try:
    import pty as _pty
except ImportError:  # pragma: no cover — Windows only
    _pty = None  # type: ignore[assignment]

# Repo root == parent of tests/. Resolved once so individual tests don't
# recompute it; every subprocess is spawned with ``cwd=_REPO_ROOT`` so
# ``uv run`` finds the right project, even when pytest is invoked from
# elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Semver shape. Resilient to version bumps (0.9.0 today, 0.9.1 tomorrow).
# Accepts an optional trailing newline because the ``--version`` action
# always appends one on its own.
_VERSION_LINE_RE = re.compile(r"^aura \d+\.\d+\.\d+\s*$")


def _aura_invocation() -> list[str] | None:
    """Return the best available command vector to run ``aura``.

    Preference order:

    1. ``uv run aura`` — mirrors how the user invokes the CLI locally.
       Picks up the project-managed venv automatically.
    2. ``python -m aura.cli.__main__`` — fallback that still exercises
       the module path, for environments where ``uv`` isn't on PATH
       (rare in our CI, but cheap insurance).

    Returns ``None`` only when BOTH are unusable — in which case the
    environment truly cannot run Aura and tests should skip.
    """
    if shutil.which("uv") is not None:
        return ["uv", "run", "aura"]
    # Fall back to module invocation. The file is importable from any
    # cwd because the aura package is already on sys.path when pytest
    # is running.
    try:
        import aura.cli.__main__  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    return [sys.executable, "-m", "aura.cli.__main__"]


@pytest.fixture(scope="module")
def aura_binary() -> Sequence[str]:
    """Module-scoped cached invocation vector.

    Skips the whole module only if there's truly no way to run aura.
    This is a deliberate "environment hard-fail" exit — not a
    per-feature skip — because every test in this file assumes the
    binary exists and is invocable.
    """
    invocation = _aura_invocation()
    if invocation is None:
        pytest.skip(
            "neither `uv` nor `python -m aura.cli.__main__` is usable in "
            "this environment — cannot run CLI smoke tests"
        )
    return invocation


def _run(
    argv: Sequence[str],
    *,
    timeout: float = 30.0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` with sane test defaults (text, capture, repo cwd).

    ``env=None`` means "inherit the current environment". Callers that
    need to add vars should copy ``os.environ`` first and mutate.
    """
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_REPO_ROOT),
        env=env,
        check=False,
    )


def _has_startup_block(stdout: str, stderr: str) -> bool:
    """Return True if aura failed to reach the REPL at startup.

    In a bare dev env without any LangChain provider SDK (or without
    the matching API key env var), ``build_agent`` raises
    ``MissingProviderDependencyError`` / ``MissingCredentialError``
    before the REPL ever takes control of stdin. When that happens the
    ``/exit`` and EOF tests can't meaningfully prove anything about the
    REPL's input handling — skip them instead of reporting a false
    failure.

    The two error class names are the contract we watch for; if either
    is renamed this helper must be updated in lock-step.
    """
    blob = f"{stdout}\n{stderr}"
    return (
        "MissingProviderDependencyError" in blob
        or "MissingCredentialError" in blob
    )


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


def test_aura_version_prints_semver(aura_binary: Sequence[str]) -> None:
    """``aura --version`` exits 0 and prints ``aura X.Y.Z``."""
    result = _run([*aura_binary, "--version"])
    assert result.returncode == 0, (
        f"--version exited {result.returncode}; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # The stdout may be preceded by `uv run` warnings on the LAST line.
    # argparse's --version action prints exactly one line; pick it out.
    version_lines = [
        line for line in result.stdout.splitlines() if line.startswith("aura ")
    ]
    assert version_lines, (
        f"no 'aura <version>' line found in stdout: {result.stdout!r}"
    )
    # Latest matching line is argparse's — uv warnings go to stderr, not
    # stdout, but defensive in case that changes upstream.
    assert _VERSION_LINE_RE.match(version_lines[-1] + "\n"), (
        f"version line {version_lines[-1]!r} does not match "
        f"expected shape 'aura X.Y.Z'"
    )


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_aura_help_exits_cleanly(aura_binary: Sequence[str]) -> None:
    """``aura --help`` exits 0, writes usage to stdout, stderr stays clean.

    argparse / click / typer should never write to stderr on --help.
    A stderr line here means either a warning from an eager import or a
    logger firing before ``main`` set itself up properly — both are
    regressions the contributor should see before merging.
    """
    result = _run([*aura_binary, "--help"])
    assert result.returncode == 0, (
        f"--help exited {result.returncode}; stderr={result.stderr!r}"
    )
    combined_out = result.stdout.lower()
    assert "usage:" in combined_out, (
        f"--help stdout missing 'usage:' prefix: {result.stdout!r}"
    )
    # stderr check — tolerate ``uv run`` venv-mismatch warnings (they
    # come from uv itself, not aura). Everything else is a regression.
    for line in result.stderr.splitlines():
        if "VIRTUAL_ENV" in line or line.strip().startswith("warning:"):
            continue
        if not line.strip():
            continue
        raise AssertionError(
            f"--help wrote unexpected stderr line: {line!r}\n"
            f"full stderr: {result.stderr!r}"
        )


def test_aura_help_has_core_sections(aura_binary: Sequence[str]) -> None:
    """``--help`` advertises every flag registered by ``_make_parser``.

    This is an integration sanity check on the argparse wiring: if a
    contributor adds a flag but forgets to document it (or renames one
    without updating callers), the help output must still surface it.
    The canonical flag list was read from aura/cli/__main__.py's
    ``_make_parser``; keeping the two in sync is a manual invariant we
    accept as the cost of having an end-to-end assertion.
    """
    result = _run([*aura_binary, "--help"])
    assert result.returncode == 0
    out = result.stdout
    # The actual flags defined in aura/cli/__main__.py:_make_parser.
    # NOT --config / --provider / --model — those don't exist (Aura
    # uses AURA_CONFIG env var + /model slash command instead).
    for flag in ("--version", "--verbose", "--log", "--bypass-permissions"):
        assert flag in out, (
            f"expected flag {flag!r} in --help output; got:\n{out}"
        )


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


def _popen_repl(aura_binary: Sequence[str]) -> subprocess.Popen[str]:
    """Spawn an aura REPL with stdio pipes ready for ``.communicate``."""
    return subprocess.Popen(
        list(aura_binary),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(_REPO_ROOT),
    )


def _assert_no_traceback(stream: str, *, label: str) -> None:
    """Fail if *stream* contains a Python traceback.

    ``Traceback (most recent call last):`` is the canonical first line
    Python emits before an unhandled exception's frames. If the REPL
    exits cleanly on ``/exit`` or EOF it must never print one — any
    AuraError surfaces via the ``_fail_startup`` red-print path which
    does NOT include the traceback marker.
    """
    assert "Traceback (most recent call last)" not in stream, (
        f"{label} contained a Python traceback (REPL exit should be clean):\n"
        f"{stream}"
    )


def test_aura_exits_on_slash_exit_via_stdin(
    aura_binary: Sequence[str],
) -> None:
    """Feed ``/exit\\n`` on stdin; REPL must exit cleanly within 15s.

    ``ExitCommand`` in aura/core/commands/builtin.py returns
    ``CommandResult(kind="exit")``; run_repl_async sees that and
    ``return``s out of its while-loop, after which main() returns 0.
    No traceback, bounded time.
    """
    proc = _popen_repl(aura_binary)
    try:
        stdout, stderr = proc.communicate(input="/exit\n", timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise AssertionError(
            f"aura did not exit within 15s on /exit\n"
            f"stdout={stdout!r}\nstderr={stderr!r}"
        ) from None

    if _has_startup_block(stdout, stderr):
        pytest.skip(
            "aura could not reach the REPL in this environment "
            "(missing provider SDK or credential); /exit path "
            "cannot be exercised end-to-end here"
        )

    _assert_no_traceback(stderr, label="stderr")
    _assert_no_traceback(stdout, label="stdout")
    # On a clean /exit, main() returns 0. We accept only 0 here because
    # any non-zero is either a startup failure (handled above by the
    # skip) or a true REPL crash worth flagging.
    assert proc.returncode == 0, (
        f"aura exited with {proc.returncode} on /exit; "
        f"stdout={stdout!r} stderr={stderr!r}"
    )


def test_aura_exits_on_ctrl_d(aura_binary: Sequence[str]) -> None:
    """Close stdin with no input; REPL must exit cleanly.

    The REPL's ``_default_input`` path raises ``EOFError`` on closed
    stdin; the while-loop catches it and returns. This is the Ctrl+D
    contract from a non-TTY (piped) subprocess — symmetric to a real
    user pressing Ctrl+D at the prompt.
    """
    proc = _popen_repl(aura_binary)
    try:
        stdout, stderr = proc.communicate(input="", timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise AssertionError(
            f"aura did not exit within 15s on EOF\n"
            f"stdout={stdout!r}\nstderr={stderr!r}"
        ) from None

    if _has_startup_block(stdout, stderr):
        pytest.skip(
            "aura could not reach the REPL in this environment "
            "(missing provider SDK or credential); EOF path "
            "cannot be exercised end-to-end here"
        )

    _assert_no_traceback(stderr, label="stderr")
    _assert_no_traceback(stdout, label="stdout")
    assert proc.returncode == 0, (
        f"aura exited with {proc.returncode} on EOF; "
        f"stdout={stdout!r} stderr={stderr!r}"
    )


# ---------------------------------------------------------------------------
# --version + bad config
# ---------------------------------------------------------------------------


def test_aura_with_bad_config_fails_gracefully(
    aura_binary: Sequence[str],
) -> None:
    """``--version`` must succeed even when ``AURA_CONFIG`` is garbage.

    The ``--version`` action in argparse calls ``sys.exit(0)`` during
    ``parser.parse_args()``, which is BEFORE main() ever reaches
    ``load_config()``. So a broken ``AURA_CONFIG`` path must not
    influence the version fast-path — if it does, that's a regression
    worth exposing (it would mean some eager import is secretly loading
    config at module import time).
    """
    env = dict(os.environ)
    env["AURA_CONFIG"] = "/nonexistent/definitely-not-a-real-path.json"
    result = _run([*aura_binary, "--version"], env=env)
    assert result.returncode == 0, (
        f"--version should bypass config loading; got rc={result.returncode}. "
        f"If --version now requires config, either fix main() to keep the "
        f"fast-path or update this test.\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    # Still prints a valid semver line.
    version_lines = [
        line for line in result.stdout.splitlines() if line.startswith("aura ")
    ]
    assert version_lines, (
        f"no 'aura <version>' line in stdout despite rc=0: "
        f"{result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# pty-backed REPL tests
# ---------------------------------------------------------------------------
#
# The tests above drive aura with a plain subprocess pipe — enough to catch
# startup / argparse / exit-path bugs, but pt's ``PromptSession`` needs a
# real pseudo-terminal to take control of stdin. These tests spawn the
# binary under a pty so the interactive render path (welcome Panel,
# bottom_toolbar, shift+tab keybinding) actually executes.
#
# If the environment can't reach the REPL (missing provider SDK or
# credential, same guard as the stdin tests), every pty test skips with
# a clear reason — never a false negative.


# Keystroke bytes sent on the pty master. Shift+Tab is the ANSI CSI Z
# sequence; every VT-compatible terminal emits it, and pt's
# ``s-tab`` binding matches on these exact bytes.
_SHIFT_TAB = b"\x1b[Z"

# Substrings that would indicate the legacy "print mode-change line to
# scrollback" regression. Any of these appearing in pty output means the
# silent-feedback contract was broken — the bottom_toolbar is the ONLY
# approved mode indicator.
_LEGACY_MODE_SPAM = (
    "mode: accept_edits (press shift+tab",
    "mode: plan (press shift+tab",
    "mode: default (press shift+tab",
)

_PTY_SKIP_REASON_NO_PTY = (
    "pty module unavailable on this platform (Windows?) — "
    "pty-based REPL tests are POSIX-only"
)


def _pty_supported() -> bool:
    """Return True only on platforms where stdlib ``pty`` is importable.

    Linux and macOS qualify; Windows does not. Used to skip the whole
    pty-test section cleanly rather than fail to import ``pty``.
    """
    return _pty is not None


class _PtyAura:
    """Context manager that spawns ``aura`` under a pseudo-terminal.

    The contract is "small, honest, and cleanup-safe": we wire a pty
    pair to the child's stdio, read from the master fd with
    ``select``-based timeouts, and on ``__exit__`` we always close the
    master and kill the child if it's still alive. Every interactive
    test in this file goes through this class so the cleanup invariant
    is enforced in exactly one place.

    Usage::

        with _PtyAura(aura_cmd) as ptya:
            ptya.expect("aura>")
            ptya.send("/exit\\r")
            output = ptya.read_all()
    """

    def __init__(
        self,
        cmd: Sequence[str],
        *,
        timeout_boot: float = 10.0,
        rows: int = 24,
        cols: int = 80,
    ) -> None:
        self._cmd = list(cmd)
        self._timeout_boot = timeout_boot
        self._rows = rows
        self._cols = cols
        self._master_fd: int | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._buffer = bytearray()

    def __enter__(self) -> _PtyAura:
        assert _pty is not None  # guarded by _pty_supported at call sites
        master, slave = _pty.openpty()
        # Best-effort window size. pt's layout (bottom_toolbar, prompt
        # width) depends on this; default 80x24 matches a normal xterm.
        try:
            import fcntl
            import struct
            import termios

            fcntl.ioctl(
                slave,
                termios.TIOCSWINSZ,
                struct.pack("HHHH", self._rows, self._cols, 0, 0),
            )
        except Exception:  # noqa: BLE001 — window-size is cosmetic
            pass

        env = dict(os.environ)
        # pt emits colour escapes when TERM looks real; we want those in
        # the captured stream so tests can assert on box-drawing bytes.
        env["TERM"] = "xterm-256color"
        env["PYTHONIOENCODING"] = "utf-8"
        # Defensive: drop any inherited NO_COLOR/FORCE_COLOR so output is
        # deterministic across local dev and CI.
        env.pop("NO_COLOR", None)

        try:
            self._proc = subprocess.Popen(
                self._cmd,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                cwd=str(_REPO_ROOT),
                env=env,
                close_fds=True,
                start_new_session=True,
            )
        finally:
            # Parent never writes to the slave end.
            os.close(slave)
        self._master_fd = master
        return self

    # -- low-level I/O ----------------------------------------------------

    def _read_some(self, timeout: float) -> bytes:
        """Read whatever bytes are available within ``timeout`` seconds.

        Returns b"" on timeout / EOF / already-closed master. The pty
        master yields EIO on Linux when the slave closes; macOS raises
        EIO similarly — both treated as EOF. Post-``close_master`` the
        fd is ``None`` and this returns b"" unconditionally so
        ``read_all`` can still be called from the test to drain what
        was already buffered.
        """
        if self._master_fd is None:
            return b""
        try:
            ready, _, _ = select.select([self._master_fd], [], [], timeout)
        except (ValueError, OSError):
            return b""
        if not ready:
            return b""
        try:
            return os.read(self._master_fd, 4096)
        except OSError as exc:
            if exc.errno in (errno.EIO, errno.EBADF):
                return b""
            raise

    def expect(self, substring: str, timeout: float = 5.0) -> str:
        """Accumulate output until ``substring`` appears or timeout elapses.

        Returns the current decoded buffer (never raises on miss — the
        caller decides whether absence is a failure). Bytes stay in
        ``self._buffer`` so ``read_all`` can see everything captured
        before AND after this call.
        """
        needle = substring.encode("utf-8", errors="replace")
        deadline = time.monotonic() + timeout
        while needle not in self._buffer and time.monotonic() < deadline:
            chunk = self._read_some(min(0.5, max(0.05, deadline - time.monotonic())))
            if not chunk:
                # No bytes this tick; loop until deadline. A permanently
                # dead child will exit the loop on timeout, which is
                # what the caller's absence-assertion will detect.
                if self._proc is not None and self._proc.poll() is not None:
                    break
                continue
            self._buffer.extend(chunk)
        return self._buffer.decode("utf-8", errors="replace")

    def send(self, data: bytes | str) -> None:
        """Write raw bytes to the pty master (as a user's keystrokes)."""
        assert self._master_fd is not None
        if isinstance(data, str):
            data = data.encode("utf-8")
        os.write(self._master_fd, data)

    def read_all(self, timeout: float = 3.0) -> str:
        """Drain remaining output until EOF or ``timeout`` seconds elapse."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            chunk = self._read_some(min(0.3, max(0.05, deadline - time.monotonic())))
            if chunk:
                self._buffer.extend(chunk)
                continue
            # No bytes — if child exited, we're done draining.
            if self._proc is not None and self._proc.poll() is not None:
                # Give one last tiny window for any trailing output.
                final = self._read_some(0.1)
                if final:
                    self._buffer.extend(final)
                break
        return self._buffer.decode("utf-8", errors="replace")

    def close_master(self) -> None:
        """Close the master fd — signals EOF to the child's stdin."""
        if self._master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._master_fd)
            self._master_fd = None

    def wait(self, timeout: float = 10.0) -> int | None:
        """Wait for child exit; return returncode or None on timeout."""
        if self._proc is None:
            return None
        try:
            return self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        # Always attempt graceful close first, then kill if the child
        # didn't take the hint. The process group start_new_session=True
        # lets us SIGKILL the whole group if uv spawned grandchildren.
        self.close_master()
        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        os.killpg(self._proc.pid, signal.SIGKILL)
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self._proc.wait(timeout=2.0)


def _skip_if_startup_blocked(output: str) -> None:
    """Skip the test if aura failed to reach the REPL under pty.

    Mirrors :func:`_has_startup_block` but operates on the combined
    pty-captured stream (stdout + stderr are merged by the pty). Every
    pty test calls this immediately after its first ``expect`` so a
    bare dev env without provider SDKs gets a clear SKIP, never a
    misleading failure.
    """
    if (
        "MissingProviderDependencyError" in output
        or "MissingCredentialError" in output
    ):
        pytest.skip(
            "aura could not reach the REPL under pty "
            "(missing provider SDK or credential); pt render path "
            "cannot be exercised end-to-end here"
        )


def _requires_pty_and_aura(aura_binary: Sequence[str]) -> None:
    """Common gate: skip when pty is unavailable or aura isn't runnable."""
    if not _pty_supported():
        pytest.skip(_PTY_SKIP_REASON_NO_PTY)
    # ``aura_binary`` fixture already skipped if unusable, but keep this
    # defensive — the pty tests assert-sensitive enough that a silent
    # mis-wire would be hard to diagnose.
    assert aura_binary, "aura_binary fixture returned empty invocation"


# ---------------------------------------------------------------------------


def test_pty_repl_boots_and_quits_on_exit(
    aura_binary: Sequence[str],
) -> None:
    """Under a real pty, aura renders the welcome, prompts, and exits on /exit."""
    _requires_pty_and_aura(aura_binary)

    with _PtyAura(aura_binary, timeout_boot=10.0) as ptya:
        boot = ptya.expect("aura>", timeout=10.0)
        _skip_if_startup_blocked(boot)
        assert "aura>" in boot, (
            f"did not see 'aura>' prompt within 10s; captured={boot!r}"
        )
        assert "Aura" in boot, (
            f"welcome banner missing 'Aura' marker; captured={boot!r}"
        )

        ptya.send("/exit\r")
        rc = ptya.wait(timeout=10.0)
        tail = ptya.read_all(timeout=2.0)

    assert rc is not None, (
        f"aura did not exit within 10s of /exit under pty; "
        f"full output={tail!r}"
    )
    # main() returns 0 on a clean /exit; anything else is either a true
    # crash (no traceback expected — checked below) or an uncaught
    # propagated signal. Accept 0 only.
    assert rc == 0, (
        f"aura exited with rc={rc} on /exit under pty; output={tail!r}"
    )
    assert "Traceback (most recent call last)" not in tail, (
        f"pty REPL exit produced a Python traceback:\n{tail}"
    )


def test_pty_shift_tab_cycles_mode_without_scrollback_spam(
    aura_binary: Sequence[str],
) -> None:
    """Shift+Tab must cycle modes silently — no scrollback spam lines.

    The silent-feedback contract says the bottom_toolbar is the ONLY
    approved mode indicator. Any ``mode: X (press shift+tab ...)`` line
    in scrollback is the legacy spam regression (img_2.png in the
    operator report). We send three shift+tab presses, quit, then scan
    the full captured stream for the banned substrings.
    """
    _requires_pty_and_aura(aura_binary)

    with _PtyAura(aura_binary, timeout_boot=10.0) as ptya:
        boot = ptya.expect("aura>", timeout=10.0)
        _skip_if_startup_blocked(boot)
        assert "aura>" in boot, (
            f"did not see 'aura>' prompt before shift+tab test; got={boot!r}"
        )

        # Three rapid shift+tab presses → cycles default → accept_edits
        # → plan → default. Rapid enough to stress the "don't print a
        # line per press" invariant.
        for _ in range(3):
            ptya.send(_SHIFT_TAB)
        # Tiny settle window so any (forbidden) spam lines have time to
        # flush before we start the exit sequence.
        time.sleep(0.3)

        ptya.send("/exit\r")
        ptya.wait(timeout=10.0)
        captured = ptya.read_all(timeout=2.0)

    for banned in _LEGACY_MODE_SPAM:
        assert banned not in captured, (
            f"shift+tab produced legacy scrollback spam {banned!r}; "
            f"bottom_toolbar is the only approved mode indicator.\n"
            f"full capture:\n{captured}"
        )


def test_pty_eof_exits_cleanly(aura_binary: Sequence[str]) -> None:
    """Closing the pty master (EOF) must exit the REPL within 10s, no traceback."""
    _requires_pty_and_aura(aura_binary)

    with _PtyAura(aura_binary, timeout_boot=10.0) as ptya:
        boot = ptya.expect("aura>", timeout=10.0)
        _skip_if_startup_blocked(boot)
        assert "aura>" in boot, (
            f"did not see 'aura>' prompt before EOF test; got={boot!r}"
        )

        # Close master → child sees EOF on stdin. pt's
        # ``prompt_async`` raises EOFError, which the REPL catches and
        # returns from its loop.
        ptya.close_master()
        rc = ptya.wait(timeout=10.0)
        tail = ptya.read_all(timeout=1.0)

    assert rc is not None, (
        f"aura did not exit within 10s of EOF under pty; output={tail!r}"
    )
    # rc=120 is Python's standard exit code when stdout flushing fails during
    # shutdown — reproducible with bare ``python3 -c 'input()'`` under the
    # same pty-close sequence. It is NOT an aura bug; the REPL's EOFError
    # handler returns cleanly and main() returns 0, but Python's interpreter
    # shutdown then hits a broken stdout (master fd closed by test) and
    # self-reports 120. Accept both as "exited cleanly".
    assert rc in (0, 120), (
        f"aura exited with unexpected rc={rc} on EOF under pty; output={tail!r}"
    )
    assert "Traceback (most recent call last)" not in tail, (
        f"pty REPL EOF exit produced a Python traceback:\n{tail}"
    )


def test_pty_welcome_banner_single_cyan_panel(
    aura_binary: Sequence[str],
) -> None:
    """Welcome banner's FINAL settled state is exactly one rich Panel.

    The v0.8.0 operator feedback requested a single compact Panel
    (cyan, ``expand=False``). Post-U2 the banner's leading glyph
    animates through a spinner frame set before settling on ``✱``,
    so the raw pty capture contains many intermediate Panel redraws
    (each Live update re-emits a full Panel). That's expected — it
    replaces the previous frame via cursor-back-and-erase sequences.

    What matters is:
      1. The banner ends in a single Panel (measured by settle-glyph +
         cwd line + tip line all appearing AFTER the last animation
         frame) — a second independent Panel would show up AFTER the
         settle frame.
      2. The settled leading glyph ``✱`` AND a ``v<semver>`` tag are
         present somewhere in the capture (the final state is what
         the user sees post-animation).
      3. No nested / stacked Panels inside the settled frame — the
         settled frame contains exactly one ``╭...╯`` pair.
    """
    _requires_pty_and_aura(aura_binary)

    with _PtyAura(aura_binary, timeout_boot=10.0) as ptya:
        boot = ptya.expect("aura>", timeout=10.0)
        _skip_if_startup_blocked(boot)
        assert "aura>" in boot, (
            f"did not see 'aura>' prompt before banner test; got={boot!r}"
        )
        ptya.send("/exit\r")
        ptya.wait(timeout=10.0)
        captured = ptya.read_all(timeout=2.0)

    # Brand + version visible. The banner does ``✱ Aura v<ver>`` as one
    # logical line, but pt's rendering can wrap — so we assert the two
    # substrings independently, both present.
    assert "✱ Aura" in captured, (
        f"welcome missing '✱ Aura' settle glyph; capture={captured!r}"
    )
    assert re.search(r"v\d+\.\d+\.\d+", captured), (
        f"welcome missing 'v<semver>' marker near Aura; capture={captured!r}"
    )

    # Locate the SETTLED frame (the last one with ``✱ Aura``); everything
    # from that point onward should contain at most one ``╭...╯`` pair
    # (the settled Panel) followed by the pt prompt. A nested or extra
    # independent Panel AFTER settle would double the corner count.
    settle_idx = captured.rfind("✱ Aura")
    # Back up to the Panel's top-left corner preceding the settle line.
    panel_top = captured.rfind("╭", 0, settle_idx)
    assert panel_top != -1, (
        "no '╭' before settled ✱ banner — Panel shape broken; "
        f"capture={captured!r}"
    )
    tail = captured[panel_top:]
    # From the settled Panel opening onward: exactly ONE opening + ONE
    # closing corner. Additional Panels AFTER the settle would fire here.
    assert tail.count("╭") == 1, (
        f"unexpected extra '╭' after settled banner; tail={tail!r}"
    )
    assert tail.count("╯") == 1, (
        f"unexpected extra '╯' after settled banner; tail={tail!r}"
    )
