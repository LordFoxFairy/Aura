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

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

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
