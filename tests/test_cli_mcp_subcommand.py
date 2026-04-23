"""Subprocess tests for ``aura mcp {add,list,remove}``.

These tests spawn the real ``aura`` binary (via ``uv run`` when
available, falling back to ``python -m aura.cli.__main__``). Each test
sets ``HOME`` to a ``tmp_path`` so the store writes to an isolated
location and the developer's real ``~/.aura/mcp_servers.json`` is never
touched.

The suite covers:
- empty-store list shows the "no servers" message
- add → list round-trip
- duplicate add is rejected with exit 1 + actionable stderr
- remove removes and list shows empty again
- remove of a non-existent name is exit 1
- ``aura --version`` still works (argparse subparser wiring preserved
  the existing global-flag path)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _aura_invocation() -> list[str] | None:
    """Return the best available command vector to run ``aura``.

    Mirrors ``tests/test_cli_smoke.py`` — ``uv run`` first, module fallback
    second. Returns None only when neither works.
    """
    if shutil.which("uv") is not None:
        return ["uv", "run", "aura"]
    try:
        import aura.cli.__main__  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    return [sys.executable, "-m", "aura.cli.__main__"]


@pytest.fixture(scope="module")
def aura_binary() -> Sequence[str]:
    invocation = _aura_invocation()
    if invocation is None:
        pytest.skip(
            "neither `uv` nor `python -m aura.cli.__main__` is usable — "
            "cannot run mcp subcommand tests"
        )
    return invocation


def _run(
    aura_binary: Sequence[str],
    extra_args: Sequence[str],
    *,
    home: Path,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    """Spawn ``aura <extra_args>`` with ``HOME=home``.

    ``HOME`` scoping keeps every subprocess test hermetic: the store reads
    and writes ``~/.aura/mcp_servers.json``, which ``Path.home()`` resolves
    via ``HOME`` on POSIX (and ``USERPROFILE`` on Windows — we stamp both
    to cover each).
    """
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)  # Windows analog, harmless on POSIX
    return subprocess.run(
        [*aura_binary, *extra_args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_REPO_ROOT),
        env=env,
        check=False,
    )


def test_list_empty_store_prints_placeholder(
    aura_binary: Sequence[str], tmp_path: Path,
) -> None:
    result = _run(aura_binary, ["mcp", "list"], home=tmp_path)
    assert result.returncode == 0, (
        f"mcp list exited {result.returncode}; stderr={result.stderr!r}"
    )
    assert "(no MCP servers configured)" in result.stdout


def test_add_then_list_round_trip(
    aura_binary: Sequence[str], tmp_path: Path,
) -> None:
    add = _run(
        aura_binary,
        ["mcp", "add", "myserver", "--transport", "stdio", "--", "echo", "hello"],
        home=tmp_path,
    )
    assert add.returncode == 0, (
        f"mcp add exited {add.returncode}; stdout={add.stdout!r} stderr={add.stderr!r}"
    )
    assert "myserver" in add.stdout
    assert "Added" in add.stdout

    # Store actually written to the expected path.
    store_path = tmp_path / ".aura" / "mcp_servers.json"
    assert store_path.exists(), f"expected store at {store_path}"
    data = json.loads(store_path.read_text(encoding="utf-8"))
    assert data["servers"][0]["name"] == "myserver"
    assert data["servers"][0]["command"] == "echo"
    assert data["servers"][0]["args"] == ["hello"]

    # list surfaces the new entry.
    lst = _run(aura_binary, ["mcp", "list"], home=tmp_path)
    assert lst.returncode == 0
    assert "myserver" in lst.stdout
    assert "stdio" in lst.stdout
    assert "echo" in lst.stdout


def test_add_default_transport_is_stdio(
    aura_binary: Sequence[str], tmp_path: Path,
) -> None:
    # Omitting --transport must default to stdio (claude-code parity).
    add = _run(
        aura_binary,
        ["mcp", "add", "defaults", "--", "cmd", "arg1", "arg2"],
        home=tmp_path,
    )
    assert add.returncode == 0, (
        f"mcp add exited {add.returncode}; stderr={add.stderr!r}"
    )
    data = json.loads(
        (tmp_path / ".aura" / "mcp_servers.json").read_text(encoding="utf-8"),
    )
    assert data["servers"][0]["transport"] == "stdio"
    assert data["servers"][0]["command"] == "cmd"
    assert data["servers"][0]["args"] == ["arg1", "arg2"]


def test_add_with_env_flags(
    aura_binary: Sequence[str], tmp_path: Path,
) -> None:
    add = _run(
        aura_binary,
        [
            "mcp", "add", "envsrv",
            "--env", "API_KEY=secret",
            "--env", "DEBUG=1",
            "--", "my-server",
        ],
        home=tmp_path,
    )
    assert add.returncode == 0, f"stderr={add.stderr!r}"
    data = json.loads(
        (tmp_path / ".aura" / "mcp_servers.json").read_text(encoding="utf-8"),
    )
    assert data["servers"][0]["env"] == {"API_KEY": "secret", "DEBUG": "1"}


def test_add_duplicate_name_rejected(
    aura_binary: Sequence[str], tmp_path: Path,
) -> None:
    first = _run(
        aura_binary, ["mcp", "add", "myserver", "--", "echo", "hello"], home=tmp_path,
    )
    assert first.returncode == 0

    dup = _run(
        aura_binary, ["mcp", "add", "myserver", "--", "echo", "again"], home=tmp_path,
    )
    assert dup.returncode == 1, (
        f"duplicate add should exit 1; got {dup.returncode}. "
        f"stdout={dup.stdout!r} stderr={dup.stderr!r}"
    )
    assert "already exists" in dup.stderr


def test_remove_existing_server(
    aura_binary: Sequence[str], tmp_path: Path,
) -> None:
    _run(aura_binary, ["mcp", "add", "myserver", "--", "echo", "hi"], home=tmp_path)
    rm = _run(aura_binary, ["mcp", "remove", "myserver"], home=tmp_path)
    assert rm.returncode == 0, f"stderr={rm.stderr!r}"
    assert "myserver" in rm.stdout

    # list now empty again.
    lst = _run(aura_binary, ["mcp", "list"], home=tmp_path)
    assert lst.returncode == 0
    assert "(no MCP servers configured)" in lst.stdout


def test_remove_nonexistent_is_user_error(
    aura_binary: Sequence[str], tmp_path: Path,
) -> None:
    rm = _run(aura_binary, ["mcp", "remove", "nonexistent"], home=tmp_path)
    assert rm.returncode == 1, (
        f"remove of unknown name should exit 1; got {rm.returncode}. "
        f"stderr={rm.stderr!r}"
    )
    assert "not found" in rm.stderr


def test_version_still_works_without_subcommand(
    aura_binary: Sequence[str], tmp_path: Path,
) -> None:
    # Backward compat — adding subparsers must not break the --version path.
    result = _run(aura_binary, ["--version"], home=tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    version_lines = [
        line for line in result.stdout.splitlines() if line.startswith("aura ")
    ]
    assert version_lines, f"no 'aura <ver>' line in stdout: {result.stdout!r}"


def test_help_lists_mcp_subcommand(
    aura_binary: Sequence[str], tmp_path: Path,
) -> None:
    result = _run(aura_binary, ["--help"], home=tmp_path)
    assert result.returncode == 0
    assert "mcp" in result.stdout


def test_mcp_help_lists_add_list_remove(
    aura_binary: Sequence[str], tmp_path: Path,
) -> None:
    result = _run(aura_binary, ["mcp", "--help"], home=tmp_path)
    assert result.returncode == 0
    # All three actions discoverable in mcp help.
    for action in ("add", "list", "remove"):
        assert action in result.stdout, (
            f"expected action {action!r} in `aura mcp --help`; got {result.stdout!r}"
        )


# --------------------------------------------------------------------- #
# --scope flag — claude-code parity (global / project layers).
# --------------------------------------------------------------------- #


def _run_in(
    aura_binary: Sequence[str],
    extra_args: Sequence[str],
    *,
    home: Path,
    cwd: Path,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    """Variant of ``_run`` that also lets the test pin ``cwd``.

    The project-layer store is resolved relative to ``cwd``, so scope
    tests need a known cwd under the fake home (otherwise the walk-up
    would escape the sandbox and read the dev's real FS).
    """
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    return subprocess.run(
        [*aura_binary, *extra_args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd),
        env=env,
        check=False,
    )


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(home, project)`` — a fake home plus a project under it.

    The project dir lives at ``<home>/project`` so the walk-up from cwd
    stops at home without crossing into the real FS.
    """
    home = tmp_path
    project = home / "project"
    project.mkdir()
    return home, project


def test_mcp_add_scope_project_writes_under_cwd(
    aura_binary: Sequence[str], sandbox: tuple[Path, Path],
) -> None:
    home, project = sandbox
    result = _run_in(
        aura_binary,
        ["mcp", "add", "projsrv", "--scope", "project", "--", "echo", "hello"],
        home=home,
        cwd=project,
    )
    assert result.returncode == 0, (
        f"exited {result.returncode}; stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # Project file created under cwd, NOT under home.
    proj_store = project / ".aura" / "mcp_servers.json"
    assert proj_store.is_file(), f"expected project store at {proj_store}"
    assert not (home / ".aura" / "mcp_servers.json").exists()

    data = json.loads(proj_store.read_text(encoding="utf-8"))
    assert data["servers"][0]["name"] == "projsrv"
    assert data["servers"][0]["command"] == "echo"


def test_mcp_add_scope_global_default_writes_under_home(
    aura_binary: Sequence[str], sandbox: tuple[Path, Path],
) -> None:
    home, project = sandbox
    # Default scope omitted — still writes to the global (home) store.
    result = _run_in(
        aura_binary,
        ["mcp", "add", "globsrv", "--", "echo", "hi"],
        home=home,
        cwd=project,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert (home / ".aura" / "mcp_servers.json").is_file()
    assert not (project / ".aura" / "mcp_servers.json").exists()


def test_mcp_list_shows_scope_column(
    aura_binary: Sequence[str], sandbox: tuple[Path, Path],
) -> None:
    home, project = sandbox
    # One entry per scope.
    _run_in(
        aura_binary,
        ["mcp", "add", "g1", "--scope", "global", "--", "gcmd"],
        home=home,
        cwd=project,
    )
    _run_in(
        aura_binary,
        ["mcp", "add", "p1", "--scope", "project", "--", "pcmd"],
        home=home,
        cwd=project,
    )
    lst = _run_in(
        aura_binary, ["mcp", "list"], home=home, cwd=project,
    )
    assert lst.returncode == 0, f"stderr={lst.stderr!r}"
    assert "SCOPE" in lst.stdout
    # Both names show with their scope tag.
    stdout = lst.stdout
    assert "g1" in stdout
    assert "p1" in stdout
    # Each name's row includes the matching scope. The table is
    # whitespace-aligned, so we just scan per-line.
    lines = stdout.splitlines()
    g_line = next(line for line in lines if line.startswith("g1"))
    p_line = next(line for line in lines if line.startswith("p1"))
    assert "global" in g_line
    assert "project" in p_line


def test_mcp_list_project_wins_on_collision(
    aura_binary: Sequence[str], sandbox: tuple[Path, Path],
) -> None:
    home, project = sandbox
    _run_in(
        aura_binary,
        ["mcp", "add", "dup", "--scope", "global", "--", "global-cmd"],
        home=home,
        cwd=project,
    )
    _run_in(
        aura_binary,
        ["mcp", "add", "dup", "--scope", "project", "--", "project-cmd"],
        home=home,
        cwd=project,
    )
    lst = _run_in(aura_binary, ["mcp", "list"], home=home, cwd=project)
    assert lst.returncode == 0
    # Only the project-scoped command is visible; the resolved row is
    # tagged ``project``.
    assert "project-cmd" in lst.stdout
    assert "global-cmd" not in lst.stdout
    dup_line = next(
        line for line in lst.stdout.splitlines() if line.startswith("dup")
    )
    assert "project" in dup_line


def test_mcp_add_duplicate_in_same_scope_rejected(
    aura_binary: Sequence[str], sandbox: tuple[Path, Path],
) -> None:
    home, project = sandbox
    first = _run_in(
        aura_binary,
        ["mcp", "add", "one", "--scope", "project", "--", "a"],
        home=home,
        cwd=project,
    )
    assert first.returncode == 0, f"stderr={first.stderr!r}"
    dup = _run_in(
        aura_binary,
        ["mcp", "add", "one", "--scope", "project", "--", "b"],
        home=home,
        cwd=project,
    )
    assert dup.returncode == 1
    assert "already exists" in dup.stderr


def test_mcp_add_same_name_across_scopes_allowed(
    aura_binary: Sequence[str], sandbox: tuple[Path, Path],
) -> None:
    # Claude-code parity: the layers are independent. Adding the same
    # name in both scopes is the primary override mechanism.
    home, project = sandbox
    g = _run_in(
        aura_binary,
        ["mcp", "add", "shared", "--scope", "global", "--", "gcmd"],
        home=home,
        cwd=project,
    )
    assert g.returncode == 0, f"stderr={g.stderr!r}"
    p = _run_in(
        aura_binary,
        ["mcp", "add", "shared", "--scope", "project", "--", "pcmd"],
        home=home,
        cwd=project,
    )
    assert p.returncode == 0, (
        f"cross-scope add should succeed; got {p.returncode}. "
        f"stderr={p.stderr!r}"
    )


def test_mcp_remove_auto_targets_project_on_collision(
    aura_binary: Sequence[str], sandbox: tuple[Path, Path],
) -> None:
    home, project = sandbox
    _run_in(
        aura_binary,
        ["mcp", "add", "shared", "--scope", "global", "--", "g"],
        home=home,
        cwd=project,
    )
    _run_in(
        aura_binary,
        ["mcp", "add", "shared", "--scope", "project", "--", "p"],
        home=home,
        cwd=project,
    )
    # No --scope → auto → remove the project entry (the one the user sees).
    rm = _run_in(
        aura_binary, ["mcp", "remove", "shared"], home=home, cwd=project,
    )
    assert rm.returncode == 0, f"stderr={rm.stderr!r}"
    assert "project" in rm.stdout

    # Global entry survives.
    g_store = home / ".aura" / "mcp_servers.json"
    data = json.loads(g_store.read_text(encoding="utf-8"))
    assert [s["name"] for s in data["servers"]] == ["shared"]
    # Project entry gone (file empty).
    p_store = project / ".aura" / "mcp_servers.json"
    p_data = json.loads(p_store.read_text(encoding="utf-8"))
    assert p_data["servers"] == []


def test_mcp_remove_explicit_scope_targets_that_layer(
    aura_binary: Sequence[str], sandbox: tuple[Path, Path],
) -> None:
    home, project = sandbox
    _run_in(
        aura_binary,
        ["mcp", "add", "shared", "--scope", "global", "--", "g"],
        home=home,
        cwd=project,
    )
    _run_in(
        aura_binary,
        ["mcp", "add", "shared", "--scope", "project", "--", "p"],
        home=home,
        cwd=project,
    )
    # Explicit --scope global removes the global entry only (the
    # shadowed one), leaving the project entry untouched.
    rm = _run_in(
        aura_binary,
        ["mcp", "remove", "shared", "--scope", "global"],
        home=home,
        cwd=project,
    )
    assert rm.returncode == 0, f"stderr={rm.stderr!r}"
    assert "global" in rm.stdout

    p_data = json.loads(
        (project / ".aura" / "mcp_servers.json").read_text(encoding="utf-8"),
    )
    assert [s["name"] for s in p_data["servers"]] == ["shared"]
    g_data = json.loads(
        (home / ".aura" / "mcp_servers.json").read_text(encoding="utf-8"),
    )
    assert g_data["servers"] == []


def test_mcp_remove_unknown_in_explicit_scope_is_user_error(
    aura_binary: Sequence[str], sandbox: tuple[Path, Path],
) -> None:
    home, project = sandbox
    _run_in(
        aura_binary,
        ["mcp", "add", "proj-only", "--scope", "project", "--", "p"],
        home=home,
        cwd=project,
    )
    # Ask to remove from global (where the entry doesn't exist) → exit 1.
    rm = _run_in(
        aura_binary,
        ["mcp", "remove", "proj-only", "--scope", "global"],
        home=home,
        cwd=project,
    )
    assert rm.returncode == 1, (
        f"expected exit 1; got {rm.returncode}. stderr={rm.stderr!r}"
    )
    assert "not found" in rm.stderr


def test_mcp_help_mentions_scope_flag(
    aura_binary: Sequence[str], tmp_path: Path,
) -> None:
    result = _run(aura_binary, ["mcp", "add", "--help"], home=tmp_path)
    assert result.returncode == 0
    assert "--scope" in result.stdout
    for choice in ("global", "project"):
        assert choice in result.stdout, (
            f"expected scope choice {choice!r} in `aura mcp add --help`"
        )
