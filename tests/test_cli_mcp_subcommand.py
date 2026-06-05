"""Subprocess tests for ``aura mcp {add,list,remove}`` against an isolated HOME."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from cli.mcp_cli import handle_mcp

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _aura_invocation() -> list[str] | None:
    """Return the best available command vector to run ``aura``.

    Mirrors ``tests/test_cli_smoke.py`` — ``uv run`` first, module fallback
    second. Returns None only when neither works.
    """
    if shutil.which("uv") is not None:
        return ["uv", "run", "aura"]
    try:
        import cli.__main__  # noqa: F401  # import is the assertion / fixture side-effect
    except Exception:  # noqa: BLE001  # blanket catch acceptable here
        return None
    return [sys.executable, "-m", "cli.__main__"]


@pytest.fixture(scope="module")
def aura_binary() -> Sequence[str]:
    invocation = _aura_invocation()
    if invocation is None:
        pytest.skip(
            "neither `uv` nor `python -m cli.__main__` is usable — "
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
    # Omitting --transport must default to stdio.
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
    # Scopes are independent; same name in both is the override mechanism.
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


# ---------------------------------------------------------------------------
# In-process handler tests.
#
# The subprocess tests above prove the wired CLI; these call ``handle_mcp``
# directly so every error branch (arg-parse failures, scope/env validation,
# transport mismatches, store-read failures) is exercised cheaply and the
# printed text + exit code are asserted together. ``Path.home`` is pinned to
# ``home`` and cwd to ``project`` (a child of home) so global resolves under
# home and project under cwd — never the developer's real ``~/.aura``.
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Pin ``Path.home`` and cwd into a sandbox so the store stays hermetic."""
    home = tmp_path
    project = home / "project"
    project.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(project)
    return home, project


def _add_ns(
    name: str,
    *,
    transport: str = "stdio",
    scope: str | None = None,
    env: list[str] | None = None,
    command_args: list[str] | None = None,
) -> argparse.Namespace:
    """Build the ``add`` Namespace the CLI parser would hand to ``handle_mcp``."""
    return argparse.Namespace(
        mcp_action="add",
        name=name,
        transport=transport,
        scope=scope,
        env=env,
        command_args=command_args if command_args is not None else [],
    )


def _remove_ns(name: str, *, scope: str | None = None) -> argparse.Namespace:
    """Build the ``remove`` Namespace the CLI parser would hand to ``handle_mcp``."""
    return argparse.Namespace(mcp_action="remove", name=name, scope=scope)


def _store_servers(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    servers = data["servers"]
    assert isinstance(servers, list)
    return servers


# --- dispatch -------------------------------------------------------------


def test_handle_mcp_missing_action_is_user_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Calling ``aura mcp`` with no sub-action must fail loudly with usage, not silently no-op."""
    rc = handle_mcp(argparse.Namespace(mcp_action=None))
    captured = capsys.readouterr()
    assert rc == 1
    assert "missing mcp action" in captured.err
    assert "{add|list|remove}" in captured.err
    assert captured.out == ""


def test_handle_mcp_unknown_action_is_user_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unrecognised mcp_action must hit the dispatch fallthrough, not crash or misdispatch."""
    rc = handle_mcp(argparse.Namespace(mcp_action="frobnicate"))
    assert rc == 1
    assert "missing mcp action" in capsys.readouterr().err


# --- _parse_env_pairs (via add) -------------------------------------------


def test_add_env_pair_without_equals_fails_loudly(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """A malformed --env pair lacking '=' must fail loudly, not silently drop the variable."""
    home, _ = mcp_env
    rc = handle_mcp(_add_ns("s", env=["NOEQUALS"], command_args=["cmd"]))
    captured = capsys.readouterr()
    assert rc == 1
    assert "KEY=VALUE" in captured.err
    assert "no '=' found" in captured.err
    # Nothing was persisted — a rejected add must not write a partial store.
    assert not (home / ".aura" / "mcp_servers.json").exists()


def test_add_env_pair_with_empty_key_fails_loudly(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """An --env pair with an empty key (=VALUE) is unusable and must be rejected, not stored."""
    home, _ = mcp_env
    rc = handle_mcp(_add_ns("s", env=["=orphanvalue"], command_args=["cmd"]))
    captured = capsys.readouterr()
    assert rc == 1
    assert "empty key" in captured.err
    assert not (home / ".aura" / "mcp_servers.json").exists()


def test_add_env_value_may_contain_equals(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """The split must be on the FIRST '=' only, so base64/URL values keep their inner '='."""
    home, _ = mcp_env
    rc = handle_mcp(
        _add_ns("s", env=["TOKEN=a=b=c"], command_args=["cmd"]),
    )
    assert rc == 0
    capsys.readouterr()
    servers = _store_servers(home / ".aura" / "mcp_servers.json")
    assert servers[0]["env"] == {"TOKEN": "a=b=c"}


# --- _resolve_write_scope (via add) ---------------------------------------


def test_add_unknown_scope_fails_loudly(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """A scope outside {global,project} (e.g. argv tampering) must be rejected before any write."""
    home, project = mcp_env
    rc = handle_mcp(_add_ns("s", scope="local", command_args=["cmd"]))
    captured = capsys.readouterr()
    assert rc == 1
    assert "unknown scope: 'local'" in captured.err
    assert not (home / ".aura" / "mcp_servers.json").exists()
    assert not (project / ".aura" / "mcp_servers.json").exists()


def test_add_explicit_global_scope_writes_under_home(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """Explicit --scope global must resolve the home layer, mirroring the omitted-flag default."""
    home, project = mcp_env
    rc = handle_mcp(_add_ns("g", scope="global", command_args=["gcmd"]))
    assert rc == 0
    out = capsys.readouterr().out
    assert "(global)" in out
    assert (home / ".aura" / "mcp_servers.json").is_file()
    assert not (project / ".aura" / "mcp_servers.json").exists()


# --- _cmd_add: stdio branch ----------------------------------------------


def test_add_stdio_without_command_tokens_fails_loudly(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """stdio with no command after '--' is unrunnable and must be rejected with usage."""
    home, _ = mcp_env
    rc = handle_mcp(_add_ns("s", transport="stdio", command_args=[]))
    captured = capsys.readouterr()
    assert rc == 1
    assert "stdio transport requires a command" in captured.err
    assert not (home / ".aura" / "mcp_servers.json").exists()


def test_add_stdio_persists_command_and_args(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """stdio add must split tokens into command + args and echo the joined invocation back."""
    home, _ = mcp_env
    rc = handle_mcp(
        _add_ns("s", transport="stdio", command_args=["npx", "-y", "pkg"]),
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Added stdio MCP server 's'" in out
    assert "npx -y pkg" in out
    assert f"File modified: {home / '.aura' / 'mcp_servers.json'}" in out
    servers = _store_servers(home / ".aura" / "mcp_servers.json")
    assert servers[0]["command"] == "npx"
    assert servers[0]["args"] == ["-y", "pkg"]


# --- _cmd_add: sse / streamable_http branch ------------------------------


def test_add_http_transport_without_url_fails_loudly(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """A URL-based transport with no token after '--' has no endpoint and must be rejected."""
    home, _ = mcp_env
    rc = handle_mcp(
        _add_ns("s", transport="sse", command_args=[]),
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "sse transport requires a URL" in captured.err
    assert not (home / ".aura" / "mcp_servers.json").exists()


def test_add_http_transport_with_extra_tokens_fails_loudly(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """A URL transport takes one token; surplus tokens are ambiguous and must be rejected."""
    home, _ = mcp_env
    rc = handle_mcp(
        _add_ns(
            "s",
            transport="streamable_http",
            command_args=["https://a.example", "https://b.example"],
        ),
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "exactly one URL" in captured.err
    assert "got 2 tokens" in captured.err
    assert not (home / ".aura" / "mcp_servers.json").exists()


def test_add_http_transport_persists_url(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """A well-formed sse add must store and echo the single URL, with no command field set."""
    home, _ = mcp_env
    rc = handle_mcp(
        _add_ns("remote", transport="sse", command_args=["https://mcp.example/mcp"]),
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Added sse MCP server 'remote'" in out
    assert "https://mcp.example/mcp" in out
    servers = _store_servers(home / ".aura" / "mcp_servers.json")
    assert servers[0]["url"] == "https://mcp.example/mcp"
    assert servers[0]["command"] is None


# --- _cmd_add: idempotency / duplicate ------------------------------------


def test_add_same_server_twice_in_same_scope_rejected(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """Re-adding a name in the same layer must fail loudly — add is not silently idempotent."""
    home, _ = mcp_env
    assert handle_mcp(_add_ns("dup", command_args=["a"])) == 0
    capsys.readouterr()
    rc = handle_mcp(_add_ns("dup", command_args=["b"]))
    captured = capsys.readouterr()
    assert rc == 1
    assert "already exists" in captured.err
    assert "aura mcp remove dup" in captured.err
    # The original entry is untouched — the second add did not overwrite it.
    servers = _store_servers(home / ".aura" / "mcp_servers.json")
    assert [s["name"] for s in servers] == ["dup"]
    assert servers[0]["command"] == "a"


def test_add_same_name_across_scopes_allowed(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """The per-layer duplicate check must allow the same name in both scopes (the override path)."""
    home, project = mcp_env
    assert handle_mcp(_add_ns("shared", scope="global", command_args=["g"])) == 0
    assert handle_mcp(_add_ns("shared", scope="project", command_args=["p"])) == 0
    capsys.readouterr()
    assert _store_servers(home / ".aura" / "mcp_servers.json")[0]["command"] == "g"
    assert _store_servers(project / ".aura" / "mcp_servers.json")[0]["command"] == "p"


# --- _cmd_list ------------------------------------------------------------


def test_list_empty_store_prints_placeholder_in_process(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty store must surface a human placeholder and exit 0, not an empty/blank table."""
    rc = handle_mcp(argparse.Namespace(mcp_action="list"))
    captured = capsys.readouterr()
    assert rc == 0
    assert "(no MCP servers configured)" in captured.out


def test_list_tags_each_row_with_its_scope(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """list must re-derive each row's origin layer so users see global vs project per server."""
    handle_mcp(_add_ns("g1", scope="global", command_args=["gcmd"]))
    handle_mcp(_add_ns("p1", scope="project", command_args=["pcmd"]))
    capsys.readouterr()
    rc = handle_mcp(argparse.Namespace(mcp_action="list"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "NAME" in out and "SCOPE" in out and "TRANSPORT" in out
    g_line = next(line for line in out.splitlines() if line.startswith("g1"))
    p_line = next(line for line in out.splitlines() if line.startswith("p1"))
    assert "global" in g_line
    assert "project" in p_line


def test_list_renders_url_for_http_transport_row(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-stdio row's COMMAND cell must show its URL, not blank, so remotes stay legible."""
    handle_mcp(
        _add_ns("remote", transport="sse", command_args=["https://mcp.example/x"]),
    )
    capsys.readouterr()
    handle_mcp(argparse.Namespace(mcp_action="list"))
    out = capsys.readouterr().out
    remote_line = next(line for line in out.splitlines() if line.startswith("remote"))
    assert "sse" in remote_line
    assert "https://mcp.example/x" in remote_line


def test_list_surfaces_corrupt_store_as_error(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """A corrupt store must fail loudly with a diagnostic, not crash or render a half-table."""
    home, _ = mcp_env
    store = home / ".aura" / "mcp_servers.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("{ this is not json", encoding="utf-8")
    rc = handle_mcp(argparse.Namespace(mcp_action="list"))
    captured = capsys.readouterr()
    assert rc == 1
    assert "error:" in captured.err
    assert "invalid JSON" in captured.err


# --- _cmd_remove ----------------------------------------------------------


def test_remove_auto_targets_resolved_layer(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """Omitting --scope must remove from whichever layer resolves the name (project wins)."""
    home, project = mcp_env
    handle_mcp(_add_ns("shared", scope="global", command_args=["g"]))
    handle_mcp(_add_ns("shared", scope="project", command_args=["p"]))
    capsys.readouterr()
    rc = handle_mcp(_remove_ns("shared"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Removed MCP server 'shared' (project)" in out
    # Project entry gone, global survivor intact.
    assert _store_servers(project / ".aura" / "mcp_servers.json") == []
    assert [s["name"] for s in _store_servers(home / ".aura" / "mcp_servers.json")] == ["shared"]


def test_remove_auto_unknown_name_is_user_error(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """Auto-removing a name absent from every layer must fail loudly; a no-op delete is an error."""
    rc = handle_mcp(_remove_ns("ghost"))
    captured = capsys.readouterr()
    assert rc == 1
    assert "not found" in captured.err
    assert "project layers" in captured.err


def test_remove_explicit_scope_targets_only_that_layer(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """Explicit --scope global removes the shadowed global entry, leaving project intact."""
    home, project = mcp_env
    handle_mcp(_add_ns("shared", scope="global", command_args=["g"]))
    handle_mcp(_add_ns("shared", scope="project", command_args=["p"]))
    capsys.readouterr()
    rc = handle_mcp(_remove_ns("shared", scope="global"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "(global)" in out
    assert _store_servers(home / ".aura" / "mcp_servers.json") == []
    assert [s["name"] for s in _store_servers(project / ".aura" / "mcp_servers.json")] == ["shared"]


def test_remove_explicit_unknown_scope_fails_loudly(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """An invalid explicit --scope on remove must be rejected, not silently fall back to auto."""
    rc = handle_mcp(_remove_ns("anything", scope="weird"))
    captured = capsys.readouterr()
    assert rc == 1
    assert "unknown scope: 'weird'" in captured.err


def test_remove_absent_from_explicit_scope_is_user_error(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """Removing a name absent from the named layer must fail loudly, even if it exists elsewhere."""
    home, project = mcp_env
    handle_mcp(_add_ns("proj-only", scope="project", command_args=["p"]))
    capsys.readouterr()
    rc = handle_mcp(_remove_ns("proj-only", scope="global"))
    captured = capsys.readouterr()
    assert rc == 1
    assert "not found" in captured.err
    # The project entry it lives in is untouched.
    assert [s["name"] for s in _store_servers(project / ".aura" / "mcp_servers.json")] == [
        "proj-only",
    ]


def test_remove_then_remove_again_is_idempotency_guard(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """A second remove of an already-removed name must fail loudly, not silently succeed twice."""
    handle_mcp(_add_ns("once", scope="global", command_args=["c"]))
    capsys.readouterr()
    assert handle_mcp(_remove_ns("once", scope="global")) == 0
    capsys.readouterr()
    rc = handle_mcp(_remove_ns("once", scope="global"))
    assert rc == 1
    assert "not found" in capsys.readouterr().err


# --- store-read failures during add/remove --------------------------------


def _corrupt_layer(home_or_project: Path) -> Path:
    store = home_or_project / ".aura" / "mcp_servers.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("}{ not json", encoding="utf-8")
    return store


def test_add_http_empty_url_token_is_schema_crash(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty URL token for a URL transport must be caught by the schema, not stored blank."""
    home, _ = mcp_env
    rc = handle_mcp(_add_ns("s", transport="sse", command_args=[""]))
    captured = capsys.readouterr()
    assert rc == 1
    assert "error:" in captured.err
    assert "url" in captured.err.lower()
    assert not (home / ".aura" / "mcp_servers.json").exists()


def test_add_surfaces_corrupt_target_layer_as_error(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """If the target layer is unreadable, add must fail loudly on the dup check, not overwrite."""
    home, _ = mcp_env
    _corrupt_layer(home)
    rc = handle_mcp(_add_ns("s", scope="global", command_args=["cmd"]))
    captured = capsys.readouterr()
    assert rc == 1
    assert "error:" in captured.err
    assert "invalid JSON" in captured.err


def test_remove_surfaces_corrupt_target_layer_as_error(
    mcp_env: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """If the scoped layer is unreadable, remove must fail loudly rather than delete blindly."""
    home, _ = mcp_env
    _corrupt_layer(home)
    rc = handle_mcp(_remove_ns("s", scope="global"))
    captured = capsys.readouterr()
    assert rc == 1
    assert "error:" in captured.err
    assert "invalid JSON" in captured.err
