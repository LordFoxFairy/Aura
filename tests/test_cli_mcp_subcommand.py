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
