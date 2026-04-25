"""Tests for F-06-002: ``${VAR}`` / ``${VAR:-default}`` expansion in MCP
stdio config.

The expander itself lives in :mod:`aura.core.mcp.adapter`; the load-time
hook lives in :mod:`aura.config.mcp_store`. We exercise both layers:

- :func:`_expand_env_vars` directly for grammar correctness.
- :func:`mcp_store.load`-equivalent paths via on-disk JSON to confirm the
  expansion fires across ``command`` / ``args`` / ``env`` / ``headers``
  / ``url`` and the missing-var journal warning is emitted exactly once
  per file regardless of how many references are missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aura.config import mcp_store
from aura.core.mcp.adapter import _expand_env_vars

# ---------------------------------------------------------------------------
# Direct expander grammar
# ---------------------------------------------------------------------------


def test_expand_passthrough_no_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """A string with no ``${...}`` returns identically — fast path."""
    monkeypatch.setenv("FOO", "bar")
    assert _expand_env_vars("plain") == "plain"
    assert _expand_env_vars("") == ""


def test_expand_simple_var_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYTOK", "secret")
    assert _expand_env_vars("Bearer ${MYTOK}") == "Bearer secret"


def test_expand_simple_var_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing ``${VAR}`` (no default) expands to empty AND lands in
    the missing log so the caller can warn once per file."""
    monkeypatch.delenv("ABSENT", raising=False)
    log: list[str] = []
    out = _expand_env_vars("x=${ABSENT}", _missing_log=log)
    assert out == "x="
    assert log == ["ABSENT"]


def test_expand_default_used_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPTIONAL", raising=False)
    assert _expand_env_vars("${OPTIONAL:-fallback}") == "fallback"


def test_expand_default_skipped_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRESENT", "real")
    assert _expand_env_vars("${PRESENT:-fallback}") == "real"


def test_expand_default_used_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty string in env should fall back to the default — matches
    claude-code and shell semantics for ``${VAR:-default}``."""
    monkeypatch.setenv("EMPTY", "")
    assert _expand_env_vars("${EMPTY:-fb}") == "fb"


def test_expand_multiple_in_one_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("A", "1")
    monkeypatch.setenv("B", "2")
    assert _expand_env_vars("${A}+${B}=3") == "1+2=3"


def test_expand_no_recursion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Output is NOT re-expanded — security-critical so a value
    containing ``${...}`` doesn't read another env var on use."""
    monkeypatch.setenv("OUTER", "${INNER}")
    monkeypatch.setenv("INNER", "secret")
    assert _expand_env_vars("${OUTER}") == "${INNER}"


def test_expand_default_with_dash_in_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``:-`` is the separator; later ``-`` in the default text is
    preserved verbatim."""
    monkeypatch.delenv("MISSING", raising=False)
    assert _expand_env_vars("${MISSING:-multi-word-default}") == "multi-word-default"


# ---------------------------------------------------------------------------
# mcp_store integration
# ---------------------------------------------------------------------------


def _write_global_store(home: Path, payload: dict[str, Any]) -> Path:
    p = home / ".aura" / "mcp_servers.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@pytest.fixture
def isolated_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point ``$HOME`` and ``cwd`` at a clean tmp tree so the loader's
    walk-up doesn't read the developer's real ``~/.aura``."""
    home = tmp_path / "home"
    cwd = tmp_path / "work"
    home.mkdir()
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    return home


def test_load_expands_command_args_env(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stdio entry with ``${VAR}`` references in command / args /
    env gets each token expanded at load time."""
    monkeypatch.setenv("MYCMD", "/usr/bin/server")
    monkeypatch.setenv("MYARG", "--token=abc")
    monkeypatch.setenv("MYTOK", "deadbeef")
    _write_global_store(isolated_home, {
        "servers": [{
            "name": "s1",
            "transport": "stdio",
            "command": "${MYCMD}",
            "args": ["--flag", "${MYARG}"],
            "env": {"TOKEN": "${MYTOK}"},
        }],
    })

    servers = mcp_store.load()
    assert len(servers) == 1
    s = servers[0]
    assert s.command == "/usr/bin/server"
    assert s.args == ["--flag", "--token=abc"]
    assert s.env == {"TOKEN": "deadbeef"}


def test_load_uses_default_when_var_missing(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NOPE", raising=False)
    _write_global_store(isolated_home, {
        "servers": [{
            "name": "s1",
            "transport": "stdio",
            "command": "/bin/sh",
            "args": ["-c", "echo ${NOPE:-fallback}"],
        }],
    })

    servers = mcp_store.load()
    # The literal ``${NOPE:-fallback}`` was rewritten to ``fallback``
    # before pydantic validation.
    assert servers[0].args == ["-c", "echo fallback"]


def test_load_missing_var_warns_once_per_file(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A file referencing missing vars emits one
    ``mcp_env_var_missing`` event per file, listing all unique missing
    names. Missing references appear in non-required fields so the
    pydantic validator still accepts the entry; the warning is
    advisory."""
    from aura.core.persistence import journal

    journal_path = tmp_path / "journal.jsonl"
    journal.configure(journal_path)
    monkeypatch.delenv("MISSING_A", raising=False)
    monkeypatch.delenv("MISSING_B", raising=False)
    _write_global_store(isolated_home, {
        "servers": [{
            "name": "s1",
            "transport": "stdio",
            "command": "/bin/sh",
            "args": ["${MISSING_A}", "${MISSING_B}", "${MISSING_A}"],
            "env": {"PARAM": "${MISSING_A}"},
        }],
    })

    mcp_store.load()
    journal.reset()

    assert journal_path.exists()
    events = [
        json.loads(line)
        for line in journal_path.read_text().splitlines()
        if line.strip()
    ]
    missing_events = [
        e for e in events if e.get("event") == "mcp_env_var_missing"
    ]
    # Exactly one event for the file.
    assert len(missing_events) == 1
    # Sorted, deduplicated list of names.
    assert missing_events[0]["missing"] == ["MISSING_A", "MISSING_B"]


def test_load_expands_url_and_headers_for_http_transports(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSE / streamable_http entries also benefit from expansion in
    URL query strings and Authorization headers."""
    monkeypatch.setenv("API_HOST", "https://api.example.com")
    monkeypatch.setenv("BEARER", "token-xyz")
    _write_global_store(isolated_home, {
        "servers": [{
            "name": "s1",
            "transport": "sse",
            "url": "${API_HOST}/mcp",
            "headers": {"Authorization": "Bearer ${BEARER}"},
        }],
    })

    servers = mcp_store.load()
    assert servers[0].url == "https://api.example.com/mcp"
    assert servers[0].headers == {"Authorization": "Bearer token-xyz"}


def test_load_no_template_unchanged(
    isolated_home: Path,
) -> None:
    """An entry with no ``${...}`` round-trips identically — fast path
    must be transparent."""
    _write_global_store(isolated_home, {
        "servers": [{
            "name": "plain",
            "transport": "stdio",
            "command": "/usr/bin/foo",
            "args": ["--bar"],
        }],
    })
    servers = mcp_store.load()
    assert servers[0].command == "/usr/bin/foo"
    assert servers[0].args == ["--bar"]
