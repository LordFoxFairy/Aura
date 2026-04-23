"""Unit tests for :mod:`aura.config.mcp_store`.

The store is the single source of truth for user-editable MCP server
entries. These tests lock its round-trip contract against
:class:`aura.config.schema.MCPServerConfig` so subtle field renames or
validator changes surface here before they break the CLI subcommands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aura.config import mcp_store
from aura.config.schema import MCPServerConfig


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` to ``tmp_path`` for the duration of a test.

    Without this, ``mcp_store.get_path()`` would point at the real user
    ``~/.aura/mcp_servers.json`` and the test would either pollute the
    developer's environment or depend on it. Scoping to ``tmp_path`` keeps
    every test hermetic.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    # ``Path.home()`` honours ``HOME`` on POSIX via ``os.path.expanduser``,
    # but on some stdlib versions it goes through ``pwd.getpwuid`` — patch
    # directly to be safe across platforms.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def test_get_path_under_home(isolated_home: Path) -> None:
    assert mcp_store.get_path() == isolated_home / ".aura" / "mcp_servers.json"


def test_load_missing_file_returns_empty(isolated_home: Path) -> None:
    # First-run contract: no file → no error, empty list.
    assert mcp_store.load() == []
    # Parent dir MUST NOT be auto-created on read — only save() does that.
    assert not (isolated_home / ".aura").exists()


def test_save_empty_then_load(isolated_home: Path) -> None:
    mcp_store.save([])
    assert mcp_store.load() == []
    # Parent dir created on save.
    assert (isolated_home / ".aura").exists()
    assert (isolated_home / ".aura" / "mcp_servers.json").exists()


def test_round_trip_stdio(isolated_home: Path) -> None:
    original = [
        MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            env={},
        ),
    ]
    mcp_store.save(original)
    loaded = mcp_store.load()
    assert loaded == original
    assert loaded[0].name == "filesystem"
    assert loaded[0].command == "npx"
    assert loaded[0].args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]


def test_round_trip_sse(isolated_home: Path) -> None:
    original = [
        MCPServerConfig(
            name="sentry",
            transport="sse",
            url="https://mcp.sentry.dev/mcp",
            headers={"Authorization": "Bearer xxx"},
        ),
    ]
    mcp_store.save(original)
    loaded = mcp_store.load()
    assert loaded == original
    assert loaded[0].transport == "sse"
    assert loaded[0].url == "https://mcp.sentry.dev/mcp"


def test_round_trip_streamable_http(isolated_home: Path) -> None:
    original = [
        MCPServerConfig(
            name="corridor",
            transport="streamable_http",
            url="https://app.corridor.dev/api/mcp",
        ),
    ]
    mcp_store.save(original)
    loaded = mcp_store.load()
    assert loaded == original
    assert loaded[0].transport == "streamable_http"


def test_round_trip_non_empty_env(isolated_home: Path) -> None:
    original = [
        MCPServerConfig(
            name="my-server",
            transport="stdio",
            command="my-mcp-server",
            args=["--flag", "value"],
            env={"API_KEY": "secret", "DEBUG": "1"},
        ),
    ]
    mcp_store.save(original)
    loaded = mcp_store.load()
    assert loaded == original
    assert loaded[0].env == {"API_KEY": "secret", "DEBUG": "1"}


def test_multiple_entries_preserve_order(isolated_home: Path) -> None:
    original = [
        MCPServerConfig(name="a", transport="stdio", command="cmd-a"),
        MCPServerConfig(name="b", transport="stdio", command="cmd-b"),
        MCPServerConfig(name="c", transport="sse", url="https://c.example/mcp"),
    ]
    mcp_store.save(original)
    loaded = mcp_store.load()
    assert [s.name for s in loaded] == ["a", "b", "c"]
    assert loaded == original


def test_load_rejects_non_object_top_level(isolated_home: Path) -> None:
    path = mcp_store.get_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(ValueError, match="expected object at top level"):
        mcp_store.load()


def test_load_rejects_invalid_json(isolated_home: Path) -> None:
    path = mcp_store.get_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json")
    with pytest.raises(ValueError, match="invalid JSON"):
        mcp_store.load()


def test_load_rejects_non_list_servers(isolated_home: Path) -> None:
    path = mcp_store.get_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"servers": "oops"}))
    with pytest.raises(ValueError, match="'servers' must be a list"):
        mcp_store.load()


def test_load_surfaces_validation_errors(isolated_home: Path) -> None:
    # stdio transport without command is a validator failure; the store
    # must surface it rather than silently dropping the entry.
    path = mcp_store.get_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "servers": [{"name": "bad", "transport": "stdio"}],
    }))
    with pytest.raises(ValueError, match="invalid server entry"):
        mcp_store.load()


def test_save_creates_parent_dir(isolated_home: Path) -> None:
    # Fresh HOME — no ~/.aura yet. save() must create it.
    assert not (isolated_home / ".aura").exists()
    mcp_store.save([MCPServerConfig(name="x", transport="stdio", command="echo")])
    assert (isolated_home / ".aura").is_dir()
    assert (isolated_home / ".aura" / "mcp_servers.json").is_file()


def test_save_overwrites_existing_file(isolated_home: Path) -> None:
    mcp_store.save([MCPServerConfig(name="first", transport="stdio", command="a")])
    mcp_store.save([MCPServerConfig(name="second", transport="stdio", command="b")])
    loaded = mcp_store.load()
    assert len(loaded) == 1
    assert loaded[0].name == "second"


def test_saved_json_shape_is_hand_editable(isolated_home: Path) -> None:
    # The file is designed to be hand-edited, so pretty-print with a
    # top-level "servers" key (matches the spec in the design doc).
    mcp_store.save([
        MCPServerConfig(name="x", transport="stdio", command="cmd"),
    ])
    raw = mcp_store.get_path().read_text(encoding="utf-8")
    data = json.loads(raw)
    assert set(data.keys()) == {"servers"}
    assert isinstance(data["servers"], list)
    assert data["servers"][0]["name"] == "x"
    # Pretty-printed — presence of a newline between fields is enough.
    assert "\n" in raw
