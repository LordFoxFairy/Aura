"""Unit tests for :mod:`aura.config.mcp_store`.

The store is the single source of truth for user-editable MCP server
entries. These tests lock its round-trip contract against
:class:`aura.config.schema.MCPServerConfig` so subtle field renames or
validator changes surface here before they break the CLI subcommands.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aura.config import mcp_store
from aura.config.schema import MCPServerConfig


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` and ``Path.cwd()`` under ``tmp_path``.

    Without this, ``mcp_store.get_path()`` would point at the real user
    ``~/.aura/mcp_servers.json`` and the test would either pollute the
    developer's environment or depend on it. We also chdir into a
    ``project/`` subdir under the faked home so the project-layer
    walk-up (added when we gained global/project layers) stops cleanly
    at home without tripping over the actual repo's ``.aura/`` or the
    developer's real ``~/.aura/``. Scoping to ``tmp_path`` keeps every
    test hermetic.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    # ``Path.home()`` honours ``HOME`` on POSIX via ``os.path.expanduser``,
    # but on some stdlib versions it goes through ``pwd.getpwuid`` — patch
    # directly to be safe across platforms.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    # Move cwd into a subdir of the faked home so the project-layer
    # walk-up stops at ``tmp_path`` without the walk escaping into the
    # real FS. Without this, ``load()`` would start at the actual test
    # cwd (the repo root) and walk to filesystem root — picking up any
    # stray ``.aura/mcp_servers.json`` along the way.
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    return tmp_path


@pytest.fixture
def project_dir(isolated_home: Path) -> Path:
    """Return the cwd the ``isolated_home`` fixture chdir'd into.

    Layer-merge tests need to write ``<cwd>/.aura/mcp_servers.json``
    explicitly rather than go through the global store, so they need a
    handle on the project root. Building on ``isolated_home`` keeps
    the two fixtures in sync.
    """
    # ``os.getcwd()`` honours the monkeypatch.chdir performed in the
    # parent fixture; Path.cwd() sees it too.
    return Path(os.getcwd())


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


# --------------------------------------------------------------------- #
# Layer merge (global + project) — parity with claude-code's scope model.
# --------------------------------------------------------------------- #


def test_global_path_and_project_path_are_distinct(
    isolated_home: Path, project_dir: Path,
) -> None:
    # The two scopes MUST resolve to different files; a collision would
    # mean project writes silently clobber the global file (or vice
    # versa).
    assert mcp_store.global_path() == isolated_home / ".aura" / "mcp_servers.json"
    assert mcp_store.project_path() == project_dir / ".aura" / "mcp_servers.json"
    assert mcp_store.global_path() != mcp_store.project_path()


def test_save_scope_project_writes_under_cwd(
    isolated_home: Path, project_dir: Path,
) -> None:
    mcp_store.save(
        [MCPServerConfig(name="proj", transport="stdio", command="echo")],
        scope="project",
    )
    # File lands at <cwd>/.aura/mcp_servers.json, NOT the global path.
    assert (project_dir / ".aura" / "mcp_servers.json").is_file()
    assert not (isolated_home / ".aura" / "mcp_servers.json").exists()


def test_save_scope_global_writes_under_home(
    isolated_home: Path, project_dir: Path,
) -> None:
    # ``scope="global"`` (default) keeps the pre-layer behaviour.
    mcp_store.save(
        [MCPServerConfig(name="glob", transport="stdio", command="echo")],
        scope="global",
    )
    assert (isolated_home / ".aura" / "mcp_servers.json").is_file()
    assert not (project_dir / ".aura" / "mcp_servers.json").exists()


def test_load_merges_global_and_project(
    isolated_home: Path, project_dir: Path,
) -> None:
    mcp_store.save(
        [MCPServerConfig(name="g1", transport="stdio", command="gcmd")],
        scope="global",
    )
    mcp_store.save(
        [MCPServerConfig(name="p1", transport="stdio", command="pcmd")],
        scope="project",
    )
    merged = mcp_store.load()
    names = {s.name for s in merged}
    assert names == {"g1", "p1"}


def test_load_project_overrides_global_on_name_collision(
    isolated_home: Path, project_dir: Path,
) -> None:
    # Same name in both layers: project wins.
    mcp_store.save(
        [MCPServerConfig(name="shared", transport="stdio", command="global-cmd")],
        scope="global",
    )
    mcp_store.save(
        [MCPServerConfig(name="shared", transport="stdio", command="project-cmd")],
        scope="project",
    )
    merged = mcp_store.load()
    assert len(merged) == 1
    assert merged[0].name == "shared"
    assert merged[0].command == "project-cmd"


def test_load_project_file_missing_falls_back_to_global(
    isolated_home: Path, project_dir: Path,
) -> None:
    # Graceful handling of a missing project file — claude-code parity.
    mcp_store.save(
        [MCPServerConfig(name="only-global", transport="stdio", command="g")],
        scope="global",
    )
    assert not (project_dir / ".aura").exists()
    loaded = mcp_store.load()
    assert [s.name for s in loaded] == ["only-global"]


def test_load_global_file_missing_surfaces_only_project(
    isolated_home: Path, project_dir: Path,
) -> None:
    # Inverse case: project-only config, no global file. The user
    # hasn't set up a personal global store yet but the project has.
    mcp_store.save(
        [MCPServerConfig(name="only-proj", transport="stdio", command="p")],
        scope="project",
    )
    assert not (isolated_home / ".aura" / "mcp_servers.json").exists()
    loaded = mcp_store.load()
    assert [s.name for s in loaded] == ["only-proj"]


def test_load_project_walks_ancestor_directories(
    isolated_home: Path, project_dir: Path,
) -> None:
    # Simulate an inner project nested inside an outer project that
    # also has its own ``.aura/mcp_servers.json``. Walk-up must see
    # both, and the innermost layer wins on a collision. This mirrors
    # skills-loader behaviour.
    outer_servers = [
        MCPServerConfig(name="outer", transport="stdio", command="outer-cmd"),
        MCPServerConfig(name="shared", transport="stdio", command="outer-shared"),
    ]
    outer_path = project_dir / ".aura" / "mcp_servers.json"
    outer_path.parent.mkdir(parents=True, exist_ok=True)
    outer_path.write_text(json.dumps({
        "servers": [s.model_dump(mode="json") for s in outer_servers],
    }))

    inner_dir = project_dir / "sub" / "inner"
    inner_dir.mkdir(parents=True)
    inner_servers = [
        MCPServerConfig(name="inner", transport="stdio", command="inner-cmd"),
        MCPServerConfig(name="shared", transport="stdio", command="inner-shared"),
    ]
    inner_path = inner_dir / ".aura" / "mcp_servers.json"
    inner_path.parent.mkdir(parents=True, exist_ok=True)
    inner_path.write_text(json.dumps({
        "servers": [s.model_dump(mode="json") for s in inner_servers],
    }))

    # Load from the innermost dir — walk-up picks both layers up.
    os.chdir(inner_dir)
    loaded = mcp_store.load()
    by_name = {s.name: s for s in loaded}
    assert set(by_name) == {"outer", "inner", "shared"}
    # Innermost wins the collision.
    assert by_name["shared"].command == "inner-shared"


def test_load_project_walk_stops_before_home(
    isolated_home: Path, project_dir: Path,
) -> None:
    # The walk must NOT enter $HOME itself — that layer is "global" and
    # would otherwise be double-counted.
    # Put a bogus "home-level" .aura/mcp_servers.json (not the global
    # store — a *different* file the walk would hit if it crossed into
    # home) to assert it's skipped. Here the global path IS that file,
    # so we only need to prove the walk doesn't blow up and that the
    # single-file contract (no duplicate) holds.
    mcp_store.save(
        [MCPServerConfig(name="g", transport="stdio", command="gcmd")],
        scope="global",
    )
    loaded = mcp_store.load()
    # Exactly one entry — no duplication from the walk passing through
    # $HOME/.aura/mcp_servers.json as both "global" AND "project".
    assert len(loaded) == 1
    assert loaded[0].name == "g"


def test_load_layer_returns_single_layer_only(
    isolated_home: Path, project_dir: Path,
) -> None:
    # The CLI duplicate-name check needs per-layer reads — make sure
    # load_layer is ACTUALLY per-layer, not merged.
    mcp_store.save(
        [MCPServerConfig(name="g", transport="stdio", command="gc")],
        scope="global",
    )
    mcp_store.save(
        [MCPServerConfig(name="p", transport="stdio", command="pc")],
        scope="project",
    )
    assert [s.name for s in mcp_store.load_layer("global")] == ["g"]
    assert [s.name for s in mcp_store.load_layer("project")] == ["p"]


def test_find_scope_of_project_wins(
    isolated_home: Path, project_dir: Path,
) -> None:
    mcp_store.save(
        [MCPServerConfig(name="shared", transport="stdio", command="g")],
        scope="global",
    )
    mcp_store.save(
        [MCPServerConfig(name="shared", transport="stdio", command="p")],
        scope="project",
    )
    # On collision, project owns the resolved entry.
    assert mcp_store.find_scope_of("shared") == "project"


def test_find_scope_of_global_only(
    isolated_home: Path, project_dir: Path,
) -> None:
    mcp_store.save(
        [MCPServerConfig(name="g-only", transport="stdio", command="g")],
        scope="global",
    )
    assert mcp_store.find_scope_of("g-only") == "global"


def test_find_scope_of_unknown_returns_none(
    isolated_home: Path, project_dir: Path,
) -> None:
    assert mcp_store.find_scope_of("ghost") is None


def test_project_layer_names_empty_when_no_project_files(
    isolated_home: Path, project_dir: Path,
) -> None:
    assert mcp_store.project_layer_names() == set()


def test_project_layer_names_picks_up_cwd(
    isolated_home: Path, project_dir: Path,
) -> None:
    mcp_store.save(
        [MCPServerConfig(name="p", transport="stdio", command="cmd")],
        scope="project",
    )
    assert mcp_store.project_layer_names() == {"p"}


def test_save_scope_project_creates_parent_dir(
    isolated_home: Path, project_dir: Path,
) -> None:
    # Fresh project dir — no ``.aura/`` yet. save(scope="project") must
    # create it, matching claude-code's "add server creates the file".
    assert not (project_dir / ".aura").exists()
    mcp_store.save(
        [MCPServerConfig(name="p", transport="stdio", command="cmd")],
        scope="project",
    )
    assert (project_dir / ".aura").is_dir()
    assert (project_dir / ".aura" / "mcp_servers.json").is_file()
