"""User-editable JSON store for MCP server entries.

The store is a thin read/write wrapper around two layers of
``mcp_servers.json`` files:

1. **Global** — ``~/.aura/mcp_servers.json`` (the original, pre-existing
   location). Holds the user's personal servers across all projects.
2. **Project** — ``<cwd>/.aura/mcp_servers.json`` and every ancestor of
   ``cwd`` up to (but excluding) ``$HOME``. Holds per-project server
   overrides that should travel with the repo.

The shape of each file mirrors :class:`aura.config.schema.MCPServerConfig`
one-for-one so the runtime ``MCPManager`` (which already consumes that
pydantic model via ``AuraConfig.mcp_servers``) can consume it without
translation.

Merge / override rules
~~~~~~~~~~~~~~~~~~~~~~

``load()`` stitches all layers together and returns a single list:

- Start from the global layer.
- Apply project layers from **outermost to innermost** (so the layer
  closest to ``cwd`` wins on a name collision).
- Project entries fully replace global entries of the same ``name``.

Claude-code mirrors this pattern (global / project / managed). We expose
only global + project because managed (enterprise-policy) config isn't a
concept in Aura yet.

Design notes
~~~~~~~~~~~~
- Pure file I/O. No event-loop, no LLM, no journal — the ``aura mcp``
  subcommands must stay snappy and side-effect-free.
- First-run friendly: a missing file round-trips as an empty list, not
  an error. ``save()`` creates the parent directory if needed.
- Round-trip uses pydantic ``model_dump`` / ``model_validate`` so every
  validator on ``MCPServerConfig`` runs at load time — the disk file is
  the same contract as the in-memory config.
- ``save(servers, *, scope=...)`` writes to the appropriate layer; the
  default ``scope="global"`` preserves the pre-project-layer behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from aura.config.schema import MCPServerConfig

Scope = Literal["global", "project"]


def get_path() -> Path:
    """Return ``~/.aura/mcp_servers.json`` (expanded, not necessarily existing).

    Kept for back-compat: existing callers (CLI messages, tests) that
    print / assert the "canonical" MCP store path always mean the global
    one. The project path varies by ``cwd``; callers that need it should
    use :func:`project_path` explicitly.
    """
    return global_path()


def global_path() -> Path:
    """Return the global (user) store path: ``~/.aura/mcp_servers.json``."""
    return Path.home() / ".aura" / "mcp_servers.json"


def project_path(cwd: Path | None = None) -> Path:
    """Return the project-layer store path at ``<cwd>/.aura/mcp_servers.json``.

    ``cwd`` defaults to :func:`Path.cwd`. This is the file a
    ``--scope project`` write targets; reads additionally walk ancestors
    up to (exclusive of) ``$HOME``.
    """
    base = cwd if cwd is not None else Path.cwd()
    return base / ".aura" / "mcp_servers.json"


def _expand_in_place(item: dict[str, object], missing: list[str]) -> dict[str, object]:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in *item*'s string values.

    Operates on a fresh copy so the caller's source data is never mutated.
    String leaves go through :func:`_expand_env_vars`; missing references
    accumulate in *missing* (deduplicated by the expander itself).
    Non-string leaves (ints, bools, None) pass through unchanged.
    """
    from aura.core.mcp.adapter import _expand_env_vars  # noqa: PLC0415

    def _walk(node: object) -> object:
        if isinstance(node, str):
            return _expand_env_vars(node, _missing_log=missing)
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        return node

    out = _walk(item)
    assert isinstance(out, dict)
    return out


def _load_layer(path: Path) -> list[MCPServerConfig]:
    """Parse and validate a single MCP store file.

    Missing file → empty list (both layers are optional). Malformed JSON
    / wrong top-level shape / validator failures all raise
    :class:`ValueError` with the path prefixed so the caller can tell
    which layer is wrong.
    """
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected object at top level, got {type(data).__name__}"
        )
    raw_servers = data.get("servers", [])
    if not isinstance(raw_servers, list):
        raise ValueError(
            f"{path}: 'servers' must be a list, got {type(raw_servers).__name__}"
        )
    # Round 6N — expand env vars in every string leaf before validation
    # so a ``${TOKEN}`` reference in command/args/env/url/headers
    # round-trips through the loader transparently. Missing references
    # accumulate per file and produce a single advisory journal warning.
    missing: list[str] = []
    expanded: list[dict[str, object]] = []
    for item in raw_servers:
        if isinstance(item, dict):
            expanded.append(_expand_in_place(item, missing))
        else:
            expanded.append(item)
    if missing:
        try:
            from aura.core import journal  # noqa: PLC0415
            journal.write(
                "mcp_env_var_missing",
                path=str(path),
                missing=sorted(set(missing)),
            )
        except Exception:  # noqa: BLE001
            pass
    try:
        return [MCPServerConfig.model_validate(item) for item in expanded]
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid server entry: {exc}") from exc


def _project_dirs_up_to_home(cwd: Path, home: Path) -> list[Path]:
    """Return cwd, cwd.parent, ..., up to (but NOT including) ``home``.

    Outermost-first order. Mirrors the walk-up pattern used by skills
    (:func:`aura.core.skills.loader._project_dirs_up_to_home`). If
    ``cwd`` is not under ``home`` (pathological test setups / CI
    running from ``/tmp``), returns ``[cwd]`` — we still scan the
    project layer at ``cwd``, but we don't traverse up into the real
    FS and risk reading the developer's unrelated ``/.aura/``
    directories when ``Path.home()`` is pointed at a fake home.
    """
    try:
        cwd.relative_to(home)
    except ValueError:
        # cwd outside $HOME — stop the walk at cwd itself.
        return [cwd]

    dirs: list[Path] = []
    current = cwd
    while True:
        dirs.append(current)
        parent = current.parent
        if parent == current:
            break  # Filesystem root.
        if current == home:
            break
        if parent == home:
            break  # Don't cross into $HOME.
        current = parent
    # Outer → inner order so a caller iterating linearly can apply
    # overrides in the right order (innermost wins).
    dirs.reverse()
    return dirs


def _load_project_layers(cwd: Path) -> list[list[MCPServerConfig]]:
    """Load every project-layer file from ``cwd`` up to (excl.) ``$HOME``.

    Skips directories whose ``mcp_servers.json`` is the same file as the
    global path (would happen if the user runs Aura from inside ``$HOME``
    itself — the walk would otherwise double-count that directory).
    Returned order matches :func:`_project_dirs_up_to_home`:
    **outermost first**.
    """
    home = Path.home().resolve()
    cwd_resolved = cwd.resolve()
    global_file = global_path().resolve()

    layers: list[list[MCPServerConfig]] = []
    for project_dir in _project_dirs_up_to_home(cwd_resolved, home):
        candidate = project_dir / ".aura" / "mcp_servers.json"
        try:
            # ``resolve`` on a non-existent path is fine in py3.11+ — it
            # normalises symlinks in the parents and returns the absolute
            # path. Guard anyway so test setups with aggressive symlink
            # layouts don't blow up.
            resolved_candidate = candidate.resolve()
        except OSError:
            resolved_candidate = candidate
        if resolved_candidate == global_file:
            # Don't double-count the global layer when cwd == $HOME.
            continue
        layers.append(_load_layer(candidate))
    return layers


def load() -> list[MCPServerConfig]:
    """Read all layers and return the merged list of validated servers.

    Merge order:

    1. Start from the global layer (``~/.aura/mcp_servers.json``).
    2. Walk ``cwd`` up toward (but excluding) ``$HOME``, applying each
       ``.aura/mcp_servers.json`` found. The innermost (closest to
       ``cwd``) project entry wins on name collision.

    A missing file at any layer yields ``[]`` — the first-time user
    path. A malformed file (bad JSON, wrong top-level shape, failing
    validator) raises :class:`ValueError` with the offending layer's
    path in the message.

    The returned list's order is stable: global entries first (in their
    saved order, minus any shadowed by a project layer), followed by
    project entries grouped outer-to-inner. This matches the "walk-up"
    reading order in claude-code.
    """
    global_servers = _load_layer(global_path())
    project_layers = _load_project_layers(Path.cwd())

    # Use a dict keyed by name for override semantics, but preserve
    # insertion order so the output is deterministic (dict is insertion-
    # ordered since py3.7). Global first; each project layer overwrites.
    merged: dict[str, MCPServerConfig] = {s.name: s for s in global_servers}
    for layer in project_layers:
        for s in layer:
            # Overwriting an existing key preserves its original position
            # in insertion order — we want project entries to appear in
            # THEIR outer→inner order, so pop-then-set.
            if s.name in merged:
                del merged[s.name]
            merged[s.name] = s
    return list(merged.values())


def save(servers: list[MCPServerConfig], *, scope: Scope = "global") -> None:
    """Persist ``servers`` to the layer named by ``scope``.

    - ``scope="global"`` (default) → ``~/.aura/mcp_servers.json``
      (preserves pre-project-layer behaviour so existing callers need
      no changes).
    - ``scope="project"`` → ``<cwd>/.aura/mcp_servers.json``.

    The parent directory is created if needed. The file is overwritten
    in place — the store is single-writer (the CLI) and there is no
    concurrent access to coordinate. JSON is pretty-printed with
    ``indent=2`` because the file is designed to be hand-edited.
    """
    if scope == "global":
        path = global_path()
    elif scope == "project":
        path = project_path()
    else:  # pragma: no cover — Literal narrows this at the type layer.
        raise ValueError(f"unknown scope: {scope!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "servers": [s.model_dump(mode="json", exclude_defaults=False) for s in servers],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def load_layer(scope: Scope) -> list[MCPServerConfig]:
    """Return the raw entries of a single layer without merging.

    Used by the CLI remove path (to know which layer owns a name) and
    tests. Keeps the per-layer file-I/O code in one place.
    """
    if scope == "global":
        return _load_layer(global_path())
    if scope == "project":
        return _load_layer(project_path())
    raise ValueError(f"unknown scope: {scope!r}")  # pragma: no cover


def project_layer_names() -> set[str]:
    """Return the set of server names present in any project-layer file.

    Convenience for CLI code that needs to tag each merged row with its
    originating scope without re-implementing the walk-up logic.
    """
    names: set[str] = set()
    for layer in _load_project_layers(Path.cwd()):
        names.update(s.name for s in layer)
    return names


def find_scope_of(name: str) -> Scope | None:
    """Return which layer currently contains ``name``, or ``None``.

    Project wins on collision (matches :func:`load`'s merge order), so
    we check project first. Walks ``cwd`` up to ``$HOME`` the same way
    :func:`load` does.
    """
    for layer in reversed(_load_project_layers(Path.cwd())):
        # Reversed → innermost first. We already know innermost wins on
        # collision, so the first layer that has the name is the owner.
        if any(s.name == name for s in layer):
            return "project"
    if any(s.name == name for s in _load_layer(global_path())):
        return "global"
    return None


__all__ = [
    "Scope",
    "find_scope_of",
    "get_path",
    "global_path",
    "load",
    "load_layer",
    "project_layer_names",
    "project_path",
    "save",
]
