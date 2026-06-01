"""Two-layer mcp_servers.json store: global (~/.aura/) + project (<cwd>/.aura/).

Merge invariants (consumed by load()):
  - Global layer first; each project layer overrides by ``name``.
  - Project walk goes outer→inner from cwd up to (excl.) $HOME; innermost wins.
  - Missing file at any layer = empty list (first-run friendly).
  - Validation runs through MCPServerConfig.model_validate at load time.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from aura.config.schema import MCPServerConfig
from aura.infrastructure.persistence import journal

Scope = Literal["global", "project"]


def get_path() -> Path:
    # Back-compat alias: callers that print "the" MCP store path mean global.
    return global_path()


def global_path() -> Path:
    return Path.home() / ".aura" / "mcp_servers.json"


def project_path(cwd: Path | None = None) -> Path:
    base = cwd if cwd is not None else Path.cwd()
    return base / ".aura" / "mcp_servers.json"


def _expand_in_place(item: dict[str, object], missing: list[str]) -> dict[str, object]:
    # Recursively expand ${VAR} / ${VAR:-default} in string leaves on a fresh
    # copy; missing refs accumulate in `missing` (deduped by the expander).
    from aura.infrastructure.mcp.adapter import (
        expand_env_vars,  # noqa: PLC0415  # deferred to break import cycle
    )

    def _walk(node: object) -> object:
        if isinstance(node, str):
            return expand_env_vars(node, _missing_log=missing)
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        return node

    out = _walk(item)
    assert isinstance(out, dict)
    return out


def _load_layer(path: Path) -> list[MCPServerConfig]:
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
    # Expand ${VAR} refs before validation so command/args/env/url/headers
    # round-trip transparently. Missing refs accumulate into one journal line.
    missing: list[str] = []
    expanded: list[dict[str, object]] = []
    for item in raw_servers:
        if isinstance(item, dict):
            expanded.append(_expand_in_place(item, missing))
        else:
            expanded.append(item)
    if missing:
        with contextlib.suppress(Exception):
            journal.write("mcp_env_var_missing", path=str(path), missing=sorted(set(missing)))
    try:
        return [MCPServerConfig.model_validate(item) for item in expanded]
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid server entry: {exc}") from exc


def _project_dirs_up_to_home(cwd: Path, home: Path) -> list[Path]:
    # cwd → ... → (excl.) home, outer-first. cwd outside $HOME = [cwd] only,
    # so a fake-home test setup doesn't walk into the developer's real FS.
    try:
        cwd.relative_to(home)
    except ValueError:
        return [cwd]

    dirs: list[Path] = []
    current = cwd
    while True:
        dirs.append(current)
        parent = current.parent
        if current == home or parent in (current, home):
            break
        current = parent
    dirs.reverse()
    return dirs


def _load_project_layers(cwd: Path) -> list[list[MCPServerConfig]]:
    home = Path.home().resolve()
    cwd_resolved = cwd.resolve()
    global_file = global_path().resolve()

    layers: list[list[MCPServerConfig]] = []
    for project_dir in _project_dirs_up_to_home(cwd_resolved, home):
        candidate = project_dir / ".aura" / "mcp_servers.json"
        try:
            resolved_candidate = candidate.resolve()
        except OSError:
            resolved_candidate = candidate
        # Skip if cwd == $HOME would otherwise double-count the global layer.
        if resolved_candidate == global_file:
            continue
        layers.append(_load_layer(candidate))
    return layers


def load() -> list[MCPServerConfig]:
    global_servers = _load_layer(global_path())
    project_layers = _load_project_layers(Path.cwd())

    # Dict for override semantics + insertion order. pop-then-set so project
    # entries take their own outer→inner position, not the global's.
    merged: dict[str, MCPServerConfig] = {s.name: s for s in global_servers}
    for layer in project_layers:
        for s in layer:
            if s.name in merged:
                del merged[s.name]
            merged[s.name] = s
    return list(merged.values())


def save(servers: list[MCPServerConfig], *, scope: Scope = "global") -> None:
    if scope == "global":
        path = global_path()
    elif scope == "project":
        path = project_path()
    else:  # pragma: no cover — Literal narrows at the type layer.
        raise ValueError(f"unknown scope: {scope!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "servers": [s.model_dump(mode="json", exclude_defaults=False) for s in servers],
    }
    # indent=2: file is designed to be hand-edited.
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def load_layer(scope: Scope) -> list[MCPServerConfig]:
    if scope == "global":
        return _load_layer(global_path())
    if scope == "project":
        return _load_layer(project_path())
    raise ValueError(f"unknown scope: {scope!r}")  # pragma: no cover


def project_layer_names() -> set[str]:
    names: set[str] = set()
    for layer in _load_project_layers(Path.cwd()):
        names.update(s.name for s in layer)
    return names


def find_scope_of(name: str) -> Scope | None:
    # Innermost-project-first matches load()'s collision precedence.
    for layer in reversed(_load_project_layers(Path.cwd())):
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
