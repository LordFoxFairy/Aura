"""User-editable JSON store for MCP server entries.

The store is a thin read/write wrapper around ``~/.aura/mcp_servers.json``.
Its shape mirrors :class:`aura.config.schema.MCPServerConfig` one-for-one so
the runtime ``MCPManager`` (which already consumes that pydantic model via
``AuraConfig.mcp_servers``) can consume this file without translation.

Design notes
~~~~~~~~~~~~
- Pure file I/O. No event-loop, no LLM, no journal — the ``aura mcp``
  subcommands must stay snappy and side-effect-free.
- First-run friendly: a missing file round-trips as an empty list, not an
  error. ``save()`` creates the parent directory if needed.
- Round-trip uses pydantic ``model_dump`` / ``model_validate`` so every
  validator on ``MCPServerConfig`` runs at load time — the disk file is
  the same contract as the in-memory config.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from aura.config.schema import MCPServerConfig


def get_path() -> Path:
    """Return ``~/.aura/mcp_servers.json`` (expanded, not necessarily existing)."""
    return Path.home() / ".aura" / "mcp_servers.json"


def load() -> list[MCPServerConfig]:
    """Read the store and return validated server entries.

    A missing file yields ``[]`` — the first-time user path. A malformed
    file (bad JSON, wrong top-level shape, failing validator) raises
    :class:`ValueError` so the caller surfaces the problem rather than
    silently eating the user's config.
    """
    path = get_path()
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected object at top level, got {type(data).__name__}")
    raw_servers = data.get("servers", [])
    if not isinstance(raw_servers, list):
        raise ValueError(
            f"{path}: 'servers' must be a list, got {type(raw_servers).__name__}"
        )
    try:
        return [MCPServerConfig.model_validate(item) for item in raw_servers]
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid server entry: {exc}") from exc


def save(servers: list[MCPServerConfig]) -> None:
    """Persist ``servers`` to disk, creating the parent directory if needed.

    The file is overwritten in place — the store is single-writer (the CLI)
    and there is no concurrent access to coordinate. The JSON layout is
    pretty-printed with ``indent=2`` because the file is designed to be
    hand-edited by the user.
    """
    path = get_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "servers": [s.model_dump(mode="json", exclude_defaults=False) for s in servers],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


__all__ = ["get_path", "load", "save"]
