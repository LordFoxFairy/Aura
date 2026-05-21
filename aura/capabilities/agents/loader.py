"""Filesystem loader for subagent type definitions.

Parity with claude-code's ``loadAgentsDir.ts``: built-in defs are
merged with user-supplied markdown files under ``.aura/agents/`` in the
project root. Filesystem entries override built-ins (so a project can
customise ``explore``'s prompt without touching Aura code).

The expected on-disk format is markdown with YAML frontmatter::

    ---
    name: research
    description: Deep-research subagent for long-context lookups.
    tools: [read_file, grep, glob, web_fetch, web_search]
    ---
    You are a **Research** subagent. Take your time. Cite sources.

Minimum required keys: ``name`` + ``description``. ``tools`` is optional
(omitted → empty frozenset → "inherit all"); the markdown body becomes
the ``system_prompt_suffix``. Unknown keys are ignored — a forward-
compatible escape hatch for richer formats later.

Failures during loading (malformed frontmatter, IO error, etc.) journal
+ skip the offending file; the built-in defs always remain available so
a broken user agent never strands a session.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aura.capabilities.agents.builtin import builtin_agents
from aura.capabilities.agents.types import AgentDef
from aura.core.persistence import journal

# Subdirectory under the project root where user-defined agents live.
# Matches the same ``.aura/`` convention used by other capability
# directories (skills, commands).
AGENTS_SUBDIR = Path(".aura") / "agents"

# Frontmatter delimiter. claude-code uses ``---`` (gray-matter); we
# mirror that so docs are interchangeable.
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)\Z",
    re.DOTALL,
)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown string into ``(frontmatter_dict, body)``.

    Uses :mod:`yaml` when available; falls back to a minimal line-based
    parser that handles ``key: value`` and ``key: [a, b, c]`` lists so
    the loader stays usable in environments without PyYAML installed.
    Returns ``({}, text)`` when no frontmatter delimiter is present.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}, text
    yaml_text = match.group("yaml")
    body = match.group("body")
    try:  # pragma: no cover — yaml import path
        import yaml as _yaml

        parsed = _yaml.safe_load(yaml_text) or {}
        if not isinstance(parsed, dict):
            return {}, body
        return parsed, body
    except ImportError:
        return _parse_simple_frontmatter(yaml_text), body


def _parse_simple_frontmatter(yaml_text: str) -> dict[str, Any]:
    """Minimal frontmatter parser used when PyYAML isn't installed.

    Handles ``key: value`` and ``key: [a, b]`` shapes only — sufficient
    for the documented agent format. Anything beyond that surfaces as
    an empty dict (and ``load_agents`` skips the file with a journal
    note rather than raising).
    """
    out: dict[str, Any] = {}
    for raw_line in yaml_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [
                v.strip().strip('"').strip("'")
                for v in value[1:-1].split(",")
                if v.strip()
            ]
            out[key] = items
        else:
            out[key] = value.strip('"').strip("'")
    return out


def _agent_from_file(path: Path) -> AgentDef | None:
    """Parse a single ``.md`` file into an :class:`AgentDef`.

    Returns ``None`` on any parse / IO failure; the loader treats those
    as "skip this file" rather than failing the whole load. A clean
    fallback to built-ins keeps the loader robust against typoed user
    files.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        journal.write(
            "agent_loader_read_error",
            path=str(path),
            error=f"{type(exc).__name__}: {exc}",
        )
        return None
    fm, body = _parse_frontmatter(text)
    # ``name`` defaults to the filename stem; an explicit frontmatter
    # ``name`` wins. Description is required — without it the LLM has
    # nothing to dispatch on, so we skip the file (mirrors claude-code's
    # zod validation refusing zero-length description).
    name = str(fm.get("name") or path.stem).strip()
    description = str(fm.get("description") or "").strip()
    if not name or not description:
        journal.write(
            "agent_loader_missing_fields",
            path=str(path),
            has_name=bool(name),
            has_description=bool(description),
        )
        return None
    tools_raw = fm.get("tools") or []
    if isinstance(tools_raw, str):
        # Tolerate a comma-separated string for hand-edited files.
        tools_raw = [t.strip() for t in tools_raw.split(",") if t.strip()]
    if not isinstance(tools_raw, list):
        tools_raw = []
    tools = frozenset(str(t) for t in tools_raw)
    return AgentDef(
        name=name,
        description=description,
        tools=tools,
        system_prompt_suffix=body.strip("\n"),
    )


def load_agents(cwd: Path | str | None = None) -> dict[str, AgentDef]:
    """Return ``{name: AgentDef}`` — built-ins merged with user files.

    ``cwd`` is the project root searched for ``.aura/agents/*.md``;
    defaults to the current working directory. Filesystem entries
    override built-ins of the same name (so a project can replace
    ``explore`` wholesale by dropping in ``.aura/agents/explore.md``).
    """
    out = builtin_agents()
    base = Path(cwd) if cwd is not None else Path.cwd()
    agents_dir = base / AGENTS_SUBDIR
    if not agents_dir.is_dir():
        return out
    try:
        files = sorted(agents_dir.glob("*.md"))
    except OSError as exc:
        journal.write(
            "agent_loader_dir_error",
            dir=str(agents_dir),
            error=f"{type(exc).__name__}: {exc}",
        )
        return out
    for path in files:
        agent = _agent_from_file(path)
        if agent is None:
            continue
        out[agent.name] = agent
    return out


def get_agent_def(name: str, cwd: Path | str | None = None) -> AgentDef:
    """Lookup an :class:`AgentDef` by name, raising ``ValueError`` if missing.

    Mirrors the legacy ``get_agent_type`` contract — error message
    enumerates the valid names so the LLM (which sees this via
    ToolError) can self-correct without another round-trip.
    """
    registry = load_agents(cwd)
    if name not in registry:
        valid = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"unknown agent_type {name!r}; valid: {valid}",
        )
    return registry[name]


def all_agent_defs(cwd: Path | str | None = None) -> tuple[AgentDef, ...]:
    """Return all registered agent defs in built-in-first order.

    Built-ins come first in their declared order (``general-purpose``,
    ``explore``, ``verify``, ``plan``); user-defined agents follow in
    filename order. Used by ``task_create`` to render the LLM-facing
    catalogue.
    """
    registry = load_agents(cwd)
    builtin = builtin_agents()
    builtin_order = [registry[n] for n in builtin if n in registry]
    extras = [registry[n] for n in sorted(registry) if n not in builtin]
    return tuple(builtin_order + extras)


__all__ = [
    "AGENTS_SUBDIR",
    "all_agent_defs",
    "get_agent_def",
    "load_agents",
]
