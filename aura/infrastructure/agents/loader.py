"""Filesystem loader for subagent type definitions.

Built-in defs are merged with user-supplied markdown files under
``.aura/agents/`` in the project root. Filesystem entries override
built-ins (so a project can customise ``explore``'s prompt without
touching Aura code).

Format: markdown with YAML frontmatter::

    ---
    name: research
    description: Deep-research subagent for long-context lookups.
    tools: [read_file, grep, glob, web_fetch, web_search]
    ---
    You are a **Research** subagent. Take your time. Cite sources.

Minimum required keys: ``name`` + ``description``. ``tools`` is optional
(omitted → empty frozenset → "inherit all"); the markdown body becomes
the ``system_prompt_suffix``. Unknown keys are ignored. Malformed files
journal + skip; built-ins always remain available.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aura.infrastructure.agents.builtin import builtin_agents
from aura.infrastructure.agents.types import AgentDef
from aura.infrastructure.persistence import journal

_AGENTS_SUBDIR = Path(".aura") / "agents"

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)\Z",
    re.DOTALL,
)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown string into ``(frontmatter_dict, body)``.

    Uses :mod:`yaml` when available; falls back to a minimal line parser
    that handles ``key: value`` and ``key: [a, b, c]`` shapes so the
    loader stays usable without PyYAML.
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
    """Minimal fallback parser — handles ``key: value`` and ``key: [a, b]``."""
    out: dict[str, Any] = {}
    for raw_line in yaml_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            out[key] = [
                v.strip().strip('"').strip("'")
                for v in value[1:-1].split(",")
                if v.strip()
            ]
        else:
            out[key] = value.strip('"').strip("'")
    return out


def _agent_from_file(path: Path) -> AgentDef | None:
    """Parse one ``.md`` file into an :class:`AgentDef`, journal+skip on failure."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        journal.write(
            "agent_loader_read_error",
            path=str(path), error=f"{type(exc).__name__}: {exc}",
        )
        return None
    fm, body = _parse_frontmatter(text)
    name = str(fm.get("name") or path.stem).strip()
    description = str(fm.get("description") or "").strip()
    if not name or not description:
        journal.write(
            "agent_loader_missing_fields",
            path=str(path), has_name=bool(name), has_description=bool(description),
        )
        return None
    tools_raw = fm.get("tools") or []
    if isinstance(tools_raw, str):
        tools_raw = [t.strip() for t in tools_raw.split(",") if t.strip()]
    if not isinstance(tools_raw, list):
        tools_raw = []
    return AgentDef(
        name=name,
        description=description,
        tools=frozenset(str(t) for t in tools_raw),
        system_prompt_suffix=body.strip("\n"),
    )


def load_agents(cwd: Path | str | None = None) -> dict[str, AgentDef]:
    """Return ``{name: AgentDef}`` — built-ins merged with user files.

    ``cwd`` is the project root searched for ``.aura/agents/*.md`` (defaults
    to the current working directory). Filesystem entries override built-ins
    of the same name.
    """
    out = builtin_agents()
    base = Path(cwd) if cwd is not None else Path.cwd()
    agents_dir = base / _AGENTS_SUBDIR
    if not agents_dir.is_dir():
        return out
    try:
        files = sorted(agents_dir.glob("*.md"))
    except OSError as exc:
        journal.write(
            "agent_loader_dir_error",
            dir=str(agents_dir), error=f"{type(exc).__name__}: {exc}",
        )
        return out
    for path in files:
        agent = _agent_from_file(path)
        if agent is not None:
            out[agent.name] = agent
    return out


def get_agent_def(name: str, cwd: Path | str | None = None) -> AgentDef:
    """Lookup an :class:`AgentDef` by name; raise ``ValueError`` if missing.

    Error message enumerates valid names so the LLM (which sees this via
    ToolError) can self-correct.
    """
    registry = load_agents(cwd)
    if name not in registry:
        valid = ", ".join(sorted(registry.keys()))
        raise ValueError(f"unknown agent_type {name!r}; valid: {valid}")
    return registry[name]


def all_agent_defs(cwd: Path | str | None = None) -> tuple[AgentDef, ...]:
    """All registered agent defs in built-in-first then filename order."""
    registry = load_agents(cwd)
    builtin = builtin_agents()
    builtin_order = [registry[n] for n in builtin if n in registry]
    extras = [registry[n] for n in sorted(registry) if n not in builtin]
    return tuple(builtin_order + extras)
