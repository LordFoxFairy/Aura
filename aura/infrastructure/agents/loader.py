"""Loader for filesystem subagent defs under .aura/agents/, overriding built-ins."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from aura.infrastructure.agents.builtin import builtin_agents
from aura.infrastructure.agents.types import AgentDef
from aura.infrastructure.persistence import journal

_AGENTS_SUBDIR = Path(".aura") / "agents"

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)\Z",
    re.DOTALL,
)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}, text
    body = match.group("body")
    parsed = yaml.safe_load(match.group("yaml")) or {}
    if not isinstance(parsed, dict):
        return {}, body
    return parsed, body


def _agent_from_file(path: Path) -> AgentDef | None:
    """Parse one ``.md`` file into an :class:`AgentDef`, journal+skip on failure."""
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
    """Built-ins merged with .aura/agents/*.md files, which override by name."""
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
            dir=str(agents_dir),
            error=f"{type(exc).__name__}: {exc}",
        )
        return out
    for path in files:
        agent = _agent_from_file(path)
        if agent is not None:
            out[agent.name] = agent
    return out


def get_agent_def(name: str, cwd: Path | str | None = None) -> AgentDef:
    """Lookup by name; error enumerates valid names so the LLM can self-correct."""
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
