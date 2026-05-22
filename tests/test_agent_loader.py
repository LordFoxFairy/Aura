"""Filesystem-loaded subagent definitions.

Covers :mod:`aura.infrastructure.agents.loader`: built-ins always load,
markdown files with YAML frontmatter parse into ``AgentDef``s, missing
``.aura/agents`` directory falls back to built-ins only.
"""

from __future__ import annotations

from pathlib import Path

from aura.infrastructure.agents import (
    AgentDef,
    all_agent_defs,
    builtin_agents,
    load_agents,
)


def test_load_agents_returns_builtins_when_no_dir(tmp_path: Path) -> None:
    # Empty project root → loader sees no ``.aura/agents/`` and falls
    # straight through to the built-in registry; the four built-in
    # names must all be present.
    out = load_agents(tmp_path)
    names = set(out.keys())
    assert {"general-purpose", "explore", "verify", "plan"} <= names
    # Built-ins are AgentDef instances (not the legacy AgentTypeDef
    # shape) — guards against an accidental re-import after the rename.
    for d in out.values():
        assert isinstance(d, AgentDef)


def test_load_agents_picks_up_filesystem_md(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".aura" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "research.md").write_text(
        "---\n"
        "name: research\n"
        "description: Deep-research subagent for long-context lookups.\n"
        "tools: [read_file, grep, glob, web_fetch, web_search]\n"
        "---\n"
        "You are a **Research** subagent. Cite sources.\n",
        encoding="utf-8",
    )
    out = load_agents(tmp_path)
    assert "research" in out
    research = out["research"]
    assert research.name == "research"
    assert "Deep-research" in research.description
    assert {"read_file", "grep", "glob", "web_fetch", "web_search"} <= research.tools
    assert "Research" in research.system_prompt_suffix


def test_filesystem_overrides_builtin(tmp_path: Path) -> None:
    # A user-defined ``explore.md`` must REPLACE the built-in explore
    # def (not merge with it). Matches claude-code's loader precedence:
    # filesystem wins.
    agents_dir = tmp_path / ".aura" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "explore.md").write_text(
        "---\n"
        "name: explore\n"
        "description: Custom explore — tailored to this project.\n"
        "tools: [read_file]\n"
        "---\n"
        "Custom prompt body.\n",
        encoding="utf-8",
    )
    out = load_agents(tmp_path)
    explore = out["explore"]
    assert "Custom explore" in explore.description
    assert explore.tools == frozenset({"read_file"})


def test_missing_required_field_skips_file(tmp_path: Path) -> None:
    # File without a ``description`` is silently dropped (journaled) —
    # built-ins still load, so a typoed user file doesn't strand the
    # whole loader.
    agents_dir = tmp_path / ".aura" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "broken.md").write_text(
        "---\nname: broken\n---\nNo description provided.\n",
        encoding="utf-8",
    )
    out = load_agents(tmp_path)
    assert "broken" not in out
    assert "general-purpose" in out


def test_all_agent_defs_leads_with_builtins(tmp_path: Path) -> None:
    # Even with user-defined agents present, the four built-ins must
    # appear first in declaration order so the LLM-facing catalogue
    # surfaces the defaults before the project-specific extras.
    agents_dir = tmp_path / ".aura" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "zeta.md").write_text(
        "---\nname: zeta\ndescription: Zeta subagent.\n---\nBody.\n",
        encoding="utf-8",
    )
    defs = all_agent_defs(tmp_path)
    names = [d.name for d in defs]
    assert names[:4] == ["general-purpose", "explore", "verify", "plan"]
    assert "zeta" in names


def test_builtin_agents_returns_fresh_dict() -> None:
    # ``builtin_agents`` must hand back a fresh dict so mutations don't
    # pin onto the shared registry.
    a = builtin_agents()
    b = builtin_agents()
    assert a == b
    a.pop("explore")
    assert "explore" in builtin_agents()
