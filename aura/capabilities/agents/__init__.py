"""Filesystem-defined subagent type registry.

Replaces the legacy hard-coded ``aura/core/tasks/agent_types.py`` with
a claude-code-style loader that merges built-in :class:`AgentDef`s
with user-supplied markdown files under ``.aura/agents/``.

Public surface:

- :class:`AgentDef` — the dataclass describing one subagent type.
- :func:`load_agents` — ``{name: AgentDef}`` for a given project root.
- :func:`get_agent_def` — lookup by name (raises on unknown).
- :func:`all_agent_defs` — ordered tuple for catalogue rendering.
- :data:`BUILTIN_AGENT_DEFS` — the four built-in defs (read-only).
"""

from aura.capabilities.agents.builtin import BUILTIN_AGENT_DEFS, builtin_agents
from aura.capabilities.agents.loader import (
    AGENTS_SUBDIR,
    all_agent_defs,
    get_agent_def,
    load_agents,
)
from aura.capabilities.agents.types import AgentDef

__all__ = [
    "AGENTS_SUBDIR",
    "BUILTIN_AGENT_DEFS",
    "AgentDef",
    "all_agent_defs",
    "builtin_agents",
    "get_agent_def",
    "load_agents",
]
