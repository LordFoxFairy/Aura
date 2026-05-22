"""Filesystem-defined subagent type registry.

Built-in :class:`AgentDef` instances are merged with user-supplied markdown
files under ``.aura/agents/``. Filesystem entries override built-ins so a
project can customise (e.g. ``explore``'s prompt) without touching Aura code.
"""

from aura.infrastructure.agents.builtin import builtin_agents
from aura.infrastructure.agents.loader import (
    all_agent_defs,
    get_agent_def,
    load_agents,
)
from aura.infrastructure.agents.types import AgentDef

__all__ = [
    "AgentDef",
    "all_agent_defs",
    "builtin_agents",
    "get_agent_def",
    "load_agents",
]
