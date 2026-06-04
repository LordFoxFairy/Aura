"""Subagent type registry merging built-in AgentDefs with filesystem overrides."""

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
