"""AgentDefinition — the static configuration of one conversation."""

from __future__ import annotations

from dataclasses import dataclass

# Stripped from every child's tool set so subagents cannot spawn subagents.
AGENT_DISALLOWED_TOOLS: frozenset[str] = frozenset({"task_create", "task_output"})


@dataclass(frozen=True)
class AgentDefinition:
    system_prompt: str
    model_spec: str
    permission_mode: str
    # None = every built-in tool; a frozenset pins an explicit allowlist.
    tool_names: frozenset[str] | None = None
