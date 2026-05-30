"""AgentDefinition — the static configuration of one conversation."""

from __future__ import annotations

from dataclasses import dataclass

# A subagent cannot spawn further subagents: recursion is bounded at one level
# by stripping these from every child's tool set (claude-code tool-exclusion).
AGENT_DISALLOWED_TOOLS: frozenset[str] = frozenset({"task_create", "task_output"})


@dataclass(frozen=True)
class AgentDefinition:
    system_prompt: str
    model_spec: str
    permission_mode: str
    # None = every built-in tool; a frozenset pins an explicit allowlist.
    tool_names: frozenset[str] | None = None
