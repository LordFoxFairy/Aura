"""AgentDefinition — the static configuration of one conversation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentDefinition:
    system_prompt: str
    model_spec: str
    permission_mode: str
    # None = every built-in tool; a frozenset pins an explicit allowlist.
    tool_names: frozenset[str] | None = None
