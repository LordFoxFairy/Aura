"""AgentDef — filesystem-loaded subagent type definition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentDef:
    """One subagent type definition.

    Fields:

    - ``name`` — stable identifier (e.g. ``general-purpose`` / ``explore``).
      Surfaced to the LLM as the ``agent_type`` enum.
    - ``description`` — LLM-facing one-liner. Rolled into ``task_create``.
    - ``tools`` — tool-name allowlist. Empty frozenset = "inherit all from
      parent" sentinel (distinct from an explicit zero-tool allowlist).
    - ``system_prompt_suffix`` — appended verbatim to the parent's system
      prompt when the child AgentSession is built.
    """

    name: str
    description: str
    tools: frozenset[str]
    system_prompt_suffix: str
