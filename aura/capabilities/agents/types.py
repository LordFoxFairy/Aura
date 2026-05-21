"""AgentDef — filesystem-loaded subagent type definition.

The :class:`AgentDef` dataclass replaces the old ``AgentTypeDef`` that
used to live in ``aura/core/tasks/agent_types.py``. Same fields, same
contract — what changed is where the definitions come from. Built-in
defs live in :mod:`aura.capabilities.agents.builtin`; user-supplied
defs are loaded from ``<cwd>/.aura/agents/*.md`` by
:func:`aura.capabilities.agents.loader.load_agents`.

Fields:

- ``name`` — stable identifier (``general-purpose`` / ``explore`` /
  ``verify`` / ``plan`` for the built-ins; user-defined agents can use
  any slug). Surfaced to the LLM as the ``agent_type`` enum.
- ``description`` — LLM-facing one-liner explaining when to pick this
  type. Rolled up into the ``task_create`` tool schema.
- ``tools`` — allowlist of tool names the child Agent may use. The
  empty frozenset is the "inherit all from parent" sentinel — distinct
  from an explicit empty allowlist. Only general-purpose uses the
  sentinel by convention.
- ``system_prompt_suffix`` — appended verbatim to the parent's system
  prompt when the child Agent is built.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentDef:
    """One subagent type definition.

    Deep-immutable: frozen dataclass + frozenset inner. The empty
    frozenset on ``tools`` is the "inherit all from parent" sentinel.
    See :meth:`SubagentFactory.spawn` for how it's interpreted.
    """

    name: str
    description: str
    tools: frozenset[str]
    system_prompt_suffix: str


__all__ = ["AgentDef"]
