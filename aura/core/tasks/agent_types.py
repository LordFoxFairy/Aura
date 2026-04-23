"""Subagent type registry — 4 flavors the LLM can pick via ``task_create(agent_type=...)``.

Each :class:`AgentTypeDef` declares:

- ``name``: stable identifier surfaced to the LLM as the enum of valid
  ``agent_type`` values.
- ``description``: LLM-facing one-liner that explains when to pick this type.
  Rolled up into the ``task_create`` tool schema so the parent model can make
  a strategic dispatch choice without needing a separate catalogue call.
- ``tools``: the allowlist of tool names the child Agent may use. The special
  empty frozenset acts as an "inherit all from parent" sentinel — distinct
  from an explicit empty allowlist (which would be strange and is not used by
  any built-in type). General-purpose is the only type that uses the sentinel.
- ``system_prompt_suffix``: appended verbatim to the parent's system prompt
  when the child Agent is built. For ``general-purpose`` this is empty so the
  child inherits the parent's identity unchanged.

Design parity vs. claude-code: claude-code ships ~6 subagent types; Aura
distils to 4 (general-purpose / explore / verify / plan) which cover the
strategic axes — "unrestricted", "read-only scan", "strict audit", "plan-only" —
without the UX cost of a longer enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AgentType = Literal["general-purpose", "explore", "verify", "plan"]


@dataclass(frozen=True)
class AgentTypeDef:
    """One row in the subagent-type registry.

    ``tools`` is a frozenset so the def is deep-immutable (frozen dataclass +
    frozenset inner). The empty frozenset is the "inherit all" sentinel — see
    :meth:`SubagentFactory.spawn` for how it's interpreted.
    """

    name: AgentType
    description: str
    tools: frozenset[str]
    system_prompt_suffix: str


_GENERAL = AgentTypeDef(
    name="general-purpose",
    description=(
        "Full-access subagent — inherits all parent tools. Use when the task "
        "requires arbitrary code edits, shell execution, or unknown "
        "capabilities. This is the default when no agent_type is specified."
    ),
    # Empty frozenset is the "inherit all from parent" sentinel. A real
    # allowlist of zero tools would make the subagent useless and is never
    # intended here — treat the emptiness as structural, not literal.
    tools=frozenset(),
    system_prompt_suffix="",
)

_EXPLORE = AgentTypeDef(
    name="explore",
    description=(
        "Read-only scanner. Use for code search, file reconnaissance, or any "
        "task that should NOT modify the repo. Tools: read_file, grep, glob, "
        "web_fetch, web_search. No bash, no write, no nested dispatch."
    ),
    tools=frozenset({"read_file", "grep", "glob", "web_fetch", "web_search"}),
    system_prompt_suffix=(
        "\n\n# Subagent context\n"
        "You are an **Explore** subagent. You ONLY have read-only tools "
        "(read_file, grep, glob, web_fetch, web_search). You cannot modify "
        "files or run shell commands. Scan, search, and summarize. Return "
        "findings concisely; the parent agent will act on them."
    ),
)

_VERIFY = AgentTypeDef(
    name="verify",
    description=(
        "Strict audit subagent. Read-only. Outputs exactly one line starting "
        "with 'VERDICT: PASS' or 'VERDICT: FAIL' plus a short justification. "
        "Use to independently validate a claim against the code."
    ),
    tools=frozenset({"read_file", "grep", "glob", "web_fetch", "web_search"}),
    system_prompt_suffix=(
        "\n\n# Subagent context\n"
        "You are a **Verify** subagent. You have read-only tools. Audit the "
        "given artifact against the claim and output EXACTLY one line starting "
        "with `VERDICT: PASS` or `VERDICT: FAIL` followed by a one-paragraph "
        "justification citing file:line evidence. Any other output is "
        "discarded."
    ),
)

_PLAN = AgentTypeDef(
    name="plan",
    description=(
        "Planning subagent. Read-only + plan-mode tools. Enters plan mode, "
        "gathers context, returns a concrete plan via exit_plan_mode. Does "
        "not modify anything itself."
    ),
    tools=frozenset({
        "read_file",
        "grep",
        "glob",
        "web_fetch",
        "web_search",
        "enter_plan_mode",
        "exit_plan_mode",
    }),
    system_prompt_suffix=(
        "\n\n# Subagent context\n"
        "You are a **Plan** subagent. Enter plan mode first, gather context "
        "via read-only tools, then call `exit_plan_mode` with a concrete, "
        "file-specific implementation plan. Do NOT modify anything yourself."
    ),
)


# Single source of truth for the 4 types. Ordering matters for
# ``all_agent_types`` (LLM-facing catalogue in the tool schema) — we lead
# with ``general-purpose`` as the obvious default, then sort the restricted
# types by increasing specificity (explore → verify → plan).
_REGISTRY: dict[AgentType, AgentTypeDef] = {
    "general-purpose": _GENERAL,
    "explore": _EXPLORE,
    "verify": _VERIFY,
    "plan": _PLAN,
}


def get_agent_type(name: str) -> AgentTypeDef:
    """Return the :class:`AgentTypeDef` registered under ``name``.

    Raises :class:`ValueError` with the full list of valid names when
    ``name`` is unknown — the error message is surfaced to the LLM via
    ``task_create`` so the model can self-correct on the next turn.
    """
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"unknown agent_type {name!r}; valid: {valid}"
        )
    return _REGISTRY[name]


def all_agent_types() -> tuple[AgentTypeDef, ...]:
    """All registered types in declaration order.

    Used by ``task_create``'s ``agent_type`` field description to render an
    LLM-readable catalogue so the model can pick the right flavor without a
    separate introspection call.
    """
    return tuple(_REGISTRY.values())
