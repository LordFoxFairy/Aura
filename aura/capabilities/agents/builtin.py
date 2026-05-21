"""Built-in :class:`AgentDef` definitions — preserved verbatim from the
legacy ``aura/core/tasks/agent_types.py`` registry.

Aura ships four built-in subagent types that the LLM can pick via
``task_create(agent_type=...)``:

- ``general-purpose`` — full-access (inherit all parent tools).
- ``explore`` — read-only scanner.
- ``verify`` — strict audit with ``VERDICT: PASS/FAIL`` output.
- ``plan`` — read-only + plan-mode tools.

These are the fallback registry — :func:`load_agents` merges any
filesystem-defined agents from ``<cwd>/.aura/agents/*.md`` on top
(filesystem overrides win, mirroring claude-code's
``loadAgentsDir`` precedence).
"""

from __future__ import annotations

from aura.capabilities.agents.types import AgentDef

_GENERAL = AgentDef(
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

_EXPLORE = AgentDef(
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

_VERIFY = AgentDef(
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

_PLAN = AgentDef(
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


# Ordering matters for the LLM-facing catalogue: lead with
# ``general-purpose`` as the obvious default, then sort the restricted
# types by increasing specificity (explore → verify → plan).
BUILTIN_AGENT_DEFS: tuple[AgentDef, ...] = (_GENERAL, _EXPLORE, _VERIFY, _PLAN)


def builtin_agents() -> dict[str, AgentDef]:
    """Return a fresh dict of built-in :class:`AgentDef`s keyed by name.

    Returning a fresh dict (rather than exposing a module-level mutable
    one) keeps callers from accidentally pinning their own user-loaded
    overlay onto the shared registry.
    """
    return {d.name: d for d in BUILTIN_AGENT_DEFS}


__all__ = ["BUILTIN_AGENT_DEFS", "builtin_agents"]
