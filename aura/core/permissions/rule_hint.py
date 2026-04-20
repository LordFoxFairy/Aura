"""Derive the most specific ``Rule`` the CLI can propose on prompt option 2.

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §8.2.

The CLI permission prompt's option 2 label ("Yes, and always allow …") needs
a concrete rule string to show the user. The best such rule is a *pattern*
rule like ``bash(npm test)`` — precise enough that the user sees exactly
what gets persisted. When we can't derive a precise rule, the caller (CLI)
falls back to tool-wide scope + session persistence; this helper returns
``None`` in that case and lets the CLI own the fallback policy.

A precise rule is derivable iff:

1. the tool's ``rule_matcher`` exposes a ``.key`` attribute (the convention
   from ``matchers.py`` — see its module docstring), AND
2. ``args[key]`` is a non-empty string.

Defensive at every step: missing metadata, missing matcher, missing key
attr, missing arg, non-string arg → ``None``, never raise.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from aura.core.permissions.rule import Rule


def derive_rule_hint(tool: BaseTool, args: dict[str, Any]) -> Rule | None:
    """Return the most specific pattern ``Rule`` for this call, or ``None``.

    See module docstring for derivation rules. Caller decides the fallback
    when ``None`` — this function never invents a tool-wide ``Rule``.
    """
    matcher = (tool.metadata or {}).get("rule_matcher")
    key = getattr(matcher, "key", None)
    if not isinstance(key, str):
        return None
    value = args.get(key)
    if not isinstance(value, str) or not value:
        return None
    return Rule(tool=tool.name, content=value)
