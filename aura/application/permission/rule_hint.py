"""Derive the most specific ``Rule`` the CLI can propose on prompt option 2.

A precise rule is derivable iff the tool's ``rule_matcher`` exposes a
``.key`` attribute (matchers.py convention) AND ``args[key]`` is a
non-empty string. Returns ``None`` otherwise; the CLI then falls back to
tool-wide + session persistence.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from aura.domain.permission.rule import Rule
from aura.schemas.tool_meta_access import meta_dict


def derive_rule_hint(tool: BaseTool, args: dict[str, Any]) -> Rule | None:
    matcher = meta_dict(tool).get("rule_matcher")
    key = getattr(matcher, "key", None)
    if not isinstance(key, str):
        return None
    value = args.get(key)
    if not isinstance(value, str) or not value:
        return None
    return Rule(tool=tool.name, content=value)
