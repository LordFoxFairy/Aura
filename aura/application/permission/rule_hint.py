"""Derive the most specific Rule the CLI can propose on prompt option 2."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from aura.domain.permission.rule import Rule
from aura.domain.tool import KeyedRuleMatcher
from aura.domain.tool_meta_access import meta_dict


def derive_rule_hint(tool: BaseTool, args: dict[str, Any]) -> Rule | None:
    matcher = meta_dict(tool).get("rule_matcher")
    if not isinstance(matcher, KeyedRuleMatcher):
        return None
    value = args.get(matcher.key)
    if not isinstance(value, str) or not value:
        return None
    return Rule(tool=tool.name, content=value)
