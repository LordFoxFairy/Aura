"""Rule containers — immutable ``RuleSet`` + mutable ``SessionRuleSet``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool

from aura.domain.permission.rule import Rule


@dataclass(frozen=True)
class RuleSet:
    """Immutable ordered list of allow-rules; first-match-wins."""

    rules: tuple[Rule, ...] = ()

    def matches(
        self, tool_name: str, args: dict[str, Any], tool: BaseTool,
    ) -> Rule | None:
        return next(
            (r for r in self.rules if r.matches(tool_name, args, tool)), None,
        )


@dataclass
class SessionRuleSet:
    """Mutable in-memory ruleset for ``always`` session-scope answers.

    ``add()`` is idempotent on ``Rule`` equality. ``rules()`` returns a
    tuple snapshot so callers can't mutate the backing store.
    """

    _rules: list[Rule] = field(default_factory=list)

    def add(self, rule: Rule) -> None:
        if rule in self._rules:
            return
        self._rules.append(rule)

    def matches(
        self, tool_name: str, args: dict[str, Any], tool: BaseTool,
    ) -> Rule | None:
        return next(
            (r for r in self._rules if r.matches(tool_name, args, tool)), None,
        )

    def rules(self) -> tuple[Rule, ...]:
        return tuple(self._rules)

    def clear(self) -> None:
        self._rules.clear()
