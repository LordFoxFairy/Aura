"""Rule containers — immutable ``RuleSet`` + mutable ``SessionRuleSet``.

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §3.5.

Two collections with the same ``matches()`` semantics (first-match-wins over
an ordered list of rules) but different mutability:

- ``RuleSet`` is a frozen snapshot loaded from ``./.aura/settings.json`` at
  agent startup. Immutable because persisted rules change only on disk.
- ``SessionRuleSet`` holds ``always`` choices that the user made this session
  but didn't persist (tool-wide rules where we refuse to silently write a
  broad rule to disk — see spec §8.2). Lives in memory only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool

from aura.core.permissions.rule import Rule


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

    ``add()`` is idempotent on ``Rule`` equality — re-answering the same
    prompt doesn't grow the list. ``rules()`` returns a tuple snapshot so
    callers can't accidentally mutate the backing store.
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
        """Drop all session-scope rules. Used by Agent.clear_session."""
        self._rules.clear()
