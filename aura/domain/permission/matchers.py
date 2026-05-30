"""Tool-agnostic rule matchers used by Rule.matches for pattern rules.

Matchers carry a .key attribute naming the args slot they inspect; the CLI
reads this to render precise rule hints. External matchers may omit it —
callers must use getattr(matcher, "key", None).
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path, PurePath

from aura.domain.tool import ToolRuleMatcher


def exact_match_on(key: str) -> ToolRuleMatcher:
    def _matches(args: dict[str, object], content: str) -> bool:
        value = args.get(key)
        if not isinstance(value, str):
            return False
        if "*" in content or "?" in content:
            return fnmatchcase(value, content)
        return value == content

    _matches.key = key  # type: ignore[attr-defined]  # WHY: tests + CLI read this slot
    return _matches


def path_prefix_on(key: str) -> ToolRuleMatcher:
    def _matches(args: dict[str, object], content: str) -> bool:
        value = args.get(key)
        if not isinstance(value, str):
            return False
        try:
            arg = _normalize_match_path(value)
            rule_path = _normalize_match_path(content)
        except (TypeError, ValueError):
            return False
        if arg == rule_path:
            return True
        try:
            return arg.is_relative_to(rule_path)
        except ValueError:
            return False

    _matches.key = key  # type: ignore[attr-defined]  # WHY: tests + CLI read this slot
    return _matches


def _normalize_match_path(raw: str) -> PurePath:
    # Collapse .. so /tmp/safe/../secret is not lexically under /tmp/safe.
    return Path(raw).expanduser().resolve(strict=False)
