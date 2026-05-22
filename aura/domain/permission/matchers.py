"""Tool-agnostic rule matchers — ``exact_match_on`` and ``path_prefix_on``.

Each matcher decides whether a pattern rule like ``bash(npm test)`` covers
a specific invocation. Both defensively return False on missing keys /
non-string values rather than raising.

The returned callables carry a ``.key`` attribute naming the arg they
inspect. The CLI uses this to derive a precise rule hint without a
parallel metadata slot. External matchers may omit ``.key``; callers
must read it via ``getattr(matcher, "key", None)``.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path, PurePath

from aura.schemas.tool import ToolRuleMatcher


def exact_match_on(key: str) -> ToolRuleMatcher:
    """Matcher: ``args[key]`` matches the rule's content.

    Glob metachars (``*`` / ``?``) in ``content`` switch to
    :func:`fnmatchcase`; without metachars the comparison is verbatim
    equality.
    """

    def _matches(args: dict[str, object], content: str) -> bool:
        value = args.get(key)
        if not isinstance(value, str):
            return False
        if "*" in content or "?" in content:
            return fnmatchcase(value, content)
        return value == content

    _matches.key = key  # type: ignore[attr-defined]
    return _matches


def path_prefix_on(key: str) -> ToolRuleMatcher:
    """Matcher: ``args[key]`` must be ``content`` or a descendant path."""

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

    _matches.key = key  # type: ignore[attr-defined]
    return _matches


def _normalize_match_path(raw: str) -> PurePath:
    """Collapse ``..`` so ``/tmp/safe/../secret`` is not lexically under
    ``/tmp/safe``. ``resolve(strict=False)`` works for both existing and
    not-yet-created paths.
    """
    return Path(raw).expanduser().resolve(strict=False)
