"""Tool-agnostic rule matchers — ``exact_match_on`` and ``path_prefix_on``.

A ``rule_matcher`` in a tool's metadata decides whether a pattern rule like
``"bash(npm test)"`` covers a specific invocation. These helpers cover the
two shapes every built-in tool needs:

- ``exact_match_on(key)`` — the arg at ``args[key]`` must equal the rule
  content verbatim. Used by tools whose input is a single scalar (bash's
  ``command``, glob/grep's ``pattern``, web_fetch's ``url``).

- ``path_prefix_on(key)`` — the arg at ``args[key]`` must equal OR be a
  descendant of the rule content, treated as a filesystem path.
  ``"/tmp"`` covers ``"/tmp/foo.txt"`` but NOT ``"/tmpfoo"`` — component
  boundary matters. Used by read/write/edit_file.

Both factories return callables matching the ``ToolRuleMatcher`` protocol
(``(args: dict, content: str) -> bool``). They defensively return False on
missing keys or non-string values rather than raising — matchers run inside
the permission gate and must never crash the agent.

**Convention: ``.key`` attribute.** Each returned callable carries the arg
key it inspects on a ``.key`` attribute. This is the single source of truth
for "which arg does this matcher key off" — the CLI uses it to derive a
precise ``rule_hint`` for option 2 ("Yes, always allow ``bash(npm test)``")
without duplicating the key into a parallel metadata slot. External
(non-``matchers``-module) matchers are free to omit ``.key``; callers must
use ``getattr(matcher, "key", None)`` and treat ``None`` as "no precise
rule available, fall back to tool-wide".
"""

from __future__ import annotations

from pathlib import PurePath

from aura.schemas.tool import ToolRuleMatcher


def exact_match_on(key: str) -> ToolRuleMatcher:
    """Matcher: ``args[key]`` must equal the rule's content verbatim."""

    def _matches(args: dict[str, object], content: str) -> bool:
        value = args.get(key)
        return isinstance(value, str) and value == content

    _matches.key = key  # type: ignore[attr-defined]
    return _matches


def path_prefix_on(key: str) -> ToolRuleMatcher:
    """Matcher: ``args[key]`` must be ``content`` or a descendant path."""

    def _matches(args: dict[str, object], content: str) -> bool:
        value = args.get(key)
        if not isinstance(value, str):
            return False
        try:
            arg = PurePath(value)
            rule_path = PurePath(content)
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
