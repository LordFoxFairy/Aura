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
"""

from __future__ import annotations

from pathlib import PurePath

from aura.schemas.tool import ToolRuleMatcher


def exact_match_on(key: str) -> ToolRuleMatcher:
    """Matcher: ``args[key]`` must equal the rule's content verbatim."""

    def _matches(args: dict[str, object], content: str) -> bool:
        value = args.get(key)
        return isinstance(value, str) and value == content

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

    return _matches
