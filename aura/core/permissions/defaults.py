"""Built-in default allow-rules composed with user rules at startup.

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §5 (post Plan B refactor).

Why this module exists
----------------------

Before the Plan B refactor (2026-04-21), the permission hook auto-allowed any
tool whose metadata declared ``is_read_only=True`` via a dedicated decision
branch — step 3 in the old spec §5 order. That branch short-circuited
*before* rule matching, which meant a read-only tool never touched the
RuleSet, never produced ``rule_allow``, and — the real bug — skipped the
safety list entirely. ``read_file("~/.ssh/id_rsa")`` was silently
auto-allowed because the safety policy only fired on destructive tools.

The fix is structural, not cosmetic: the decision order collapses back to
``bypass → safety → rules → ask``, and the friction-reduction role of the
old step 3 moves here — into a tuple of built-in allow-rules composed with
user rules at agent startup as ``RuleSet(rules=disk_rules.rules + DEFAULT_ALLOW_RULES)``
— user rules first so their audit entry wins when both match; defaults
act as the backstop.

What it gives us
----------------

- **Composable.** A user can add their own rules in
  ``.aura/settings.json``; built-in defaults are appended at startup. Same
  type (``Rule``), same matching semantics (``RuleSet.matches``), same
  journal reason (``rule_allow``).
- **Visible.** The journal event for a default-allowed tool records the
  matched rule string (``read_file``), not a magic reason like
  ``read_only``. Auditors see exactly which rule fired.
- **Safety-aware.** Rules sit downstream of the safety check (§5 step 2),
  so a default-allowed ``read_file`` on ``~/.ssh/id_rsa`` still blocks — the
  read-protected list catches it before the rule gets a chance.

What counts as "friction-low enough to auto-allow"
--------------------------------------------------

Pure local-filesystem reads whose blast radius is "you already have the
bytes on disk": ``read_file``, ``grep``, ``glob``. Not ``web_fetch`` (leaks
outbound), not ``bash`` (arbitrary code). The safety rail (§6) remains the
backstop for sensitive paths regardless of whether a rule matched.
"""

from __future__ import annotations

from aura.core.permissions.rule import Rule

DEFAULT_ALLOW_RULES: tuple[Rule, ...] = (
    Rule(tool="read_file", content=None),
    Rule(tool="grep", content=None),
    Rule(tool="glob", content=None),
)

__all__ = ["DEFAULT_ALLOW_RULES"]
