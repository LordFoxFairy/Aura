"""Permission subsystem — rules, decisions, modes, safety, persistence.

Spec: ``docs/specs/2026-04-19-aura-permission.md``.

Deliberate deferral — bash second-tier classifier (F-04-003)
============================================================

claude-code's permission pipeline has a second-tier "is this bash command
safe?" classifier that fires AFTER the rule layer misses but BEFORE the
asker. It uses a small LLM to look at e.g. ``rm /tmp/foo`` and auto-allow
it without prompting, while still asking for ``rm -rf $HOME``.

Aura intentionally does not ship this tier today:

- The classifier requires an extra model call per uncovered bash command,
  which is real latency + cost on the hot path. A heuristic-only version
  (regex table) would be the alternative but is brittle (the regex
  coverage gap is exactly what F-02-016 is about).
- Without the classifier, every uncovered bash command falls through to
  the asker. UX is noisier than claude-code's, but the safety floor is
  the same — ``bash_safety.py`` already hard-blocks the genuinely
  dangerous calls (``rm -rf /``, fork bombs, etc.) before any prompt
  fires.
- Adding it later is non-disruptive: the asker fallback would simply
  stop firing for commands the classifier auto-allows.

This is documented here (not deferred to a TODO file) so future
contributors see the reasoning when they wonder why the gate looks
shorter than upstream's. See ``docs/audit-2026-04-25/04-permission-
security.md`` F-04-003 for the original audit finding.
"""

from aura.core.permissions.mode import DEFAULT_MODE, Mode
from aura.core.permissions.safety import DEFAULT_SAFETY, SafetyPolicy, is_protected

__all__ = [
    "DEFAULT_MODE",
    "DEFAULT_SAFETY",
    "Mode",
    "SafetyPolicy",
    "is_protected",
]
