"""Permission decision — the outcome of the permission gate.

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §3.2–§3.3.

Every tool invocation produces exactly one ``Decision``. Its ``reason`` is
closed-set (``DecisionReason``) and governs two invariants enforced in
``__post_init__``:

1. Each reason implies a fixed ``allow`` value. ``safety_blocked`` always
   denies; ``mode_bypass`` always allows; etc. Constructing a Decision with
   the wrong ``allow`` raises ``ValueError`` so the caller notices the bug
   at construction time, not later when auditing log rows.
2. ``rule_allow`` and ``user_always`` carry a concrete ``Rule`` (the one
   that matched, or the one just persisted). Other reasons leave ``rule``
   None.

``audit_line()`` returns the short one-liner the CLI renderer dims and
appends after the tool-call label — see §8.4 for the exact output examples.

Historical note: the ``read_only`` reason existed prior to the Plan B
refactor (2026-04-21) to mark auto-allows for tools declaring
``is_read_only=True``. That branch was removed because it bypassed the
safety list. Tools that used to be auto-allowed via ``read_only`` now
flow through ``rule_allow`` against ``DEFAULT_ALLOW_RULES`` (see
``aura/core/permissions/defaults.py``), which keeps safety in the path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aura.core.permissions.rule import Rule

DecisionReason = Literal[
    "rule_allow",
    "user_accept",
    "user_always",
    "mode_bypass",
    "user_deny",
    "safety_blocked",
]

_ALLOW_REASONS: frozenset[str] = frozenset(
    {"rule_allow", "user_accept", "user_always", "mode_bypass"}
)
_DENY_REASONS: frozenset[str] = frozenset({"user_deny", "safety_blocked"})
_RULE_REQUIRED_REASONS: frozenset[str] = frozenset({"rule_allow", "user_always"})


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: DecisionReason
    rule: Rule | None = None
    # Populated when reason == "safety_blocked" with the resolved path that
    # tripped the policy. Audit consumers record this so operators can see
    # *which* path was rejected without re-running. Args as a whole are not
    # captured (they can carry secrets); the target is already on a public
    # protected list so logging it reveals nothing new.
    target: str | None = None

    def __post_init__(self) -> None:
        if self.reason in _ALLOW_REASONS and not self.allow:
            raise ValueError(
                f"reason {self.reason!r} implies allow=True, got allow=False"
            )
        if self.reason in _DENY_REASONS and self.allow:
            raise ValueError(
                f"reason {self.reason!r} implies allow=False, got allow=True"
            )
        if self.reason in _RULE_REQUIRED_REASONS and self.rule is None:
            raise ValueError(
                f"reason {self.reason!r} requires a rule, got None"
            )
        if self.target is not None and self.reason != "safety_blocked":
            raise ValueError(
                f"reason {self.reason!r} must not carry a target; "
                "target is only meaningful for safety_blocked"
            )

    def audit_line(self) -> str:
        """One-line summary for the renderer + journal.

        "auto-allowed" marks passive allows (the user was never prompted);
        "allowed" marks active allows (user accepted, bypass mode, etc.).
        """
        match self.reason:
            case "rule_allow":
                assert self.rule is not None  # invariant
                return f"auto-allowed: rule `{self.rule.to_string()}`"
            case "mode_bypass":
                return "allowed: mode_bypass"
            case "user_accept":
                return "allowed: user"
            case "user_always":
                assert self.rule is not None  # invariant
                return f"allowed: user (rule saved: `{self.rule.to_string()}`)"
            case "user_deny":
                return "denied: user"
            case "safety_blocked":
                return "blocked: safety"
