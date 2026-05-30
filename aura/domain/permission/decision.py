"""Permission decision — the outcome of the permission gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aura.domain.permission.rule import Rule

DecisionReason = Literal[
    "rule_allow",
    "rule_deny",
    "user_accept",
    "user_always",
    "mode_bypass",
    "mode_accept_edits",
    "user_deny",
    "safety_blocked",
    "plan_mode_blocked",
    "restrict_tools_blocked",
    "chain_empty",
]

_ALLOW_REASONS: frozenset[str] = frozenset({
    "rule_allow", "user_accept", "user_always",
    "mode_bypass", "mode_accept_edits", "chain_empty",
})
_DENY_REASONS: frozenset[str] = frozenset({
    "rule_deny",
    "user_deny",
    "safety_blocked",
    "plan_mode_blocked",
    "restrict_tools_blocked",
})
_RULE_REQUIRED_REASONS: frozenset[str] = frozenset({"rule_allow", "user_always"})


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: DecisionReason
    rule: Rule | None = None
    # Populated only for safety_blocked; args omitted (may carry secrets).
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
        match self.reason:
            case "rule_allow":
                assert self.rule is not None
                return f"auto-allowed: rule `{self.rule.to_string()}`"
            case "rule_deny":
                if self.rule is not None:
                    return f"blocked: deny rule `{self.rule.to_string()}`"
                return "blocked: deny rule"
            case "mode_bypass":
                return "allowed: mode_bypass"
            case "mode_accept_edits":
                return "allowed: mode_accept_edits"
            case "user_accept":
                return "allowed: user"
            case "user_always":
                assert self.rule is not None
                return f"allowed: user (rule saved: `{self.rule.to_string()}`)"
            case "user_deny":
                return "denied: user"
            case "safety_blocked":
                return "blocked: safety"
            case "plan_mode_blocked":
                return "blocked: plan mode (dry-run)"
            case "restrict_tools_blocked":
                return "blocked: skill restrict-tools whitelist"
            case "chain_empty":
                return "allowed: no permission hook"
