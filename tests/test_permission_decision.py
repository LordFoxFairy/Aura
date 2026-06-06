"""Tests for aura.domain.permission.decision — Decision shape + invariants."""

from __future__ import annotations

import dataclasses

import pytest

from aura.domain.permission.decision import Decision, DecisionReason
from aura.domain.permission.rule import Rule


def test_simple_allow_decision_has_no_rule() -> None:
    d = Decision(allow=True, reason="user_accept")
    assert d.allow is True
    assert d.reason == "user_accept"
    assert d.rule is None


def test_rule_allow_reason_requires_rule() -> None:
    with pytest.raises(ValueError, match="rule"):
        Decision(allow=True, reason="rule_allow")


def test_user_always_reason_requires_rule() -> None:
    with pytest.raises(ValueError, match="rule"):
        Decision(allow=True, reason="user_always")


def test_safety_blocked_reason_must_be_deny() -> None:
    with pytest.raises(ValueError, match="allow"):
        Decision(allow=True, reason="safety_blocked")


def test_mode_bypass_reason_must_be_allow() -> None:
    with pytest.raises(ValueError, match="allow"):
        Decision(allow=False, reason="mode_bypass")


def test_user_deny_reason_must_be_deny() -> None:
    with pytest.raises(ValueError, match="allow"):
        Decision(allow=True, reason="user_deny")


def test_decision_is_frozen() -> None:
    d = Decision(allow=True, reason="user_accept")
    with pytest.raises((AttributeError, TypeError)):
        d.__setattr__("allow", False)


def test_rule_allow_with_rule_roundtrips() -> None:
    rule = Rule(tool="bash", content="npm test")
    d = Decision(allow=True, reason="rule_allow", rule=rule)
    assert d.rule is rule


def test_target_rejected_when_reason_is_not_safety_blocked() -> None:
    # Invariant: target is only meaningful for safety_blocked. Any other
    # reason carrying a target means the audit record would misrepresent
    # what was evaluated — reject at construction so the bug never
    # reaches events.jsonl.
    with pytest.raises(ValueError, match="target"):
        Decision(
            allow=True, reason="rule_allow",
            rule=Rule(tool="read_file", content=None), target="/x",
        )
    with pytest.raises(ValueError, match="target"):
        Decision(allow=False, reason="user_deny", target="/x")


def test_safety_blocked_accepts_target() -> None:
    d = Decision(allow=False, reason="safety_blocked", target="/secret")
    assert d.target == "/secret"


def test_safety_blocked_without_target_still_valid() -> None:
    # Historical + defensive: a safety_blocked without target is permitted
    # (e.g. tests that don't care about the path). Only the *inverse* — a
    # target on a non-safety_blocked reason — is the invariant.
    d = Decision(allow=False, reason="safety_blocked")
    assert d.target is None


def test_audit_line_rule_allow_embeds_rule() -> None:
    rule = Rule(tool="bash", content="npm test")
    line = Decision(allow=True, reason="rule_allow", rule=rule).audit_line()
    assert line == "auto-allowed: rule `bash(npm test)`"


def test_audit_line_mode_bypass() -> None:
    assert Decision(allow=True, reason="mode_bypass").audit_line() == "allowed: mode_bypass"


def test_audit_line_user_accept() -> None:
    assert Decision(allow=True, reason="user_accept").audit_line() == "allowed: user"


def test_audit_line_user_always_embeds_saved_rule() -> None:
    rule = Rule(tool="bash", content="npm test")
    line = Decision(allow=True, reason="user_always", rule=rule).audit_line()
    assert line == "allowed: user (rule saved: `bash(npm test)`)"


def test_audit_line_user_deny() -> None:
    assert Decision(allow=False, reason="user_deny").audit_line() == "denied: user"


def test_audit_line_safety_blocked() -> None:
    assert Decision(allow=False, reason="safety_blocked").audit_line() == "blocked: safety"


def test_mode_accept_edits_is_allow_reason() -> None:
    d = Decision(allow=True, reason="mode_accept_edits")
    assert d.allow is True
    assert d.reason == "mode_accept_edits"
    assert d.rule is None


def test_mode_accept_edits_must_be_allow() -> None:
    with pytest.raises(ValueError, match="allow"):
        Decision(allow=False, reason="mode_accept_edits")


def test_plan_mode_blocked_is_deny_reason() -> None:
    d = Decision(allow=False, reason="plan_mode_blocked")
    assert d.allow is False
    assert d.reason == "plan_mode_blocked"
    assert d.rule is None


def test_plan_mode_blocked_must_be_deny() -> None:
    with pytest.raises(ValueError, match="allow"):
        Decision(allow=True, reason="plan_mode_blocked")


def test_audit_line_rule_deny_with_rule_names_the_rule() -> None:
    # A deny triggered by a stored deny-rule must surface WHICH rule fired so
    # the audit trail can be matched back to the user's settings.json entry.
    rule = Rule(tool="bash", content="rm -rf /", kind="deny")
    line = Decision(allow=False, reason="rule_deny", rule=rule).audit_line()
    assert line == "blocked: deny rule `bash(rm -rf /)`"


def test_audit_line_rule_deny_without_rule_falls_back() -> None:
    # rule_deny is not in _RULE_REQUIRED_REASONS, so the gate may emit it with
    # rule=None; the audit line must degrade gracefully, never crash on None.
    line = Decision(allow=False, reason="rule_deny").audit_line()
    assert line == "blocked: deny rule"


def test_audit_line_mode_accept_edits() -> None:
    # accept-edits mode auto-allows; the audit line must distinguish it from a
    # plain user accept so reviewers know the allow came from the mode, not a click.
    line = Decision(allow=True, reason="mode_accept_edits").audit_line()
    assert line == "allowed: mode_accept_edits"


def test_audit_line_plan_mode_blocked() -> None:
    # Plan mode is a dry-run: writes are blocked. The audit line must say so,
    # otherwise a silently-blocked tool looks like a runtime failure.
    line = Decision(allow=False, reason="plan_mode_blocked").audit_line()
    assert line == "blocked: plan mode (dry-run)"


def test_audit_line_restrict_tools_blocked() -> None:
    # A skill's restrict-tools whitelist denial must be auditable and
    # attributable to the whitelist, not confused with a safety block.
    line = Decision(allow=False, reason="restrict_tools_blocked").audit_line()
    assert line == "blocked: skill restrict-tools whitelist"


def test_audit_line_chain_empty() -> None:
    # No permission hook configured means default-allow; the audit must record
    # that the allow was the absence of a gate, not an explicit grant.
    line = Decision(allow=True, reason="chain_empty").audit_line()
    assert line == "allowed: no permission hook"


def test_restrict_tools_blocked_must_be_deny() -> None:
    # restrict-tools is a denial reason; constructing it as allow=True would let
    # a whitelisted-out tool slip through — reject at the boundary.
    with pytest.raises(ValueError, match="allow"):
        Decision(allow=True, reason="restrict_tools_blocked")


def test_chain_empty_must_be_allow() -> None:
    # chain_empty means "no gate, so allow"; an allow=False with this reason is
    # contradictory and must be rejected so callers can't fabricate a fake deny.
    with pytest.raises(ValueError, match="allow"):
        Decision(allow=False, reason="chain_empty")


def test_rule_deny_with_allow_true_is_rejected() -> None:
    # rule_deny is a deny reason; allow=True inverts the security meaning and
    # would auto-permit something the user denied — must raise.
    with pytest.raises(ValueError, match="allow"):
        Decision(allow=True, reason="rule_deny")


_DENY_REASONS: list[DecisionReason] = [
    "rule_deny",
    "user_deny",
    "safety_blocked",
    "plan_mode_blocked",
    "restrict_tools_blocked",
]
_ALLOW_REASONS: list[DecisionReason] = [
    "rule_allow",
    "user_accept",
    "user_always",
    "mode_bypass",
    "mode_accept_edits",
    "chain_empty",
]


@pytest.mark.parametrize("reason", _DENY_REASONS)
def test_all_deny_reasons_reject_allow_true(reason: DecisionReason) -> None:
    # Boundary matrix: every deny reason must forbid allow=True so no single
    # reason becomes a backdoor that flips a block into a grant.
    with pytest.raises(ValueError, match="allow"):
        Decision(allow=True, reason=reason)


@pytest.mark.parametrize("reason", _ALLOW_REASONS)
def test_all_allow_reasons_reject_allow_false(reason: DecisionReason) -> None:
    # Boundary matrix: every allow reason must forbid allow=False so an allow
    # cannot be silently downgraded to a deny by mislabeling the reason.
    with pytest.raises(ValueError, match="allow"):
        Decision(allow=False, reason=reason)


def test_chain_empty_carrying_target_is_rejected() -> None:
    # target is exclusive to safety_blocked; an allow-with-target on chain_empty
    # would smuggle a path into an audit record that never evaluated one.
    with pytest.raises(ValueError, match="target"):
        Decision(allow=True, reason="chain_empty", target="/etc/passwd")


def test_target_empty_string_is_treated_as_present() -> None:
    # An empty-string target is still "not None"; the invariant keys on identity
    # (is None), so "" on a non-safety reason must still be rejected — guarding
    # against a falsy-target loophole.
    with pytest.raises(ValueError, match="target"):
        Decision(allow=False, reason="user_deny", target="")


def test_safety_blocked_accepts_empty_string_target() -> None:
    # Conversely, safety_blocked may legitimately carry any target including ""
    # (e.g. a blocked tool with no path); it must be preserved verbatim.
    d = Decision(allow=False, reason="safety_blocked", target="")
    assert d.target == ""


def test_construction_is_idempotent_value_equality() -> None:
    # Decision is a frozen value object; two builds from identical inputs must be
    # equal and hashable so they dedup cleanly in audit/event sets.
    rule = Rule(tool="bash", content="npm test")
    a = Decision(allow=True, reason="rule_allow", rule=rule)
    b = Decision(allow=True, reason="rule_allow", rule=rule)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_post_init_runs_again_on_dataclasses_replace() -> None:
    # dataclasses.replace re-invokes __post_init__; flipping a valid allow into a
    # contradictory state via replace must still raise, proving the invariant is
    # enforced on every construction path, not just the literal constructor.
    valid = Decision(allow=True, reason="mode_bypass")
    with pytest.raises(ValueError, match="allow"):
        dataclasses.replace(valid, allow=False)
