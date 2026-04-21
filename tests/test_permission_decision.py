"""Tests for aura.core.permissions.decision — Decision shape + invariants."""

from __future__ import annotations

import pytest

from aura.core.permissions.decision import Decision
from aura.core.permissions.rule import Rule


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
        d.allow = False  # type: ignore[misc]


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


# ---------------------------------------------------------------------------
# audit_line — one-liner the renderer dims and appends to tool-call output
# ---------------------------------------------------------------------------


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
