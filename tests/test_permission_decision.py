"""Tests for aura.core.permissions.decision — Decision shape + invariants."""

from __future__ import annotations

import pytest

from aura.core.permissions.decision import Decision
from aura.core.permissions.rule import Rule


def test_simple_allow_decision_has_no_rule() -> None:
    d = Decision(allow=True, reason="read_only")
    assert d.allow is True
    assert d.reason == "read_only"
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


def test_read_only_reason_must_be_allow() -> None:
    with pytest.raises(ValueError, match="allow"):
        Decision(allow=False, reason="read_only")


def test_user_deny_reason_must_be_deny() -> None:
    with pytest.raises(ValueError, match="allow"):
        Decision(allow=True, reason="user_deny")


def test_decision_is_frozen() -> None:
    d = Decision(allow=True, reason="read_only")
    with pytest.raises((AttributeError, TypeError)):
        d.allow = False  # type: ignore[misc]


def test_rule_allow_with_rule_roundtrips() -> None:
    rule = Rule(tool="bash", content="npm test")
    d = Decision(allow=True, reason="rule_allow", rule=rule)
    assert d.rule is rule


# ---------------------------------------------------------------------------
# audit_line — one-liner the renderer dims and appends to tool-call output
# ---------------------------------------------------------------------------


def test_audit_line_read_only() -> None:
    assert Decision(allow=True, reason="read_only").audit_line() == "auto-allowed: read_only"


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
