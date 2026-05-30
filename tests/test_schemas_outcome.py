"""Tests for the ``Outcome`` tagged union in ``aura.schemas.permissions``.

Phase 1 Task 1 — schema contracts only. Hooks migrate to return
``Outcome`` in Tasks 8-10. These tests assert the variant shapes,
``__post_init__`` invariants, and frozen semantics.

Spec §3.2 — four variants:
- ``Allow(decision: Decision)`` — ``decision.allow`` MUST be True.
- ``Block(decision: Decision)`` — ``decision.allow`` MUST be False.
- ``Ask(reason: str)`` — non-empty ``reason``.
- ``Replace(result: ToolResult, decision: Decision)`` — ``decision.allow``
  MUST be False AND ``result is not None``.
"""

from __future__ import annotations

import dataclasses

import pytest

from aura.domain.permission.decision import Decision
from aura.domain.permission.outcome import Allow, Ask, Block, Outcome, Replace
from aura.domain.tool import ToolResult


def _allow_decision() -> Decision:
    """A valid allow ``Decision`` (uses ``mode_bypass`` — needs no rule)."""
    return Decision(allow=True, reason="mode_bypass")


def _deny_decision() -> Decision:
    return Decision(allow=False, reason="user_deny")


def test_allow_accepts_allow_true_decision() -> None:
    decision = _allow_decision()
    outcome = Allow(decision=decision)
    assert outcome.decision is decision


def test_allow_rejects_allow_false_decision() -> None:
    """Allow's invariant: ``decision.allow`` must be True. Constructing
    with a deny decision raises immediately so the bug surfaces at the
    hook boundary, not later when the loop tries to execute."""
    with pytest.raises(ValueError, match="Allow"):
        Allow(decision=_deny_decision())


def test_allow_is_frozen() -> None:
    outcome = Allow(decision=_allow_decision())
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.decision = _allow_decision()  # type: ignore[misc]  # rebinding/mutating frozen field for test


def test_block_accepts_allow_false_decision() -> None:
    decision = _deny_decision()
    outcome = Block(decision=decision)
    assert outcome.decision is decision


def test_block_rejects_allow_true_decision() -> None:
    with pytest.raises(ValueError, match="Block"):
        Block(decision=_allow_decision())


def test_block_is_frozen() -> None:
    outcome = Block(decision=_deny_decision())
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.decision = _deny_decision()  # type: ignore[misc]  # rebinding/mutating frozen field for test


def test_ask_accepts_non_empty_reason() -> None:
    outcome = Ask(reason="user must confirm destructive bash")
    assert outcome.reason == "user must confirm destructive bash"


def test_ask_rejects_empty_reason() -> None:
    """A blank ``Ask.reason`` reaches the user widget as a label —
    rejecting empty at construction prevents an unrenderable prompt."""
    with pytest.raises(ValueError, match="reason"):
        Ask(reason="")


def test_ask_rejects_whitespace_only_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        Ask(reason="   ")


def test_ask_is_frozen() -> None:
    outcome = Ask(reason="confirm please")
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.reason = "different"  # type: ignore[misc]  # rebinding/mutating frozen field for test


def test_replace_accepts_deny_decision_with_result() -> None:
    result = ToolResult(ok=True, output="synthetic replacement")
    decision = _deny_decision()
    outcome = Replace(result=result, decision=decision)
    assert outcome.result is result
    assert outcome.decision is decision


def test_replace_rejects_allow_true_decision() -> None:
    """Replace means 'tool not invoked, this result substitutes' — the
    decision MUST be a deny so audit consumers see the substitution
    accurately."""
    result = ToolResult(ok=True, output="x")
    with pytest.raises(ValueError, match="Replace"):
        Replace(result=result, decision=_allow_decision())


def test_replace_rejects_none_result() -> None:
    """``result is None`` would leave the loop with nothing to append
    as a ToolMessage — the substitution is meaningless without it."""
    with pytest.raises((ValueError, TypeError)):
        # deliberately off-type arg to exercise path
        Replace(result=None, decision=_deny_decision())  # type: ignore[arg-type]


def test_replace_is_frozen() -> None:
    outcome = Replace(
        result=ToolResult(ok=True, output="x"), decision=_deny_decision(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.decision = _deny_decision()  # type: ignore[misc]  # rebinding/mutating frozen field for test


def test_outcome_union_accepts_each_variant() -> None:
    """``Outcome`` is the union ``Allow | Block | Ask | Replace`` —
    pattern matching on the variant is the loop's contract."""
    variants: list[Outcome] = [
        Allow(decision=_allow_decision()),
        Block(decision=_deny_decision()),
        Ask(reason="confirm"),
        Replace(
            result=ToolResult(ok=True, output="x"), decision=_deny_decision(),
        ),
    ]
    # Sanity — the union contains all four shapes.
    kinds: set[str] = set()
    for outcome in variants:
        match outcome:
            case Allow():
                kinds.add("allow")
            case Block():
                kinds.add("block")
            case Ask():
                kinds.add("ask")
            case Replace():
                kinds.add("replace")
    assert kinds == {"allow", "block", "ask", "replace"}


def test_outcome_variants_exported_from_domain_outcome() -> None:
    """All four variant classes plus the ``Outcome`` alias live at the
    domain permission home."""
    from aura.domain.permission import outcome as outcome_mod

    assert hasattr(outcome_mod, "Allow")
    assert hasattr(outcome_mod, "Block")
    assert hasattr(outcome_mod, "Ask")
    assert hasattr(outcome_mod, "Replace")
    assert hasattr(outcome_mod, "Outcome")
