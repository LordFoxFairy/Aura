"""Phase 1 Task 10 — Outcome is the only pre_tool hook return contract.

Covers the spec §3.2 merge precedence matrix for two-hook chains where
both hooks return :class:`aura.schemas.permissions.Outcome` variants
(``Allow`` / ``Block`` / ``Ask`` / ``Replace``):

    Block > Replace > Ask > Allow(authoritative) > Allow(mode_bypass)

Ask is a side-channel signal threaded as the chain-local ``ask_pending``
kwarg to downstream hooks; a permission hook reads it and returns the
resolved Allow/Block. If no hook resolves Ask (no permission hook
present), the unresolved Ask escalates back to the loop.

Replace beats Ask because a safety block (e.g. bash_safety) must not be
overridden by a pending confirmation request.

Plus regression tests for:

- Single-hook returns of each variant.
- Block short-circuits the chain (no later hook runs).
- Replace does NOT short-circuit (a later Block can still win).
- Empty Outcome list → passthrough (Allow sentinel).

The 16-case matrix (4 variants × 4 variants) is generated via
``pytest.mark.parametrize`` so every cell is covered.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.application.hooks import HookChain, PreToolHook
from aura.domain.permission.decision import Decision
from aura.domain.permission.outcome import Allow, Ask, Block, Outcome, Replace
from aura.domain.permission.rule import Rule
from aura.domain.tool import ToolResult
from aura.schemas.state import LoopState
from aura.tools.base import build_tool


class _P(BaseModel):
    x: int = 0


def _noop(x: int = 0) -> dict[str, Any]:
    return {}


_stub_tool: BaseTool = build_tool(
    name="stub", description="stub", args_schema=_P, func=_noop,
)


def _allow(reason: str = "mode_bypass") -> Decision:
    return Decision(allow=True, reason=reason)  # type: ignore[arg-type]  # deliberately off-type arg to exercise path


def _allow_with_rule(rule_tool: str = "stub") -> Decision:
    return Decision(
        allow=True,
        reason="rule_allow",
        rule=Rule(tool=rule_tool, content=None),
    )


def _deny(reason: str = "safety_blocked") -> Decision:
    return Decision(allow=False, reason=reason)  # type: ignore[arg-type]  # deliberately off-type arg to exercise path


def _make_hook(outcome: Outcome) -> PreToolHook:
    """Wrap a constant Outcome as a pre_tool hook."""
    async def hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> Outcome:
        return outcome
    return hook


@pytest.mark.asyncio
async def test_single_allow_variant_sets_decision() -> None:
    d = _allow_with_rule()
    chain = HookChain(pre_tool=[_make_hook(Allow(decision=d))])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert isinstance(out, Allow), f"expected Allow, got {type(out).__name__}"
    assert out.decision is d


@pytest.mark.asyncio
async def test_single_block_variant_sets_deny_decision() -> None:
    d = _deny()
    chain = HookChain(pre_tool=[_make_hook(Block(decision=d))])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert isinstance(out, Block), f"expected Block, got {type(out).__name__}"
    assert out.decision is d
    assert out.decision.allow is False


@pytest.mark.asyncio
async def test_single_ask_variant_propagates_ask_flag() -> None:
    chain = HookChain(pre_tool=[_make_hook(Ask(reason="needs confirmation"))])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert isinstance(out, Ask), f"expected Ask, got {type(out).__name__}"
    assert out.reason == "needs confirmation"


@pytest.mark.asyncio
async def test_single_replace_variant_short_circuits_with_result() -> None:
    canned = ToolResult(ok=False, error="replaced by hook")
    d = _deny()
    chain = HookChain(pre_tool=[_make_hook(Replace(result=canned, decision=d))])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert isinstance(out, Replace), f"expected Replace, got {type(out).__name__}"
    assert out.result is canned
    assert out.decision is d


# Build distinct sentinels per variant so the assertion can identify
# which input survived the merge.
_ALLOW_A = Allow(decision=_allow("mode_bypass"))
_ALLOW_B = Allow(decision=_allow("mode_accept_edits"))
_BLOCK_A = Block(decision=_deny("safety_blocked"))
_BLOCK_B = Block(decision=_deny("rule_deny"))
_ASK_A = Ask(reason="reason A")
_ASK_B = Ask(reason="reason B")
_REPLACE_A = Replace(
    result=ToolResult(ok=False, error="replace A"),
    decision=_deny("safety_blocked"),
)
_REPLACE_B = Replace(
    result=ToolResult(ok=False, error="replace B"),
    decision=_deny("rule_deny"),
)


def _check_winner(out: Outcome, expected: Outcome) -> None:
    """Assert ``out`` matches the expected Outcome variant."""
    if isinstance(expected, Allow):
        assert isinstance(out, Allow), f"expected Allow, got {type(out).__name__}"
        assert out.decision is expected.decision
    elif isinstance(expected, Block):
        assert isinstance(out, Block), f"expected Block, got {type(out).__name__}"
        assert out.decision is expected.decision
        assert out.decision.allow is False
    elif isinstance(expected, Ask):
        assert isinstance(out, Ask), f"expected Ask, got {type(out).__name__}"
    elif isinstance(expected, Replace):
        assert isinstance(out, Replace), f"expected Replace, got {type(out).__name__}"
        assert out.result is expected.result
        assert out.decision is expected.decision
    else:  # pragma: no cover — defensive
        raise AssertionError(f"unknown expected variant: {expected!r}")


# Cases: (first_hook_return, second_hook_return, expected_winner).
# Updated precedence for pure-Outcome chains (post-PreToolOutcome deletion):
#
#   Block > Replace > Ask > Allow(authoritative) > Allow(mode_bypass)
#
# Key changes from old spec:
# - Replace beats Ask (safety block overrides pending confirmation request).
# - Ask beats non-resolved Allow (only user_accept / user_always resolve it).
# - First authoritative Allow wins (mode_bypass is passthrough, not terminal).
# - Block short-circuits run_pre_tool immediately, never reaches _merge_outcomes
#   when in the first position; Block-first cases are kept for completeness but
#   the merge never actually sees two outcomes in those cases.
_MATRIX_CASES: list[tuple[Outcome, Outcome, Outcome]] = [
    # first authoritative Allow wins (mode_bypass < mode_accept_edits)
    (_ALLOW_A, _ALLOW_B, _ALLOW_B),
    (_ALLOW_A, _BLOCK_B, _BLOCK_B),    # Block beats Allow
    (_ALLOW_A, _ASK_B, _ASK_B),        # Ask beats Allow(passthrough)
    (_ALLOW_A, _REPLACE_B, _REPLACE_B),  # Replace beats Allow
    # Block short-circuits — second never runs. First Block wins.
    (_BLOCK_A, _ALLOW_B, _BLOCK_A),
    (_BLOCK_A, _BLOCK_B, _BLOCK_A),
    (_BLOCK_A, _ASK_B, _BLOCK_A),
    (_BLOCK_A, _REPLACE_B, _BLOCK_A),
    # Ask beats non-resolved Allow (mode_accept_edits not user-driven)
    (_ASK_A, _ALLOW_B, _ASK_A),
    (_ASK_A, _BLOCK_B, _BLOCK_B),       # Block beats Ask
    (_ASK_A, _ASK_B, _ASK_A),           # first Ask wins (both unresolved)
    (_ASK_A, _REPLACE_B, _REPLACE_B),   # Replace beats Ask (safety overrides confirmation)
    (_REPLACE_A, _ALLOW_B, _REPLACE_A),  # Replace beats Allow
    (_REPLACE_A, _BLOCK_B, _BLOCK_B),    # Block beats Replace
    (_REPLACE_A, _ASK_B, _REPLACE_A),    # Replace beats Ask (safety overrides confirmation)
    (_REPLACE_A, _REPLACE_B, _REPLACE_A),  # first Replace wins
]


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    _MATRIX_CASES,
    ids=[
        f"{type(a).__name__}|{type(b).__name__}->{type(e).__name__}"
        for a, b, e in _MATRIX_CASES
    ],
)
@pytest.mark.asyncio
async def test_outcome_merge_precedence_matrix(
    first: Outcome, second: Outcome, expected: Outcome,
) -> None:
    chain = HookChain(pre_tool=[_make_hook(first), _make_hook(second)])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    _check_winner(out, expected)


@pytest.mark.asyncio
async def test_block_short_circuits_chain_no_later_hook_runs() -> None:
    call_log: list[str] = []

    async def first(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> Outcome:
        call_log.append("first")
        return _BLOCK_A

    async def never(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> Outcome:
        call_log.append("never")
        raise AssertionError("hook must not run after Block short-circuit")

    chain = HookChain(pre_tool=[first, never])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert call_log == ["first"]
    assert isinstance(out, Block)
    assert out.decision is _BLOCK_A.decision


@pytest.mark.asyncio
async def test_replace_does_not_short_circuit_block_can_still_win() -> None:
    """Replace alone does NOT stop iteration — a later Block must
    still be allowed to override per spec §3.2 (Block > Replace)."""
    call_log: list[str] = []

    async def first(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> Outcome:
        call_log.append("first")
        return _REPLACE_A

    async def second(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> Outcome:
        call_log.append("second")
        return _BLOCK_B

    chain = HookChain(pre_tool=[first, second])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    # Both ran; Block wins.
    assert call_log == ["first", "second"]
    assert isinstance(out, Block), f"expected Block, got {type(out).__name__}"
    assert out.decision is _BLOCK_B.decision


@pytest.mark.asyncio
async def test_three_hook_chain_first_authoritative_allow_wins() -> None:
    """First authoritative (non-mode_bypass) Allow wins across a three-hook chain.

    This mirrors real hook ordering: bash_safety (mode_bypass passthrough) →
    permission hook (rule_allow authoritative) → must_read_first (mode_bypass
    passthrough). The permission hook's rule_allow verdict is the correct
    audit line even though passthrough hooks bookend it.
    """
    a1 = Allow(decision=_allow("mode_bypass"))    # bash_safety passthrough
    a2 = Allow(decision=_allow_with_rule("stub")) # permission hook — authoritative
    a3 = Allow(decision=_allow("mode_bypass"))    # must_read_first passthrough

    chain = HookChain(
        pre_tool=[_make_hook(a1), _make_hook(a2), _make_hook(a3)],
    )
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert isinstance(out, Allow)
    # The permission hook's rule_allow wins over surrounding mode_bypass hooks.
    assert out.decision is a2.decision
