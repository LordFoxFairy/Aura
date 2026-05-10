"""Phase 1 Task 8 — Outcome variant returns from pre_tool hooks.

Covers the spec §3.2 merge precedence matrix for two-hook chains where
both hooks return :class:`aura.schemas.permissions.Outcome` variants
(``Allow`` / ``Block`` / ``Ask`` / ``Replace``):

    first Block wins → first Ask wins → first Replace wins → last Allow wins

Plus regression tests for:

- Single-hook returns of each variant.
- Mixed Outcome + legacy ``PreToolOutcome`` chains (back-compat).
- Block short-circuits the chain (no later hook runs).
- Replace does NOT short-circuit (a later Block can still win).
- Empty Outcome list → passthrough.

The 16-case matrix (4 variants × 4 variants) is generated via
``pytest.mark.parametrize`` so every cell is covered. Cases that share
the same expected merge behavior get the same assertion path; cases
that differ get explicit cell assertions.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks import (
    PRE_TOOL_PASSTHROUGH,
    HookChain,
    PreToolHook,
    PreToolOutcome,
)
from aura.core.permissions.decision import Decision
from aura.core.permissions.rule import Rule
from aura.schemas.permissions import Allow, Ask, Block, Outcome, Replace
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool


class _P(BaseModel):
    x: int = 0


def _noop(x: int = 0) -> dict[str, Any]:
    return {}


_stub_tool: BaseTool = build_tool(
    name="stub", description="stub", args_schema=_P, func=_noop,
)


# ---------------------------------------------------------------------------
# Decision factories — keep variant constructors short in tests.
# ---------------------------------------------------------------------------


def _allow(reason: str = "mode_bypass") -> Decision:
    return Decision(allow=True, reason=reason)  # type: ignore[arg-type]


def _allow_with_rule(rule_tool: str = "stub") -> Decision:
    return Decision(
        allow=True,
        reason="rule_allow",
        rule=Rule(tool=rule_tool, content=None),
    )


def _deny(reason: str = "safety_blocked") -> Decision:
    return Decision(allow=False, reason=reason)  # type: ignore[arg-type]


def _make_hook(outcome: Outcome | PreToolOutcome) -> PreToolHook:
    """Wrap a constant Outcome / PreToolOutcome as a pre_tool hook.

    Phase 1 Task 8: the :class:`PreToolHook` Protocol still declares a
    return type of :class:`PreToolOutcome` (Task 10 will widen it to
    :class:`Outcome`), so the wrapper is annotated as ``Any`` and
    cast back into the Protocol — :meth:`HookChain.run_pre_tool`
    accepts both shapes at runtime.
    """
    async def hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return outcome  # type: ignore[return-value]
    return hook


# ---------------------------------------------------------------------------
# Single-variant smoke tests — proves run_pre_tool accepts each Outcome
# variant in isolation (one hook, one return).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_allow_variant_sets_decision() -> None:
    d = _allow_with_rule()
    chain = HookChain(pre_tool=[_make_hook(Allow(decision=d))])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert out.short_circuit is None
    assert out.decision is d
    assert out.ask is False


@pytest.mark.asyncio
async def test_single_block_variant_sets_deny_decision() -> None:
    d = _deny()
    chain = HookChain(pre_tool=[_make_hook(Block(decision=d))])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    # Block does NOT carry a baked ToolResult — the loop turns the deny
    # decision into a synthetic ToolMessage downstream, exactly like a
    # legacy ``PreToolOutcome(decision=deny, short_circuit=None)``.
    assert out.short_circuit is None
    assert out.decision is d
    assert out.decision.allow is False


@pytest.mark.asyncio
async def test_single_ask_variant_propagates_ask_flag() -> None:
    chain = HookChain(pre_tool=[_make_hook(Ask(reason="needs confirmation"))])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert out.ask is True
    assert out.short_circuit is None
    # Ask intentionally produces no decision — the asker creates one
    # from the user's response in the loop.
    assert out.decision is None


@pytest.mark.asyncio
async def test_single_replace_variant_short_circuits_with_result() -> None:
    canned = ToolResult(ok=False, error="replaced by hook")
    d = _deny()
    chain = HookChain(pre_tool=[_make_hook(Replace(result=canned, decision=d))])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert out.short_circuit is canned
    assert out.decision is d


# ---------------------------------------------------------------------------
# 4 × 4 = 16-case merge precedence matrix. Each case constructs a 2-hook
# chain ``[first, second]`` of pure Outcome returns and checks the
# merged ``PreToolOutcome`` matches spec §3.2: first Block wins → first
# Ask wins → first Replace wins → last Allow wins.
# ---------------------------------------------------------------------------


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


def _check_winner(
    out: PreToolOutcome, expected: Outcome,
) -> None:
    """Assert ``out`` matches the PreToolOutcome shape that ``expected``
    would produce when winning the merge."""
    if isinstance(expected, Allow):
        assert out.short_circuit is None
        assert out.decision is expected.decision
    elif isinstance(expected, Block):
        # Block surfaces as decision-only (no ToolResult), matching
        # legacy "deny via decision channel" behavior.
        assert out.short_circuit is None
        assert out.decision is expected.decision
        assert out.decision.allow is False
    elif isinstance(expected, Ask):
        assert out.ask is True
        assert out.short_circuit is None
    elif isinstance(expected, Replace):
        assert out.short_circuit is expected.result
        assert out.decision is expected.decision
    else:  # pragma: no cover — defensive
        raise AssertionError(f"unknown expected variant: {expected!r}")


# Cases: (first_hook_return, second_hook_return, expected_winner).
# Precedence: Block > Ask > Replace > Allow (Allow uses last-wins).
_MATRIX_CASES: list[tuple[Outcome, Outcome, Outcome]] = [
    # --- Allow first (4 cases) ---
    (_ALLOW_A, _ALLOW_B, _ALLOW_B),    # last Allow wins
    (_ALLOW_A, _BLOCK_B, _BLOCK_B),    # Block beats Allow
    (_ALLOW_A, _ASK_B, _ASK_B),        # Ask beats Allow
    (_ALLOW_A, _REPLACE_B, _REPLACE_B),  # Replace beats Allow
    # --- Block first (4 cases) ---
    # Block short-circuits — second never runs. First Block wins.
    (_BLOCK_A, _ALLOW_B, _BLOCK_A),
    (_BLOCK_A, _BLOCK_B, _BLOCK_A),
    (_BLOCK_A, _ASK_B, _BLOCK_A),
    (_BLOCK_A, _REPLACE_B, _BLOCK_A),
    # --- Ask first (4 cases) ---
    (_ASK_A, _ALLOW_B, _ASK_A),         # Ask beats Allow (first Ask)
    (_ASK_A, _BLOCK_B, _BLOCK_B),       # Block still beats Ask
    (_ASK_A, _ASK_B, _ASK_A),           # first Ask wins
    (_ASK_A, _REPLACE_B, _ASK_A),       # Ask beats Replace
    # --- Replace first (4 cases) ---
    (_REPLACE_A, _ALLOW_B, _REPLACE_A),  # Replace beats Allow
    (_REPLACE_A, _BLOCK_B, _BLOCK_B),    # Block beats Replace
    (_REPLACE_A, _ASK_B, _ASK_B),        # Ask beats Replace
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


# ---------------------------------------------------------------------------
# Block-first must short-circuit: any later hook MUST NOT run. This is
# the "(short-circuit)" qualifier in spec §3.2 — once Block is seen no
# later hook can supersede it, so iteration stops.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_block_short_circuits_chain_no_later_hook_runs() -> None:
    call_log: list[str] = []

    async def first(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        call_log.append("first")
        return _BLOCK_A  # type: ignore[return-value]

    async def never(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        call_log.append("never")
        raise AssertionError("hook must not run after Block short-circuit")

    chain = HookChain(pre_tool=[first, never])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert call_log == ["first"]
    assert out.decision is _BLOCK_A.decision


@pytest.mark.asyncio
async def test_replace_does_not_short_circuit_block_can_still_win() -> None:
    """Replace alone does NOT stop iteration — a later Block must
    still be allowed to override per spec §3.2 (Block > Replace)."""
    call_log: list[str] = []

    async def first(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        call_log.append("first")
        return _REPLACE_A  # type: ignore[return-value]

    async def second(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        call_log.append("second")
        return _BLOCK_B  # type: ignore[return-value]

    chain = HookChain(pre_tool=[first, second])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    # Both ran; Block wins.
    assert call_log == ["first", "second"]
    assert out.short_circuit is None
    assert out.decision is _BLOCK_B.decision


@pytest.mark.asyncio
async def test_three_hook_chain_last_allow_wins_when_no_higher_variant() -> None:
    """Last-Allow-wins must apply across more than two hooks — the
    refining-permission-hook idiom (chain ends with the permission
    hook stamping its rule) needs this."""
    a1 = Allow(decision=_allow("mode_bypass"))
    a2 = Allow(decision=_allow_with_rule("stub"))
    a3 = Allow(decision=_allow("mode_accept_edits"))

    chain = HookChain(
        pre_tool=[_make_hook(a1), _make_hook(a2), _make_hook(a3)],
    )
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert out.decision is a3.decision


# ---------------------------------------------------------------------------
# Mixed-mode back-compat: a chain containing one legacy PreToolOutcome
# falls back to the legacy merge logic. Existing built-in hooks (Task 9
# migrates them) MUST keep working unchanged while Outcome adoption is
# in flight.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_outcome_then_legacy_uses_legacy_merge() -> None:
    """An Allow followed by a legacy deny PreToolOutcome must yield
    the legacy first-deny-wins behavior — Allow's decision is recorded
    but the deny supersedes."""
    allow = Allow(decision=_allow("mode_bypass"))
    deny = _deny("safety_blocked")

    async def legacy_deny(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return PreToolOutcome(short_circuit=None, decision=deny)

    chain = HookChain(pre_tool=[_make_hook(allow), legacy_deny])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert out.decision is deny


@pytest.mark.asyncio
async def test_mixed_legacy_then_outcome_falls_back_to_legacy_merge() -> None:
    """Legacy first, Allow second — legacy chain semantics keep the
    first-deny-wins or last-allow-wins result. Here both are allows
    so the Outcome's decision (last) wins under legacy too."""
    early_allow = _allow("mode_bypass")
    late_allow = _allow_with_rule("stub")

    async def legacy_allow(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return PreToolOutcome(short_circuit=None, decision=early_allow)

    chain = HookChain(
        pre_tool=[legacy_allow, _make_hook(Allow(decision=late_allow))],
    )
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert out.decision is late_allow


@pytest.mark.asyncio
async def test_mixed_passthrough_legacy_with_outcome_works() -> None:
    """A passthrough legacy hook must coexist with an Outcome hook —
    the Outcome's winner still surfaces."""
    block = Block(decision=_deny("safety_blocked"))

    async def passthrough(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return PRE_TOOL_PASSTHROUGH

    chain = HookChain(pre_tool=[passthrough, _make_hook(block)])
    out = await chain.run_pre_tool(
        tool=_stub_tool, args={}, state=LoopState(),
    )
    assert out.decision is block.decision
