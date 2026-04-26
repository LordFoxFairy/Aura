"""Tests for aura.core.hooks.HookChain."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks import (
    PRE_TOOL_ASK_PENDING_KEY,
    PRE_TOOL_PASSTHROUGH,
    HookChain,
    PreToolOutcome,
)
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool


class _P(BaseModel):
    x: int = 0


def _noop(x: int = 0) -> dict[str, Any]:
    return {}


_stub_tool: BaseTool = build_tool(
    name="stub",
    description="stub",
    args_schema=_P,
    func=_noop,
)


@pytest.mark.asyncio
async def test_hookchain_empty_is_noop() -> None:
    chain = HookChain()
    history: list[BaseMessage] = []
    ai_msg = AIMessage(content="hi")
    args: dict[str, Any] = {}
    result = ToolResult(ok=True, output={})
    state = LoopState()

    await chain.run_pre_model(history=history, state=state)
    await chain.run_post_model(ai_message=ai_msg, history=history, state=state)
    outcome = await chain.run_pre_tool(tool=_stub_tool, args=args, state=state)
    final = await chain.run_post_tool(
        tool=_stub_tool, args=args, result=result, state=state
    )

    assert history == []
    assert outcome.short_circuit is None
    assert outcome.decision is None
    assert final is result


@pytest.mark.asyncio
async def test_pre_model_sees_history_and_can_mutate() -> None:
    async def inject(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        history.append(SystemMessage(content="injected"))

    chain = HookChain(pre_model=[inject])
    history: list[BaseMessage] = []
    await chain.run_pre_model(history=history, state=LoopState())

    assert len(history) == 1
    assert isinstance(history[0], SystemMessage)
    assert history[0].content == "injected"


@pytest.mark.asyncio
async def test_post_model_sees_ai_message() -> None:
    captured: list[AIMessage] = []

    async def capture(
        *, ai_message: AIMessage, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        captured.append(ai_message)

    chain = HookChain(post_model=[capture])
    ai_msg = AIMessage(content="test")
    await chain.run_post_model(ai_message=ai_msg, history=[], state=LoopState())

    assert len(captured) == 1
    assert captured[0] is ai_msg


@pytest.mark.asyncio
async def test_pre_tool_short_circuits_with_tool_result() -> None:
    denied = ToolResult(ok=False, error="denied")

    async def deny(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        return PreToolOutcome(short_circuit=denied, decision=None)

    chain = HookChain(pre_tool=[deny])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert outcome.short_circuit is denied


@pytest.mark.asyncio
async def test_pre_tool_first_short_circuit_wins() -> None:
    call_log: list[str] = []
    first_denial = ToolResult(ok=False, error="first")

    async def first(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        call_log.append("first")
        return PreToolOutcome(short_circuit=first_denial, decision=None)

    async def second(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        call_log.append("second")
        return PreToolOutcome(
            short_circuit=ToolResult(ok=False, error="second"), decision=None,
        )

    chain = HookChain(pre_tool=[first, second])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert outcome.short_circuit is first_denial
    assert call_log == ["first"]


@pytest.mark.asyncio
async def test_pre_tool_decision_last_allow_wins_when_no_deny() -> None:
    """Among ALLOW decisions (no hook denies), last-wins applies — the
    permission hook (typically last in the chain) gets to stamp its
    ``rule_allow`` / ``mode_bypass`` reason as the authoritative audit
    line."""
    from aura.core.permissions.decision import Decision
    from aura.core.permissions.rule import Rule

    first_decision = Decision(
        allow=True, reason="rule_allow", rule=Rule(tool="stub", content=None),
    )
    last_decision = Decision(allow=True, reason="mode_bypass")

    async def early(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        return PreToolOutcome(short_circuit=None, decision=first_decision)

    async def late(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        return PreToolOutcome(short_circuit=None, decision=last_decision)

    chain = HookChain(pre_tool=[early, late])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert outcome.decision is last_decision


@pytest.mark.asyncio
async def test_pre_tool_first_deny_beats_later_allow() -> None:
    """BUG-AUDIT-B1 regression — a deny decision must not be silently
    overridden by a later allow. Pre-fix the merge was last-wins, which
    meant a permission hook returning ``mode_bypass`` (allow) would
    erase a safety hook's earlier ``safety_blocked`` (deny) and an
    audit reader would only see the allow."""
    from aura.core.permissions.decision import Decision

    deny_first = Decision(allow=False, reason="safety_blocked")
    allow_later = Decision(allow=True, reason="mode_bypass")

    async def deny_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        # Deliberately does NOT short_circuit — exercises the merge
        # path where a later hook would otherwise silently win.
        return PreToolOutcome(short_circuit=None, decision=deny_first)

    async def allow_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        return PreToolOutcome(short_circuit=None, decision=allow_later)

    chain = HookChain(pre_tool=[deny_hook, allow_hook])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert outcome.decision is deny_first


@pytest.mark.asyncio
async def test_pre_tool_first_deny_wins_regardless_of_position() -> None:
    """First-deny-wins must hold whether the deny is the first or the
    last hook to fire. An allow that ran before the deny is replaced
    by the deny (deny supersedes prior allows); a later allow cannot
    override the deny."""
    from aura.core.permissions.decision import Decision
    from aura.core.permissions.rule import Rule

    allow_first = Decision(
        allow=True, reason="rule_allow", rule=Rule(tool="stub", content=None),
    )
    deny_second = Decision(allow=False, reason="safety_blocked")

    async def allow_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        return PreToolOutcome(short_circuit=None, decision=allow_first)

    async def deny_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        return PreToolOutcome(short_circuit=None, decision=deny_second)

    chain = HookChain(pre_tool=[allow_hook, deny_hook])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert outcome.decision is deny_second


@pytest.mark.asyncio
async def test_pre_tool_per_hook_decision_journaled(tmp_path: Any) -> None:
    """Every non-None hook decision must emit its own
    ``pre_tool_hook_decision`` journal event before the merge fires —
    so even when a later hook overrides an earlier one (or vice-versa
    under first-deny-wins), the audit trail captures BOTH verdicts."""
    import json

    from aura.core.permissions.decision import Decision
    from aura.core.persistence import journal

    deny = Decision(allow=False, reason="safety_blocked")
    allow = Decision(allow=True, reason="mode_bypass")

    async def deny_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        return PreToolOutcome(short_circuit=None, decision=deny)

    async def allow_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        return PreToolOutcome(short_circuit=None, decision=allow)

    chain = HookChain(pre_tool=[deny_hook, allow_hook])
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)
    try:
        outcome = await chain.run_pre_tool(
            tool=_stub_tool, args={}, state=LoopState(),
        )
    finally:
        journal.reset()

    assert outcome.decision is deny  # first-deny-wins

    # Replay the audit trail: BOTH hook decisions must be present.
    events = [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]
    decisions = [e for e in events if e["event"] == "pre_tool_hook_decision"]
    assert len(decisions) == 2, decisions
    assert decisions[0]["allow"] is False
    assert decisions[0]["reason"] == "safety_blocked"
    assert decisions[1]["allow"] is True
    assert decisions[1]["reason"] == "mode_bypass"
    # ``hook`` field must identify the source — closure qualname is
    # acceptable (matches the production hooks emitted via
    # make_*_hook factories).
    assert "deny_hook" in decisions[0]["hook"]
    assert "allow_hook" in decisions[1]["hook"]


@pytest.mark.asyncio
async def test_pre_tool_decision_preserved_when_short_circuit_fires() -> None:
    """When a hook short-circuits the chain, any decision collected so
    far (from earlier hooks that ran) is preserved on the outcome —
    the short-circuit does not erase decisions. A later hook's decision
    that would have been last-wins is naturally not considered because
    its hook never runs."""
    from aura.core.permissions.decision import Decision

    early_decision = Decision(allow=True, reason="mode_bypass")
    sc = ToolResult(ok=False, error="stopped by middle hook")

    async def decider(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        return PreToolOutcome(short_circuit=None, decision=early_decision)

    async def blocker(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        return PreToolOutcome(short_circuit=sc, decision=None)

    async def never_ran(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        raise AssertionError("hook must not run after a short-circuit")

    chain = HookChain(pre_tool=[decider, blocker, never_ran])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert outcome.short_circuit is sc
    assert outcome.decision is early_decision


@pytest.mark.asyncio
async def test_post_tool_chains_in_order() -> None:
    async def append_a(
        *, tool: BaseTool, args: dict[str, Any], result: ToolResult, state: LoopState,
        **_: object,
    ) -> ToolResult:
        out = list(result.output) if isinstance(result.output, list) else []
        out.append("a")
        return ToolResult(ok=True, output=out)

    async def append_b(
        *, tool: BaseTool, args: dict[str, Any], result: ToolResult, state: LoopState,
        **_: object,
    ) -> ToolResult:
        out = list(result.output) if isinstance(result.output, list) else []
        out.append("b")
        return ToolResult(ok=True, output=out)

    chain = HookChain(post_tool=[append_a, append_b])
    final = await chain.run_post_tool(
        tool=_stub_tool, args={}, result=ToolResult(ok=True, output=[]), state=LoopState()
    )

    assert final.output == ["a", "b"]


@pytest.mark.asyncio
async def test_multiple_hooks_of_same_type_run_in_registration_order() -> None:
    call_log: list[str] = []

    async def first(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        call_log.append("first")

    async def second(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        call_log.append("second")

    chain = HookChain(pre_model=[first, second])
    await chain.run_pre_model(history=[], state=LoopState())

    assert call_log == ["first", "second"]


@pytest.mark.asyncio
async def test_pre_tool_chain_passes_through_when_all_passthrough() -> None:
    async def pass_through(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> PreToolOutcome:
        return PRE_TOOL_PASSTHROUGH

    chain = HookChain(pre_tool=[pass_through, pass_through])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert outcome.short_circuit is None
    assert outcome.decision is None


@pytest.mark.asyncio
async def test_hooks_receive_state_kwarg() -> None:
    received: list[LoopState] = []

    async def capture(
        *, history: list[BaseMessage], state: LoopState, **_: object
    ) -> None:
        received.append(state)

    hooks = HookChain(pre_model=[capture])
    s = LoopState()
    await hooks.run_pre_model(history=[], state=s)

    assert received == [s]


def test_pre_model_hook_protocol_accepts_correct_signature() -> None:
    async def ok_hook(*, history: list[BaseMessage], state: LoopState, **_: object) -> None:
        return None

    chain = HookChain(pre_model=[ok_hook])
    assert len(chain.pre_model) == 1


def test_post_tool_hook_protocol_accepts_correct_signature() -> None:
    async def ok_hook(
        *, tool: BaseTool, args: dict[str, Any], result: ToolResult, state: LoopState,
        **_: object,
    ) -> ToolResult:
        return result

    chain = HookChain(post_tool=[ok_hook])
    assert len(chain.post_tool) == 1


# ---------------------------------------------------------------------------
# Turn-cycle slot defaults — regression guard that HookChain() exposes each
# slot as field(default_factory=list). A new hook type added as a module-
# level list would silently bypass this contract and break isolated merge
# semantics.
# ---------------------------------------------------------------------------


def test_hookchain_defaults_include_all_turn_cycle_slots() -> None:
    chain = HookChain()
    assert chain.pre_model == []
    assert chain.post_model == []
    assert chain.pre_tool == []
    assert chain.post_tool == []


def test_merge_concatenates_all_turn_cycle_slots() -> None:
    async def _noop(**_: object) -> None:
        return None

    async def _pre_tool(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState,
        **_: object,
    ) -> PreToolOutcome:
        return PRE_TOOL_PASSTHROUGH

    async def _post_tool(
        *, tool: BaseTool, args: dict[str, Any], result: ToolResult,
        state: LoopState, **_: object,
    ) -> ToolResult:
        return result

    left = HookChain(
        pre_model=[_noop],
        post_model=[_noop],
        pre_tool=[_pre_tool],
        post_tool=[_post_tool],
    )
    right = HookChain(
        pre_model=[_noop],
        post_model=[_noop],
        pre_tool=[_pre_tool],
        post_tool=[_post_tool],
    )
    merged = left.merge(right)
    # Every slot should carry 2 hooks after merge — no slot dropped.
    assert len(merged.pre_model) == 2
    assert len(merged.post_model) == 2
    assert len(merged.pre_tool) == 2
    assert len(merged.post_tool) == 2
    # Non-destructive — originals untouched.
    assert len(left.pre_model) == 1
    assert len(right.post_tool) == 1


# ---------------------------------------------------------------------------
# F-04-002 — PreToolOutcome.ask channel + merge precedence (deny > ask > allow)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_tool_ask_propagates_to_merged_outcome() -> None:
    async def asker(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return PreToolOutcome(ask=True)

    chain = HookChain(pre_tool=[asker])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())
    assert outcome.ask is True


@pytest.mark.asyncio
async def test_pre_tool_ask_seen_by_downstream_hook_via_state() -> None:
    """When an upstream hook sets ``ask=True``, downstream hooks see
    ``state.custom[PRE_TOOL_ASK_PENDING_KEY]`` so a permission hook
    later in the chain can detect the demand and demote any auto-allow
    to the asker path."""
    seen: list[bool] = []

    async def upstream(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return PreToolOutcome(ask=True)

    async def downstream(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        seen.append(bool(state.custom.get(PRE_TOOL_ASK_PENDING_KEY)))
        return PRE_TOOL_PASSTHROUGH

    chain = HookChain(pre_tool=[upstream, downstream])
    state = LoopState()
    await chain.run_pre_tool(tool=_stub_tool, args={}, state=state)
    assert seen == [True]
    # Sentinel must NOT leak past the chain run — next tool call should
    # see no pending ask unless re-requested.
    assert PRE_TOOL_ASK_PENDING_KEY not in state.custom


@pytest.mark.asyncio
async def test_pre_tool_ask_does_not_leak_when_no_hook_asks() -> None:
    state = LoopState()
    state.custom["unrelated"] = "x"
    chain = HookChain(pre_tool=[])
    out = await chain.run_pre_tool(tool=_stub_tool, args={}, state=state)
    assert out.ask is False
    assert PRE_TOOL_ASK_PENDING_KEY not in state.custom
    assert state.custom["unrelated"] == "x"


@pytest.mark.asyncio
async def test_pre_tool_deny_beats_ask() -> None:
    """deny > ask. Even if a hook upstream said ``ask``, a downstream
    deny is the final verdict."""
    from aura.core.permissions.decision import Decision

    deny = Decision(allow=False, reason="safety_blocked")

    async def asker(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return PreToolOutcome(ask=True)

    async def denier(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return PreToolOutcome(decision=deny)

    chain = HookChain(pre_tool=[asker, denier])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())
    # ask flag still propagates as a side-channel; the deny is the
    # authoritative decision.
    assert outcome.decision is deny
    assert outcome.ask is True


@pytest.mark.asyncio
async def test_pre_tool_ask_overrides_prior_allow() -> None:
    """ask > allow. A prior allow does not survive a subsequent ask."""
    from aura.core.permissions.decision import Decision

    allow = Decision(allow=True, reason="mode_bypass")

    async def allower(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return PreToolOutcome(decision=allow)

    async def asker(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return PreToolOutcome(ask=True)

    chain = HookChain(pre_tool=[allower, asker])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())
    assert outcome.ask is True
    # decision channel still records the prior allow, but the ask
    # signal demands the loop re-prompt — the permission hook in
    # production reads PRE_TOOL_ASK_PENDING_KEY to do exactly that.
    assert outcome.decision is allow
