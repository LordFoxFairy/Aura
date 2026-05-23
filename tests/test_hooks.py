"""Tests for aura.application.hooks.HookChain."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.application.hooks import HookChain
from aura.application.permission.decision import Decision
from aura.domain.permission.rule import Rule
from aura.schemas.permissions import Allow, Ask, Block, Outcome, Replace
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


def _allow(reason: str = "mode_bypass") -> Decision:
    return Decision(allow=True, reason=reason)  # type: ignore[arg-type]


def _allow_with_rule(tool: str) -> Decision:
    return Decision(allow=True, reason="rule_allow", rule=Rule(tool=tool, content=None))


def _deny(reason: str = "safety_blocked") -> Decision:
    return Decision(allow=False, reason=reason)  # type: ignore[arg-type]


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
    # Empty chain returns a neutral Allow sentinel.
    assert isinstance(outcome, Allow)
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
async def test_pre_tool_replace_carries_result() -> None:
    """A Replace hook carries a synthetic ToolResult."""
    denied = ToolResult(ok=False, error="denied")

    async def replace_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Replace(result=denied, decision=_deny())

    chain = HookChain(pre_tool=[replace_hook])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert isinstance(outcome, Replace)
    assert outcome.result is denied


@pytest.mark.asyncio
async def test_pre_tool_first_block_wins() -> None:
    """First Block in the chain wins; subsequent hooks are NOT called."""
    call_log: list[str] = []

    async def first(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        call_log.append("first")
        return Block(decision=_deny("safety_blocked"))

    async def second(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        call_log.append("second")
        return Block(decision=_deny("rule_deny"))

    chain = HookChain(pre_tool=[first, second])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert isinstance(outcome, Block)
    assert outcome.decision.reason == "safety_blocked"
    assert call_log == ["first"]


@pytest.mark.asyncio
async def test_pre_tool_decision_first_authoritative_allow_wins() -> None:
    """Among ALLOW decisions, the first authoritative (non-mode_bypass) reason
    wins. Passthrough hooks return mode_bypass to signal "no opinion"; the
    first hook with a real verdict (rule_allow, user_accept, …) is the
    authoritative audit line.

    Hook order: early(rule_allow) → late(mode_bypass).
    early's rule_allow is authoritative; late's mode_bypass is passthrough.
    Merged outcome = early's decision (rule_allow).
    """
    first_decision = _allow_with_rule("stub")
    passthrough_decision = _allow("mode_bypass")

    async def early(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=first_decision)

    async def late(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=passthrough_decision)

    chain = HookChain(pre_tool=[early, late])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert isinstance(outcome, Allow)
    assert outcome.decision is first_decision


@pytest.mark.asyncio
async def test_pre_tool_decision_passthrough_only_allows_use_last() -> None:
    """When every Allow is mode_bypass (all passthrough, no authoritative
    decision), the last mode_bypass Allow is the merged result."""
    first_passthrough = _allow("mode_bypass")
    last_passthrough = _allow("mode_bypass")

    async def early(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=first_passthrough)

    async def late(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=last_passthrough)

    chain = HookChain(pre_tool=[early, late])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert isinstance(outcome, Allow)
    assert outcome.decision is last_passthrough


@pytest.mark.asyncio
async def test_pre_tool_first_block_beats_later_allow() -> None:
    """BUG-AUDIT-B1 regression — a Block must not be silently overridden
    by a later Allow. First Block wins over any later Allow."""
    block_decision = _deny("safety_blocked")
    allow_decision = _allow("mode_bypass")

    async def block_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Block(decision=block_decision)

    async def allow_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=allow_decision)

    chain = HookChain(pre_tool=[block_hook, allow_hook])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    # Block short-circuits; allow_hook never runs (Block is first-wins).
    assert isinstance(outcome, Block)
    assert outcome.decision is block_decision


@pytest.mark.asyncio
async def test_pre_tool_block_beats_prior_allow() -> None:
    """Allow first, Block second — Block still wins (first Block in
    registration order, not first in execution order)."""
    allow_first = _allow_with_rule("stub")
    block_second = _deny("safety_blocked")

    async def allow_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=allow_first)

    async def block_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Block(decision=block_second)

    chain = HookChain(pre_tool=[allow_hook, block_hook])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert isinstance(outcome, Block)
    assert outcome.decision is block_second


@pytest.mark.asyncio
async def test_pre_tool_per_hook_decision_journaled(tmp_path: Any) -> None:
    """Every hook carrying a decision emits a pre_tool_hook_decision
    journal event — audit readers can reconstruct the full chain."""
    import json

    from aura.infrastructure.persistence import journal

    deny = _deny("safety_blocked")
    allow = _allow("mode_bypass")

    async def deny_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Block(decision=deny)

    async def allow_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=allow)

    # allow_hook never runs because Block short-circuits — so only one
    # audit event fires. Test with Block first to verify short-circuit.
    chain = HookChain(pre_tool=[deny_hook, allow_hook])
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)
    try:
        await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())
    finally:
        journal.reset()

    events = [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]
    decisions = [e for e in events if e["event"] == "pre_tool_hook_decision"]
    assert len(decisions) == 1
    assert decisions[0]["allow"] is False
    assert decisions[0]["reason"] == "safety_blocked"
    assert "deny_hook" in decisions[0]["hook"]


@pytest.mark.asyncio
async def test_pre_tool_per_hook_decision_journaled_for_block_replace(
    tmp_path: Any,
) -> None:
    """Block and Replace outcomes produce pre_tool_hook_decision journal events;
    Allow (implicit passthrough) is NOT journaled — only non-trivial verdicts
    need an explicit audit trail entry.

    Chain: Allow (passthrough) → Replace → Allow (passthrough).
    Block short-circuits immediately, so this verifies Replace + later Allow.
    Expect exactly 1 journal event (for the Replace).
    """
    import json

    from aura.infrastructure.persistence import journal
    from aura.schemas.tool import ToolResult

    sc = ToolResult(ok=False, error="canned")

    async def passthrough_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=_allow("mode_bypass"))

    async def replace_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Replace(result=sc, decision=_deny("safety_blocked"))

    async def trailing_allow(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=_allow("mode_bypass"))

    chain = HookChain(pre_tool=[passthrough_hook, replace_hook, trailing_allow])
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)
    try:
        await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())
    finally:
        journal.reset()

    events = [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]
    decisions = [e for e in events if e["event"] == "pre_tool_hook_decision"]
    # Only Replace produced a journal entry; both Allow hooks are silent.
    assert len(decisions) == 1, f"expected 1 (Replace only), got {decisions!r}"
    assert "replace_hook" in decisions[0]["hook"]
    assert decisions[0]["allow"] is False
    assert decisions[0]["reason"] == "safety_blocked"


@pytest.mark.asyncio
async def test_pre_tool_allow_outcomes_not_journaled(tmp_path: Any) -> None:
    """Allow is the implicit passthrough default — it is NOT journaled.
    Journaling only non-trivial outcomes (Block/Ask/Replace) keeps the audit
    log focused on decisions that actually constrain execution.
    """
    import json

    from aura.infrastructure.persistence import journal

    async def allow_hook_1(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=_allow_with_rule("stub"))

    async def allow_hook_2(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=_allow("mode_bypass"))

    chain = HookChain(pre_tool=[allow_hook_1, allow_hook_2])
    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)
    try:
        await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())
    finally:
        journal.reset()

    # Journal file may not exist if no events were written.
    if log_path.exists():
        events = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line
        ]
        decisions = [e for e in events if e["event"] == "pre_tool_hook_decision"]
        assert decisions == [], f"Allow hooks must not journal; got {decisions!r}"


@pytest.mark.asyncio
async def test_pre_tool_replace_then_allow_replace_wins() -> None:
    """Replace beats Allow — a Replace hook followed by Allow yields Replace."""
    sc = ToolResult(ok=False, error="canned")

    async def replace_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Replace(result=sc, decision=_deny())

    async def allow_hook(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=_allow())

    chain = HookChain(pre_tool=[replace_hook, allow_hook])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert isinstance(outcome, Replace)
    assert outcome.result is sc


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
async def test_pre_tool_chain_authoritative_allow_beats_passthrough() -> None:
    """First authoritative (non-mode_bypass) Allow wins over passthrough Allow.

    hook1 returns rule_allow (authoritative); hook2 returns mode_bypass (passthrough).
    Expected winner: hook1's rule_allow decision.
    """
    d1 = _allow_with_rule("stub")   # authoritative
    d2 = _allow("mode_bypass")      # passthrough

    async def hook1(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=d1)

    async def hook2(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=d2)

    chain = HookChain(pre_tool=[hook1, hook2])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert isinstance(outcome, Allow)
    assert outcome.decision is d1


@pytest.mark.asyncio
async def test_pre_tool_chain_all_passthrough_last_wins() -> None:
    """When all Allow hooks are passthrough (mode_bypass), last-wins applies."""
    d1 = _allow("mode_bypass")
    d2 = _allow("mode_bypass")

    async def hook1(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=d1)

    async def hook2(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object
    ) -> Outcome:
        return Allow(decision=d2)

    chain = HookChain(pre_tool=[hook1, hook2])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())

    assert isinstance(outcome, Allow)
    assert outcome.decision is d2


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
    ) -> Outcome:
        return Allow(decision=Decision(allow=True, reason="mode_bypass"))

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
# Ask escalation channel — Ask propagates via the chain-local ``ask_pending``
# kwarg so downstream hooks (permission) see the escalation demand without
# any cross-turn state.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_tool_ask_propagates_as_outcome() -> None:
    """A single Ask hook returns Ask as the merged outcome."""
    async def asker(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> Outcome:
        return Ask(reason="needs confirmation")

    chain = HookChain(pre_tool=[asker])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())
    assert isinstance(outcome, Ask)


@pytest.mark.asyncio
async def test_pre_tool_ask_seen_by_downstream_hook_via_kwarg() -> None:
    """When an upstream hook returns Ask, downstream hooks receive
    ``ask_pending=True`` via kwargs so a permission hook later in the
    chain can detect the demand and demote any auto-allow to the asker path.
    """
    seen: list[bool] = []

    async def upstream(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> Outcome:
        return Ask(reason="needs confirmation")

    async def downstream(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        ask_pending: bool = False,
        **_: object,
    ) -> Outcome:
        seen.append(ask_pending)
        return Allow(decision=Decision(allow=True, reason="mode_bypass"))

    chain = HookChain(pre_tool=[upstream, downstream])
    await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())
    assert seen == [True]


@pytest.mark.asyncio
async def test_pre_tool_ask_kwarg_defaults_false_for_first_hook() -> None:
    """First hook in the chain sees ``ask_pending=False`` because no
    upstream hook has run yet."""
    seen: list[bool] = []

    async def hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        ask_pending: bool = False,
        **_: object,
    ) -> Outcome:
        seen.append(ask_pending)
        return Allow(decision=Decision(allow=True, reason="mode_bypass"))

    chain = HookChain(pre_tool=[hook])
    out = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())
    assert isinstance(out, Allow)
    assert seen == [False]


@pytest.mark.asyncio
async def test_pre_tool_block_beats_ask() -> None:
    """Block > Ask precedence. A Block hook beats a prior Ask."""
    deny = _deny("safety_blocked")

    async def asker(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> Outcome:
        return Ask(reason="needs confirmation")

    async def blocker(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> Outcome:
        return Block(decision=deny)

    chain = HookChain(pre_tool=[asker, blocker])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())
    # Block short-circuits as soon as it is seen.
    assert isinstance(outcome, Block)
    assert outcome.decision is deny


@pytest.mark.asyncio
async def test_pre_tool_ask_beats_allow() -> None:
    """Ask > Allow. A subsequent Ask overrides a prior Allow."""
    async def allower(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> Outcome:
        return Allow(decision=_allow("mode_bypass"))

    async def asker(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> Outcome:
        return Ask(reason="needs confirmation")

    chain = HookChain(pre_tool=[allower, asker])
    outcome = await chain.run_pre_tool(tool=_stub_tool, args={}, state=LoopState())
    assert isinstance(outcome, Ask)
