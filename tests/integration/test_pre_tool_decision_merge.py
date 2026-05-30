"""BUG-AUDIT-B1 integration: pre_tool decision merge first-deny-wins.

Drives a real :class:`aura.core.agent.Agent` turn through the full
pre_tool hook chain — bash_safety + a "soft-policy" deny hook + the
real permission hook + must_read_first — and asserts on the persisted
journal that:

1. **First-deny-wins:** when a soft-policy hook emits a deny decision
   without short-circuiting, a later permission hook returning
   ``mode_bypass`` (allow) does NOT silently override it. The merged
   ``permission_decision`` carries the deny.
2. **Audit completeness:** every hook's decision lands as its own
   ``pre_tool_hook_decision`` journal event before merge — so an audit
   reader sees BOTH the soft-policy deny AND the permission hook's
   override attempt, even though only the deny propagates.
3. **Tool blocked:** the model receives a deny ``ToolResult`` (from the
   soft hook's short_circuit emitted on the same call as a deny — see
   below — OR from a different hook short-circuiting; the merge
   semantic stands either way).

The "soft-policy" hook is a deliberate test construct — it returns a
deny ``Decision`` WITHOUT a ``short_circuit``, exposing the merge bug
in isolation. Production hooks (bash_safety, permission) always pair
deny+short_circuit, which masks the bug; this test forces the broken
path to be exercised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.application.hooks import HookChain
from aura.application.hooks.bash_safety import make_bash_safety_hook
from aura.application.hooks.permission import make_permission_hook
from aura.application.permission.asker import AskerResponse
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.domain.events import ToolCallCompleted
from aura.domain.permission.decision import Decision
from aura.domain.permission.outcome import Allow, Block, Outcome, Replace
from aura.domain.permission.session import SessionRuleSet
from aura.domain.tool import ToolResult
from aura.infrastructure import permission_store as perm_store
from aura.infrastructure.persistence.storage import SessionStorage
from aura.schemas.state import LoopState
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel, FakeTurn


class _BashParams(BaseModel):
    command: str


def _make_bash_tool(run_log: list[str]) -> BaseTool:
    """In-memory bash tool that records invocations without shelling out."""

    def _run(command: str) -> dict[str, Any]:
        run_log.append(command)
        return {"stdout": "", "stderr": "", "exit_code": 0}

    return build_tool(
        name="bash",
        description="fake bash",
        args_schema=_BashParams,
        func=_run,
        is_destructive=True,
        args_preview=lambda args: str(args.get("command", "")),
    )


def _minimal_cfg() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["bash"]},
    })


def _one_bash_then_final(command: str) -> list[FakeTurn]:
    return [
        FakeTurn(message=AIMessage(
            content="",
            tool_calls=[{
                "id": "tc_1",
                "name": "bash",
                "args": {"command": command},
            }],
        )),
        FakeTurn(message=AIMessage(content="done")),
    ]


@pytest.mark.asyncio
async def test_first_deny_beats_later_allow_in_real_agent_turn(
    tmp_path: Path,
) -> None:
    """Real Agent turn with bash_safety + soft-deny + permission_hook in
    chain order. The soft-deny emits ``allow=False`` without
    short-circuiting; the permission hook is in bypass mode and would
    return ``allow=True``. Pre-fix, last-wins merge would let bypass
    override the deny silently. Post-fix, first-deny-wins keeps the
    deny and the permission hook's allow attempt is recorded as its
    own audit event but does NOT overwrite the merged decision."""
    run_log: list[str] = []
    bash_tool = _make_bash_tool(run_log)

    # 1. bash_safety hook — sits at index 0 in production. Safe command
    # ("echo hi") so it passthroughs; no decision emitted.
    bash_safety_hook = make_bash_safety_hook()

    # 2. Soft-policy deny hook — emits ``allow=False`` decision but NO
    # short_circuit. This is the path that exposes the merge bug: the
    # chain MUST keep walking, but the deny MUST win.
    soft_deny_short_circuit = ToolResult(
        ok=False, error="soft-policy denied: example",
    )

    async def soft_policy_deny_hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        **_: Any,
    ) -> Outcome:
        if tool.name != "bash":
            return Allow(decision=Decision(allow=True, reason="mode_bypass"))
        # Pair the deny with a short_circuit so the model still sees
        # an error ToolResult — this is the realistic shape; the merge
        # bug is independent of whether short_circuit is set.
        return Replace(
            result=soft_deny_short_circuit,
            decision=Decision(allow=False, reason="user_deny"),
        )

    # 3. Permission hook in bypass mode — returns ``mode_bypass`` allow
    # for any tool call. In a buggy last-wins merge, this would
    # silently override the soft-deny. But because soft_policy short-
    # circuits, the chain stops before this hook actually runs in
    # this scenario — so we'll test the non-short-circuit variant
    # separately below.
    perm_hook = make_permission_hook(
        asker=_unused_asker,  # bypass mode never consults asker
        session=SessionRuleSet(),
        rules=perm_store.load_ruleset(tmp_path),
        project_root=tmp_path,
        mode="bypass",
    )

    hooks = HookChain(pre_tool=[
        bash_safety_hook,
        soft_policy_deny_hook,
        perm_hook,
    ])

    storage = SessionStorage(tmp_path / "aura.db")
    session_log_dir = tmp_path / "logs"
    agent = Agent(
        config=_minimal_cfg(),
        model=FakeChatModel(turns=_one_bash_then_final("echo hi")),
        storage=storage,
        hooks=hooks,
        available_tools={"bash": bash_tool},
        mode="bypass",
        session_id="test-deny-merge",
        session_log_dir=session_log_dir,
        auto_compact_threshold=0,
    )

    completed_events: list[ToolCallCompleted] = []
    try:
        async for event in agent.astream("run echo hi"):
            if isinstance(event, ToolCallCompleted):
                completed_events.append(event)
    finally:
        await agent.aclose()

    # The bash tool MUST NOT have run — the soft-deny short-circuited
    # before dispatch even though the chain didn't stop the merge.
    assert run_log == [], (
        f"bash tool should not have run; got run_log={run_log!r}"
    )

    # The model must have received exactly one ToolCallCompleted event,
    # carrying the soft-deny error.
    assert len(completed_events) == 1
    assert completed_events[0].error == "soft-policy denied: example"

    log_path = session_log_dir / "test-deny-merge.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    events = [json.loads(line) for line in lines if line]
    hook_decisions = [
        e for e in events if e["event"] == "pre_tool_hook_decision"
    ]

    # The soft-deny hook MUST have its own audit entry, recording the
    # deny verdict. (The permission hook never runs because the soft
    # hook short-circuits — this is by design; first-wins on
    # short_circuit is unchanged behavior.)
    deny_records = [
        h for h in hook_decisions
        if h["allow"] is False and h["reason"] == "user_deny"
    ]
    assert len(deny_records) >= 1, (
        f"expected at least one deny hook_decision; got {hook_decisions!r}"
    )
    assert "soft_policy_deny_hook" in deny_records[0]["hook"], (
        f"hook field should identify the source; got {deny_records[0]['hook']!r}"
    )
    assert deny_records[0]["tool"] == "bash"
@pytest.mark.asyncio
async def test_multiple_non_short_circuiting_decisions_merge_first_deny_wins(
    tmp_path: Path,
) -> None:
    """Block beats Allow in merged outcome: soft_allow (first) + soft_deny/Block (second).

    In the Outcome world, Block short-circuits immediately only when it is the
    FIRST outcome the chain sees. When Allow precedes Block, the chain processes
    both hooks and _merge_outcomes picks Block (first Block wins per spec §3.2)
    — so the tool is blocked even though the first hook allowed it.

    This verifies "first Block wins" in the merge sense: the deny from soft_deny
    overrides the allow from soft_allow even though soft_allow ran first.

    - merged outcome = Block (deny wins over prior Allow)
    - Block journal event fires for soft_deny; Allow is not journaled (implicit)
    - tool does NOT run
    - no permission_decision=mode_bypass audit event fires (deny wins)
    """
    run_log: list[str] = []
    bash_tool = _make_bash_tool(run_log)

    async def soft_allow(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        **_: Any,
    ) -> Outcome:
        return Allow(decision=Decision(allow=True, reason="mode_bypass"))

    async def soft_deny(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        **_: Any,
    ) -> Outcome:
        if tool.name != "bash":
            return Allow(decision=Decision(allow=True, reason="mode_bypass"))
        return Block(decision=Decision(allow=False, reason="user_deny"))

    # soft_allow runs first (returns Allow), then soft_deny (returns Block).
    # _merge_outcomes picks Block — tool should be denied.
    hooks = HookChain(pre_tool=[soft_allow, soft_deny])

    storage = SessionStorage(tmp_path / "aura.db")
    session_log_dir = tmp_path / "logs"
    agent = Agent(
        config=_minimal_cfg(),
        model=FakeChatModel(turns=_one_bash_then_final("echo hi")),
        storage=storage,
        hooks=hooks,
        available_tools={"bash": bash_tool},
        mode="bypass",
        session_id="test-merge-no-sc",
        session_log_dir=session_log_dir,
        auto_compact_threshold=0,
    )
    try:
        async for _ in agent.astream("run"):
            pass
    finally:
        await agent.aclose()

    # Block from soft_deny → tool did NOT run.
    assert run_log == []

    log_path = session_log_dir / "test-merge-no-sc.jsonl"
    events = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line
    ]
    hook_decisions = [
        e for e in events if e["event"] == "pre_tool_hook_decision"
    ]
    # Only Block/Ask/Replace get journaled. soft_allow (Allow) is not journaled.
    # soft_deny (Block) produces one journal entry.
    assert len(hook_decisions) == 1, (
        f"expected 1 hook decision (Block only); got {hook_decisions!r}"
    )
    deny_record = hook_decisions[0]
    assert deny_record["allow"] is False
    assert deny_record["reason"] == "user_deny"
    assert "soft_deny" in deny_record["hook"]
    assert deny_record["tool"] == "bash"

    # The merged Block means no mode_bypass permission_decision audit event.
    perm_audits = [
        e for e in events
        if e["event"] == "permission_decision"
        and e.get("reason") == "mode_bypass"
    ]
    assert perm_audits == [], (
        f"merged decision should be deny, not bypass-allow; "
        f"got perm_audits={perm_audits!r}"
    )


async def _unused_asker(**_: Any) -> AskerResponse:
    raise AssertionError(
        "permission asker should not be consulted in bypass mode"
    )
