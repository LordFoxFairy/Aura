"""Workstream C1 — subagent installs an auto-deny-prompt permission hook.

Parity with claude-code's Task tool: child Agents that can't show a
permission UI are constructed with a permission context that
auto-denies any call that would otherwise have to ask the user
(``shouldAvoidPermissionPrompts: true`` in the TS source,
``runAgent.ts:440-451``).

Contract for Aura:

- ``allow`` decisions from a rule / safety-passed / bypass mode still
  allow. Inherited parent ``RuleSet`` + parent ``SafetyPolicy`` ride into
  the child hook unchanged, so a parent's ``allow read_file`` rule lets
  the subagent read without prompting.
- ``safety_blocked`` / ``plan_mode_blocked`` still deny — the hook runs
  the same ``_decide`` pipeline as the parent.
- Any path that would prompt the user (``ask`` branch of ``_decide``)
  silently denies with ``reason="user_deny"`` — the child never sees a
  UI, so asking would hang the turn forever.
- A fresh :class:`SessionRuleSet` is handed to the child hook. Session
  rules approved inside the subagent (shouldn't happen under auto-deny,
  but tested for isolation) MUST NOT leak back into the parent.
- Child inherits parent's permission mode if the parent is in
  ``bypass``; otherwise child is in ``default``. ``plan`` / ``accept_edits``
  don't make sense on a non-interactive subagent and collapse to
  ``default``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.permissions.defaults import DEFAULT_ALLOW_RULES
from aura.core.permissions.rule import Rule
from aura.core.permissions.safety import DEFAULT_SAFETY
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import (
    SUBAGENT_AUTO_DENY_FEEDBACK,
    SubagentFactory,
    _SubagentPermissionAsker,
)
from aura.schemas.state import LoopState
from aura.schemas.tool import (
    ToolMetadata,
    ToolResult,  # noqa: F401
)
from tests.conftest import FakeChatModel, FakeTurn


def _sc(outcome: object) -> ToolResult | None:
    """Extract the short-circuit result from Replace Outcome."""
    from aura.schemas.permissions import Replace
    if isinstance(outcome, Replace):
        return outcome.result
    return getattr(outcome, "short_circuit", None)


# ---------------------------------------------------------------------------
# Fixture / helpers
# ---------------------------------------------------------------------------


class _EchoParams(BaseModel):
    value: str = "x"


class _EchoTool(BaseTool):
    """Tool with no built-in allow rule — forces the ask path unless a
    parent rule explicitly permits it."""

    name: str = "echo_tool"
    description: str = "test-only echo"
    args_schema: type[BaseModel] = _EchoParams
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=False,
        is_concurrency_safe=False,
        rule_matcher=None,
        args_preview=None,
        timeout_sec=None,
    )

    def _run(self, value: str = "x") -> str:
        return value


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate(
        {
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }
    )


def _build_factory(
    *,
    parent_ruleset: RuleSet | None = None,
    parent_safety: Any = DEFAULT_SAFETY,
    parent_mode: str = "default",
    parent_deny_rules: RuleSet | None = None,
    parent_ask_rules: RuleSet | None = None,
) -> SubagentFactory:
    return SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        parent_ruleset=parent_ruleset,
        parent_safety=parent_safety,
        parent_mode_provider=lambda: parent_mode,
        parent_deny_rules=parent_deny_rules,
        parent_ask_rules=parent_ask_rules,
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="done"))]
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )


# ---------------------------------------------------------------------------
# AC-C1-1 — ask path inside subagent auto-denies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_denies_tool_requiring_user_prompt() -> None:
    """Tool has no matching rule → parent would ask → subagent must deny."""
    factory = _build_factory(parent_ruleset=RuleSet())
    child = factory.spawn("prompt")
    try:
        outcome = await child._hooks.run_pre_tool(
            tool=_EchoTool(),
            args={"value": "x"},
            state=LoopState(),
        )
        assert _sc(outcome) is not None, (
            "subagent hook must short-circuit (deny) on the ask path"
        )
        assert outcome.decision is not None  # type: ignore[union-attr]
        assert outcome.decision.allow is False  # type: ignore[union-attr]
        assert outcome.decision.reason == "user_deny"  # type: ignore[union-attr]
        # Model-facing error string identifies this as a subagent auto-deny
        # via the ``user_deny`` reason (the asker's ``feedback`` field
        # carries the marker string; deny formatter appends it so the LLM
        # sees "subagent_auto_deny").
        assert _sc(outcome).ok is False  # type: ignore[union-attr]
        assert "subagent_auto_deny" in (_sc(outcome).error or "")  # type: ignore[union-attr]
    finally:
        await child.aclose()


# ---------------------------------------------------------------------------
# AC-C1-2 — parent allow rule rides into the child
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_honors_parent_allow_rule() -> None:
    """Parent ``Rule(read_file)`` → subagent allows read_file without prompt."""
    # Compose a ruleset that allows echo_tool tool-wide. Parent rules
    # flow into the child hook unchanged.
    parent_ruleset = RuleSet(
        rules=(Rule(tool="echo_tool", content=None),) + DEFAULT_ALLOW_RULES,
    )
    factory = _build_factory(parent_ruleset=parent_ruleset)
    child = factory.spawn("prompt")
    try:
        outcome = await child._hooks.run_pre_tool(
            tool=_EchoTool(),
            args={"value": "x"},
            state=LoopState(),
        )
        # Allow path: no short_circuit, decision.allow True, reason rule_allow.
        assert _sc(outcome) is None
        assert outcome.decision is not None  # type: ignore[union-attr]
        assert outcome.decision.allow is True  # type: ignore[union-attr]
        assert outcome.decision.reason == "rule_allow"  # type: ignore[union-attr]
    finally:
        await child.aclose()


@pytest.mark.asyncio
async def test_subagent_honors_parent_deny_rule_over_allow_rule() -> None:
    parent_ruleset = RuleSet(rules=(Rule(tool="echo_tool", content=None),))
    parent_deny_rules = RuleSet(rules=(Rule(tool="echo_tool", content=None),))
    factory = _build_factory(
        parent_ruleset=parent_ruleset,
        parent_deny_rules=parent_deny_rules,
    )
    child = factory.spawn("prompt")
    try:
        outcome = await child._hooks.run_pre_tool(
            tool=_EchoTool(),
            args={"value": "x"},
            state=LoopState(),
        )
        assert _sc(outcome) is not None
        assert outcome.decision is not None  # type: ignore[union-attr]
        assert outcome.decision.allow is False  # type: ignore[union-attr]
        assert outcome.decision.reason == "rule_deny"  # type: ignore[union-attr]
    finally:
        await child.aclose()


@pytest.mark.asyncio
async def test_subagent_honors_parent_ask_rule_by_auto_denying_prompt() -> None:
    parent_ruleset = RuleSet(rules=(Rule(tool="echo_tool", content=None),))
    parent_ask_rules = RuleSet(rules=(Rule(tool="echo_tool", content=None),))
    factory = _build_factory(
        parent_ruleset=parent_ruleset,
        parent_ask_rules=parent_ask_rules,
    )
    child = factory.spawn("prompt")
    try:
        outcome = await child._hooks.run_pre_tool(
            tool=_EchoTool(),
            args={"value": "x"},
            state=LoopState(),
        )
        assert _sc(outcome) is not None
        assert outcome.decision is not None  # type: ignore[union-attr]
        assert outcome.decision.allow is False  # type: ignore[union-attr]
        assert outcome.decision.reason == "user_deny"  # type: ignore[union-attr]
        assert "subagent_auto_deny" in (_sc(outcome).error or "")  # type: ignore[union-attr]
    finally:
        await child.aclose()


@pytest.mark.asyncio
async def test_agent_wiring_passes_deny_and_ask_rules_to_subagent_factory(
    tmp_path: Path,
) -> None:
    cfg = AuraConfig.model_validate(
        {
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": ["write_file"]},
        }
    )
    ruleset = RuleSet(rules=(Rule(tool="write_file", content=None),))
    deny_ruleset = RuleSet(rules=(Rule(tool="write_file", content=None),))
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="done"))]),
        storage=SessionStorage(tmp_path / "parent.db"),
        ruleset=ruleset,
        deny_ruleset=deny_ruleset,
        ask_ruleset=RuleSet(),
        safety=DEFAULT_SAFETY,
    )
    agent._subagent_factory._model_factory = lambda: FakeChatModel(
        turns=[FakeTurn(AIMessage(content="done"))]
    )
    agent._subagent_factory._storage_factory = lambda: SessionStorage(
        Path(":memory:")
    )
    child = agent._subagent_factory.spawn("prompt")
    try:
        write_tool = child._available_tools["write_file"]
        outcome = await child._hooks.run_pre_tool(
            tool=write_tool,
            args={"path": str(tmp_path / "out.txt"), "content": "x"},
            state=LoopState(),
        )
        assert _sc(outcome) is not None
        assert outcome.decision is not None  # type: ignore[union-attr]
        assert outcome.decision.reason == "rule_deny"  # type: ignore[union-attr]
    finally:
        await child.aclose()
        await agent.aclose()


@pytest.mark.asyncio
async def test_nested_subagent_factory_inherits_permission_context() -> None:
    parent_ruleset = RuleSet(rules=(Rule(tool="echo_tool", content=None),))
    parent_deny_rules = RuleSet(rules=(Rule(tool="echo_tool", content=None),))
    factory = _build_factory(
        parent_ruleset=parent_ruleset,
        parent_deny_rules=parent_deny_rules,
    )

    child = factory.spawn("prompt")
    child._subagent_factory._model_factory = lambda: FakeChatModel(
        turns=[FakeTurn(AIMessage(content="done"))]
    )
    child._subagent_factory._storage_factory = lambda: SessionStorage(Path(":memory:"))
    grandchild = child._subagent_factory.spawn("nested prompt")
    try:
        outcome = await grandchild._hooks.run_pre_tool(
            tool=_EchoTool(),
            args={"value": "x"},
            state=LoopState(),
        )
        assert _sc(outcome) is not None
        assert outcome.decision is not None  # type: ignore[union-attr]
        assert outcome.decision.allow is False  # type: ignore[union-attr]
        assert outcome.decision.reason == "rule_deny"  # type: ignore[union-attr]
    finally:
        await grandchild.aclose()
        await child.aclose()


# ---------------------------------------------------------------------------
# AC-C1-3 — subagent gets a fresh SessionRuleSet; no leak back to parent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_does_not_pollute_parent_session_rules() -> None:
    """A rule added to child.SessionRuleSet doesn't appear in the parent's."""
    parent_session = SessionRuleSet()
    # Parent-state proxy for "session rules the parent knows about".
    parent_session.add(Rule(tool="bash", content="ls"))
    factory = SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        parent_ruleset=RuleSet(),
        parent_safety=DEFAULT_SAFETY,
        parent_mode_provider=lambda: "default",
        parent_session=parent_session,  # exposes the link so the test can probe
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="done"))]
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    child = factory.spawn("prompt")
    try:
        # The child's session ruleset is NOT the parent's. Mutating child's
        # doesn't touch parent's.
        assert child._session_rules is not parent_session
        assert child._session_rules is not None
        child._session_rules.add(Rule(tool="echo_tool", content=None))
        # Parent still has only the single ``bash(ls)`` rule it started with.
        parent_rules = parent_session.rules()
        assert len(parent_rules) == 1
        assert parent_rules[0].tool == "bash"
        # Child's rule is present on the child.
        child_rules = child._session_rules.rules()
        assert any(r.tool == "echo_tool" for r in child_rules)
    finally:
        await child.aclose()


# ---------------------------------------------------------------------------
# AC-C1-4 — parent's ``bypass`` mode inherits into the child
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_inherits_bypass_mode_from_parent() -> None:
    """Parent in ``bypass`` → child in ``bypass`` → hook auto-allows."""
    factory = _build_factory(parent_ruleset=RuleSet(), parent_mode="bypass")
    child = factory.spawn("prompt")
    try:
        # Spawned Agent's mode must reflect the inherited bypass.
        assert child.mode == "bypass"
        # And the hook behaviour must match: a tool with no rule would
        # normally hit the ask → auto-deny path, but bypass short-circuits
        # first.
        outcome = await child._hooks.run_pre_tool(
            tool=_EchoTool(),
            args={"value": "x"},
            state=LoopState(),
        )
        assert _sc(outcome) is None
        assert outcome.decision is not None  # type: ignore[union-attr]
        assert outcome.decision.allow is True  # type: ignore[union-attr]
        assert outcome.decision.reason == "mode_bypass"  # type: ignore[union-attr]
    finally:
        await child.aclose()


@pytest.mark.asyncio
async def test_subagent_freezes_parent_mode_at_spawn() -> None:
    """Parent mode flips after spawn do not change child hook behavior."""
    mode = "default"
    factory = SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        parent_ruleset=RuleSet(),
        parent_safety=DEFAULT_SAFETY,
        parent_mode_provider=lambda: mode,
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="done"))]
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )

    child = factory.spawn("prompt")
    mode = "bypass"
    try:
        assert child.mode == "default"
        outcome = await child._hooks.run_pre_tool(
            tool=_EchoTool(),
            args={"value": "x"},
            state=LoopState(),
        )
        assert _sc(outcome) is not None
        assert outcome.decision is not None  # type: ignore[union-attr]
        assert outcome.decision.allow is False  # type: ignore[union-attr]
        assert outcome.decision.reason == "user_deny"  # type: ignore[union-attr]
        assert "subagent_auto_deny" in (_sc(outcome).error or "")  # type: ignore[union-attr]
    finally:
        await child.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_mode", ["plan", "accept_edits"])
async def test_subagent_collapses_interactive_parent_modes_to_default(
    parent_mode: str,
) -> None:
    """Plan / accept_edits do not inherit into non-interactive children."""
    factory = _build_factory(parent_ruleset=RuleSet(), parent_mode=parent_mode)
    child = factory.spawn("prompt")
    try:
        assert child.mode == "default"
        outcome = await child._hooks.run_pre_tool(
            tool=_EchoTool(),
            args={"value": "x"},
            state=LoopState(),
        )
        assert _sc(outcome) is not None
        assert outcome.decision is not None  # type: ignore[union-attr]
        assert outcome.decision.allow is False  # type: ignore[union-attr]
        assert outcome.decision.reason == "user_deny"  # type: ignore[union-attr]
        assert "subagent_auto_deny" in (_sc(outcome).error or "")  # type: ignore[union-attr]
    finally:
        await child.aclose()


# ---------------------------------------------------------------------------
# Auxiliary — the asker itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_permission_asker_returns_deny_immediately() -> None:
    """Inlined subagent asker returns ``deny`` / ``session`` with zero I/O."""
    asker = _SubagentPermissionAsker()
    response = await asker(
        tool=_EchoTool(),
        args={"value": "x"},
        rule_hint=Rule(tool="echo_tool", content=None),
    )
    assert response.choice == "deny"
    assert response.scope == "session"
    assert response.rule is None
    # ``feedback`` carries a machine-readable marker the hook uses to
    # stamp "subagent_auto_deny" onto the model-facing error message.
    assert response.feedback == SUBAGENT_AUTO_DENY_FEEDBACK


# ---------------------------------------------------------------------------
# Regression — subagent still respects safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_still_denies_on_safety_violation(tmp_path: Path) -> None:
    """Safety policy runs BEFORE the ask path — ``.git/`` writes deny."""
    # Build a write-family tool mock that would otherwise hit the ask
    # path. DefaultSafety includes ``**/.git/**`` on writes.
    class _FakeWriteParams(BaseModel):
        path: str
        content: str = ""

    class _FakeWrite(BaseTool):
        name: str = "write_file"
        description: str = "test write"
        args_schema: type[BaseModel] = _FakeWriteParams
        aura_metadata: ToolMetadata = ToolMetadata(
            is_read_only=False,
            is_destructive=True,
            is_concurrency_safe=False,
            rule_matcher=None,
            args_preview=None,
            timeout_sec=None,
        )

        def _run(self, path: str, content: str = "") -> str:
            return path

    factory = _build_factory(parent_ruleset=RuleSet())
    child = factory.spawn("prompt")
    try:
        protected = tmp_path / ".git" / "HEAD"
        outcome = await child._hooks.run_pre_tool(
            tool=_FakeWrite(),
            args={"path": str(protected), "content": "x"},
            state=LoopState(),
        )
        assert _sc(outcome) is not None
        assert outcome.decision is not None  # type: ignore[union-attr]
        assert outcome.decision.allow is False  # type: ignore[union-attr]
        assert outcome.decision.reason == "safety_blocked"  # type: ignore[union-attr]
    finally:
        await child.aclose()
