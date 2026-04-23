"""Tests for the exit_plan_mode user-approval gate.

Focus: the gate itself — does the tool ask the user BEFORE mutating mode,
and does it honor the answer? The generic "tool flips mode" /
"pydantic rejects bypass" paths live in ``test_plan_mode_tools.py``;
this file owns the approval-flow behavior added to match claude-code's
``ExitPlanModeV2Tool.checkPermissions`` semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks.permission import AskerResponse, make_permission_hook
from aura.core.permissions.rule import Rule
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolError
from aura.tools.base import build_tool
from aura.tools.exit_plan_mode import ExitPlanMode


class _FakeAgent:
    def __init__(self, mode: str = "plan") -> None:
        self.mode = mode

    def set_mode(self, mode: str) -> None:
        self.mode = mode


class _RecordingAsker:
    """QuestionAsker spy — records each call, returns a configured answer.

    Mirrors the ``_stub_asker`` in ``test_ask_user.py`` but as a class so
    tests can mutate ``return_value`` between calls (exercising the
    approve-then-deny or deny-then-approve ordering).
    """

    def __init__(self, return_value: str = "Yes") -> None:
        self.return_value = return_value
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        question: str,
        options: list[str] | None,
        default: str | None,
    ) -> str:
        self.calls.append(
            {"question": question, "options": options, "default": default},
        )
        return self.return_value


def _make_tool(
    agent: _FakeAgent, asker: _RecordingAsker,
) -> ExitPlanMode:
    return ExitPlanMode(
        mode_setter=agent.set_mode,
        mode_getter=lambda: agent.mode,
        asker=asker,
    )


# ---------------------------------------------------------------------------
# Approval gate — the core contract added in this change.
# ---------------------------------------------------------------------------


async def test_asks_user_before_mutating_mode() -> None:
    # The asker MUST be called, and at the moment it's called the mode
    # must still be "plan" — otherwise we've already transitioned, which
    # defeats the approval gate. We enforce the ordering by peeking at
    # agent.mode from inside the asker via a custom spy.
    agent = _FakeAgent(mode="plan")
    observed_modes: list[str] = []

    async def _peeking_asker(
        _q: str, _opts: list[str] | None, _default: str | None,
    ) -> str:
        observed_modes.append(agent.mode)
        return "Yes"

    tool = ExitPlanMode(
        mode_setter=agent.set_mode,
        mode_getter=lambda: agent.mode,
        asker=_peeking_asker,
    )
    await tool.ainvoke({"plan": "1. do the thing"})
    assert observed_modes == ["plan"], (
        "asker must be called while still in plan mode — gate violated"
    )
    assert agent.mode == "default"


async def test_approval_flips_mode_and_includes_plan_in_result() -> None:
    agent = _FakeAgent(mode="plan")
    asker = _RecordingAsker("Yes")
    tool = _make_tool(agent, asker)
    result = await tool.ainvoke({"plan": "1. edit foo\n2. run tests"})
    assert agent.mode == "default"
    assert result == {
        "previous_mode": "plan",
        "new_mode": "default",
        "plan": "1. edit foo\n2. run tests",
        "approved": True,
    }
    assert len(asker.calls) == 1
    # The rendered prompt embeds the plan so the user sees WHAT they're
    # approving — not just a naked yes/no.
    assert "1. edit foo" in asker.calls[0]["question"]
    assert asker.calls[0]["options"] == ["Yes", "No"]
    # Fail-safe default: "No" so a stray Enter doesn't auto-approve.
    assert asker.calls[0]["default"] == "No"


async def test_approval_with_accept_edits_target() -> None:
    agent = _FakeAgent(mode="plan")
    asker = _RecordingAsker("Yes")
    tool = _make_tool(agent, asker)
    result = await tool.ainvoke({
        "plan": "1. refactor bar", "to_mode": "accept_edits",
    })
    assert agent.mode == "accept_edits"
    assert result["new_mode"] == "accept_edits"
    assert result["approved"] is True


async def test_denial_preserves_plan_mode_and_raises_toolerror() -> None:
    agent = _FakeAgent(mode="plan")
    asker = _RecordingAsker("No")
    tool = _make_tool(agent, asker)
    with pytest.raises(ToolError) as exc_info:
        await tool.ainvoke({"plan": "1. nuke prod"})
    # Mode stays — the whole point of the gate.
    assert agent.mode == "plan"
    # Error is informative so the LLM knows why it bounced + what to do.
    assert "reject" in str(exc_info.value).lower()
    assert "plan mode" in str(exc_info.value).lower()


async def test_denial_via_empty_answer_is_failsafe_denial() -> None:
    # An empty answer (CLI cancel / Ctrl+C / asker returning "") must be
    # treated as a denial, NOT as an accidental approval. Fail-safe.
    agent = _FakeAgent(mode="plan")
    asker = _RecordingAsker("")
    tool = _make_tool(agent, asker)
    with pytest.raises(ToolError):
        await tool.ainvoke({"plan": "1. anything"})
    assert agent.mode == "plan"


async def test_denial_case_insensitive_non_yes_is_denial() -> None:
    # Any non-"yes" answer (case-insensitive) counts as denial — belt and
    # braces against CLI pickers that return "no" / "NO" / "nope".
    agent = _FakeAgent(mode="plan")
    asker = _RecordingAsker("nope")
    tool = _make_tool(agent, asker)
    with pytest.raises(ToolError):
        await tool.ainvoke({"plan": "1. thing"})
    assert agent.mode == "plan"


async def test_approval_answer_is_case_insensitive() -> None:
    # "yes" / "YES" / "Yes" all approve — robust to asker implementations
    # that normalize differently.
    for answer in ("yes", "YES", "Yes"):
        agent = _FakeAgent(mode="plan")
        asker = _RecordingAsker(answer)
        tool = _make_tool(agent, asker)
        result = await tool.ainvoke({"plan": "1. thing"})
        assert agent.mode == "default", f"{answer!r} should approve"
        assert result["approved"] is True


# ---------------------------------------------------------------------------
# Category errors — still enforced.
# ---------------------------------------------------------------------------


async def test_called_outside_plan_mode_raises_toolerror_without_asking() -> None:
    # If the tool was somehow invoked from default mode (LLM confusion,
    # bug), we must reject BEFORE asking — asking "exit plan mode?" when
    # we're not in plan mode would be absurd and waste the user's time.
    agent = _FakeAgent(mode="default")
    asker = _RecordingAsker("Yes")
    tool = _make_tool(agent, asker)
    with pytest.raises(ToolError):
        await tool.ainvoke({"plan": "1. thing"})
    assert asker.calls == [], "must not prompt when outside plan mode"
    assert agent.mode == "default"


def test_sync_run_raises_notimplemented() -> None:
    # Async-only — the asker needs an event loop. Mirror ask_user_question.
    agent = _FakeAgent(mode="plan")
    asker = _RecordingAsker("Yes")
    tool = _make_tool(agent, asker)
    with pytest.raises(NotImplementedError):
        tool.invoke({"plan": "1. thing"})


# ---------------------------------------------------------------------------
# Permission-layer contract — tool still exempt from plan-mode enforcement.
# The gate we added is INSIDE the tool; the outer permission allowlist
# must stay in place or the LLM could never reach the gate in the first
# place. Regression guard for the "exemption preserved" rule.
# ---------------------------------------------------------------------------


class _P(BaseModel):
    plan: str
    to_mode: str = "default"


def _mk_tool(name: str) -> BaseTool:
    def _noop() -> dict[str, Any]:
        return {}

    return build_tool(
        name=name, description=name, args_schema=_P, func=_noop,
    )


class _HookSpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(
        self, *, tool: BaseTool, args: dict[str, Any], rule_hint: Rule,
    ) -> AskerResponse:
        self.calls.append(tool.name)
        # Unused in this test — falls through to rule_allow instead.
        raise AssertionError(
            "asker should not be reached; exit_plan_mode is exempt + rule-allowed",
        )


async def test_exit_plan_mode_bypasses_plan_mode_blocklist_at_hook() -> None:
    # The permission hook must let exit_plan_mode through in plan mode
    # (otherwise the gate we added inside the tool is unreachable). A
    # matching default-allow rule means the hook returns None (allow).
    spy = _HookSpy()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(rules=(Rule(tool="exit_plan_mode", content=None),)),
        project_root=Path("/tmp"),
        mode="plan",
    )
    tool = _mk_tool("exit_plan_mode")
    result = await hook(
        tool=tool, args={"plan": "x"}, state=LoopState(),
    )
    assert result is None
    assert spy.calls == []
