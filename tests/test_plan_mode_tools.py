"""Tests for enter_plan_mode / exit_plan_mode tools + plan-mode exemption.

Covers:

- enter_plan_mode flips Agent.mode -> "plan" and returns the envelope.
- enter_plan_mode from plan mode is a no-op (previous_mode == "plan",
  note set).
- exit_plan_mode from plan -> "default" (default to_mode).
- exit_plan_mode(to_mode="accept_edits") -> "accept_edits".
- exit_plan_mode from non-plan raises ToolError.
- Plan mode blocks write_file via the permission hook (regression for
  the permission-layer path).
- Plan mode allows read_file via the permission hook (the new allow-list).
- enter_plan_mode itself is NOT blocked by plan-mode enforcement — the
  bootstrap-safety contract that lets the LLM leave plan mode.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks.permission import AskerResponse, make_permission_hook
from aura.core.permissions.rule import Rule
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolError, ToolResult
from aura.tools.base import build_tool
from aura.tools.enter_plan_mode import EnterPlanMode
from aura.tools.exit_plan_mode import ExitPlanMode

# ---------------------------------------------------------------------------
# Fake Agent that exposes set_mode / mode — the two entry points the tools use.
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, mode: str = "default") -> None:
        self.mode = mode

    def set_mode(self, mode: str) -> None:
        self.mode = mode


def _enter_tool(agent: _FakeAgent) -> EnterPlanMode:
    return EnterPlanMode(
        mode_setter=agent.set_mode, mode_getter=lambda: agent.mode,
    )


def _exit_tool(agent: _FakeAgent) -> ExitPlanMode:
    return ExitPlanMode(
        mode_setter=agent.set_mode, mode_getter=lambda: agent.mode,
    )


# ---------------------------------------------------------------------------
# enter_plan_mode
# ---------------------------------------------------------------------------


def test_enter_plan_mode_flips_mode_and_returns_envelope() -> None:
    agent = _FakeAgent(mode="default")
    tool = _enter_tool(agent)
    result = tool.invoke({"plan": "1. read foo.py\n2. write bar.py"})
    assert agent.mode == "plan"
    assert result == {
        "previous_mode": "default",
        "new_mode": "plan",
        "plan": "1. read foo.py\n2. write bar.py",
    }


def test_enter_plan_mode_from_plan_is_noop_with_note() -> None:
    agent = _FakeAgent(mode="plan")
    tool = _enter_tool(agent)
    result = tool.invoke({"plan": "refined plan"})
    assert agent.mode == "plan"
    assert isinstance(result, dict)
    assert result["previous_mode"] == "plan"
    assert result["new_mode"] == "plan"
    assert "note" in result


def test_enter_plan_mode_rejects_empty_plan() -> None:
    from pydantic import ValidationError

    agent = _FakeAgent()
    tool = _enter_tool(agent)
    with pytest.raises(ValidationError):
        tool.invoke({"plan": ""})


def test_enter_plan_mode_caps_plan_length() -> None:
    from pydantic import ValidationError

    agent = _FakeAgent()
    tool = _enter_tool(agent)
    with pytest.raises(ValidationError):
        tool.invoke({"plan": "x" * 4001})


# ---------------------------------------------------------------------------
# exit_plan_mode
# ---------------------------------------------------------------------------


def test_exit_plan_mode_defaults_to_default() -> None:
    agent = _FakeAgent(mode="plan")
    tool = _exit_tool(agent)
    result = tool.invoke({})
    assert agent.mode == "default"
    assert result == {"previous_mode": "plan", "new_mode": "default"}


def test_exit_plan_mode_accepts_accept_edits_target() -> None:
    agent = _FakeAgent(mode="plan")
    tool = _exit_tool(agent)
    result = tool.invoke({"to_mode": "accept_edits"})
    assert agent.mode == "accept_edits"
    assert result == {"previous_mode": "plan", "new_mode": "accept_edits"}


def test_exit_plan_mode_rejects_non_plan_origin() -> None:
    agent = _FakeAgent(mode="default")
    tool = _exit_tool(agent)
    with pytest.raises(ToolError):
        tool.invoke({})


def test_exit_plan_mode_rejects_bypass_as_target() -> None:
    # ExitTarget Literal excludes "bypass" — pydantic must reject before
    # the tool body runs. Proves "bypass" can't be entered via this path.
    from pydantic import ValidationError

    agent = _FakeAgent(mode="plan")
    tool = _exit_tool(agent)
    with pytest.raises(ValidationError):
        tool.invoke({"to_mode": "bypass"})


# ---------------------------------------------------------------------------
# permission hook — plan-mode exemption / read-allow / write-block
# ---------------------------------------------------------------------------


class _P(BaseModel):
    pass


class _PathArgs(BaseModel):
    path: str


class _PlanArgs(BaseModel):
    plan: str


def _mk_tool(
    name: str,
    *,
    is_read_only: bool = False,
    is_destructive: bool = False,
    rule_matcher: Callable[[dict[str, Any], str], bool] | None = None,
    args_schema: type[BaseModel] = _P,
) -> BaseTool:
    def _noop() -> dict[str, Any]:
        return {}

    return build_tool(
        name=name,
        description=name,
        args_schema=args_schema,
        func=_noop,
        is_read_only=is_read_only,
        is_destructive=is_destructive,
        rule_matcher=rule_matcher,
    )


class _Spy:
    def __init__(self, response: AskerResponse | None = None) -> None:
        self.response = response
        self.calls: list[str] = []

    async def __call__(
        self, *, tool: BaseTool, args: dict[str, Any], rule_hint: Rule,
    ) -> AskerResponse:
        self.calls.append(tool.name)
        assert self.response is not None
        return self.response


async def test_plan_mode_blocks_write_file_through_hook() -> None:
    spy = _Spy()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
        mode="plan",
    )
    tool = _mk_tool("write_file", is_destructive=True, args_schema=_PathArgs)
    result = await hook(
        tool=tool, args={"path": "/tmp/new.txt"}, state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "plan mode" in result.error
    assert spy.calls == []


async def test_plan_mode_allows_read_file_through_hook() -> None:
    # read_file is on _PLAN_MODE_READ_TOOLS. A default-allow rule covers
    # it, so the hook falls through to rule_allow and returns None.
    spy = _Spy()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(rules=(Rule(tool="read_file", content=None),)),
        project_root=Path("/tmp"),
        mode="plan",
    )
    tool = _mk_tool("read_file", is_read_only=True, args_schema=_PathArgs)
    result = await hook(
        tool=tool, args={"path": "/tmp/ordinary.txt"}, state=LoopState(),
    )
    assert result is None
    assert spy.calls == []


async def test_enter_plan_mode_is_not_blocked_by_plan_mode() -> None:
    # Bootstrap safety: enter_plan_mode must be reachable even when plan
    # mode is already on, otherwise re-entering would dry-run-deny. A
    # matching rule means the tool falls straight through to rule_allow.
    spy = _Spy()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(rules=(Rule(tool="enter_plan_mode", content=None),)),
        project_root=Path("/tmp"),
        mode="plan",
    )
    tool = _mk_tool("enter_plan_mode", args_schema=_PlanArgs)
    result = await hook(
        tool=tool, args={"plan": "do the thing"}, state=LoopState(),
    )
    assert result is None
    assert spy.calls == []


async def test_exit_plan_mode_is_not_blocked_by_plan_mode() -> None:
    # The whole point of the exemption: the LLM must be able to leave
    # plan mode. A matching rule falls straight through.
    spy = _Spy()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(rules=(Rule(tool="exit_plan_mode", content=None),)),
        project_root=Path("/tmp"),
        mode="plan",
    )
    tool = _mk_tool("exit_plan_mode", args_schema=_P)
    result = await hook(tool=tool, args={}, state=LoopState())
    assert result is None
    assert spy.calls == []


async def test_plan_mode_still_blocks_unknown_tools() -> None:
    # Fail closed: a tool the hook has never heard of is NOT on the
    # read-allow list and NOT on the exempt list, so plan mode dry-runs it.
    hook = make_permission_hook(
        asker=_Spy(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
        mode="plan",
    )
    tool = _mk_tool("weird_custom_tool", args_schema=_P)
    result = await hook(tool=tool, args={}, state=LoopState())
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "plan mode" in result.error
