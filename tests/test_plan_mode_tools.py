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
from aura.schemas.tool import ToolError
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


async def _always_yes_asker(
    _question: str, _options: list[str] | None, _default: str | None,
) -> str:
    # Default asker for exit_plan_mode tests that don't care about the
    # approval gate itself — test_exit_plan_mode.py covers the gate.
    return "Yes"


def _exit_tool(
    agent: _FakeAgent,
    *,
    asker: Any = None,
) -> ExitPlanMode:
    return ExitPlanMode(
        mode_setter=agent.set_mode,
        mode_getter=lambda: agent.mode,
        asker=asker or _always_yes_asker,
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


def test_enter_plan_mode_saves_prior_mode_via_closure() -> None:
    # Entering plan from "accept_edits" stashes "accept_edits" via the
    # injected save closure — that's the single-writer contract the
    # Agent relies on to restore the mode on exit.
    agent = _FakeAgent(mode="accept_edits")
    saved: list[str] = []
    tool = EnterPlanMode(
        mode_setter=agent.set_mode,
        mode_getter=lambda: agent.mode,
        save_prior_mode=saved.append,
    )
    tool.invoke({"plan": "1. thing"})
    assert agent.mode == "plan"
    assert saved == ["accept_edits"]


def test_enter_plan_mode_save_prior_is_noop_when_already_plan() -> None:
    # Re-entering from plan mode is a no-op — the save closure must NOT
    # be invoked, otherwise prior_mode would get overwritten with "plan"
    # and exit_plan_mode would have nothing useful to restore.
    agent = _FakeAgent(mode="plan")
    saved: list[str] = []
    tool = EnterPlanMode(
        mode_setter=agent.set_mode,
        mode_getter=lambda: agent.mode,
        save_prior_mode=saved.append,
    )
    tool.invoke({"plan": "refined"})
    assert saved == []
    assert agent.mode == "plan"


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


async def test_exit_plan_mode_defaults_to_default() -> None:
    agent = _FakeAgent(mode="plan")
    tool = _exit_tool(agent)
    result = await tool.ainvoke({"plan": "1. do thing"})
    assert agent.mode == "default"
    assert result == {
        "previous_mode": "plan",
        "new_mode": "default",
        "plan": "1. do thing",
        "approved": True,
    }


async def test_exit_plan_mode_accepts_accept_edits_target() -> None:
    agent = _FakeAgent(mode="plan")
    tool = _exit_tool(agent)
    result = await tool.ainvoke({
        "plan": "1. edit foo", "to_mode": "accept_edits",
    })
    assert agent.mode == "accept_edits"
    assert result == {
        "previous_mode": "plan",
        "new_mode": "accept_edits",
        "plan": "1. edit foo",
        "approved": True,
    }


async def test_exit_plan_mode_rejects_non_plan_origin() -> None:
    agent = _FakeAgent(mode="default")
    tool = _exit_tool(agent)
    with pytest.raises(ToolError):
        await tool.ainvoke({"plan": "1. thing"})


def test_exit_plan_mode_rejects_bypass_as_target() -> None:
    # ExitTarget Literal excludes "bypass" — pydantic must reject before
    # the tool body runs. Proves "bypass" can't be entered via this path.
    from pydantic import ValidationError

    agent = _FakeAgent(mode="plan")
    tool = _exit_tool(agent)
    with pytest.raises(ValidationError):
        tool.invoke({"plan": "1. thing", "to_mode": "bypass"})


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
    outcome = await hook(
        tool=tool, args={"path": "/tmp/new.txt"}, state=LoopState(),
    )
    assert outcome.short_circuit is not None
    assert outcome.short_circuit.ok is False
    assert outcome.short_circuit.error is not None
    assert "plan mode" in outcome.short_circuit.error
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
    outcome = await hook(
        tool=tool, args={"path": "/tmp/ordinary.txt"}, state=LoopState(),
    )
    assert outcome.short_circuit is None
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
    outcome = await hook(
        tool=tool, args={"plan": "do the thing"}, state=LoopState(),
    )
    assert outcome.short_circuit is None
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
    outcome = await hook(tool=tool, args={}, state=LoopState())
    assert outcome.short_circuit is None
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
    outcome = await hook(tool=tool, args={}, state=LoopState())
    assert outcome.short_circuit is not None
    assert outcome.short_circuit.ok is False
    assert outcome.short_circuit.error is not None
    assert "plan mode" in outcome.short_circuit.error
