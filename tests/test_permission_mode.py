"""Tests for aura.core.permissions.mode — Mode literal alias + DEFAULT_MODE.

Extended (2026-04-21) for the 4-mode permission completion:
``plan`` and ``accept_edits`` join the existing ``default`` / ``bypass``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks.permission import (
    AskerResponse,
    make_permission_hook,
)
from aura.core.permissions.decision import Decision
from aura.core.permissions.mode import DEFAULT_MODE, Mode
from aura.core.permissions.rule import Rule
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool


def test_default_mode_is_default_string() -> None:
    assert DEFAULT_MODE == "default"


def test_mode_literal_accepts_all_four_values() -> None:
    # Mode is a Literal alias — the four string values are its runtime args.
    assert set(get_args(Mode)) == {"default", "bypass", "plan", "accept_edits"}


def test_mode_alias_matches_schemas_permissions_mode_field() -> None:
    # Parallel type-equivalence with PermissionsConfig.mode: same Literal values.
    from aura.schemas.permissions import PermissionsConfig

    schema_field = PermissionsConfig.model_fields["mode"].annotation
    assert set(get_args(schema_field)) == set(get_args(Mode))


# ---------------------------------------------------------------------------
# Shared helpers for plan / accept_edits hook tests.
# ---------------------------------------------------------------------------


class _P(BaseModel):
    pass


class _PathArgs(BaseModel):
    path: str


class _BashArgs(BaseModel):
    command: str


def _noop() -> dict[str, Any]:
    return {}


def _tool(
    name: str = "writer",
    *,
    is_read_only: bool = False,
    is_destructive: bool = False,
    rule_matcher: Callable[[dict[str, Any], str], bool] | None = None,
    args_schema: type[BaseModel] = _P,
) -> BaseTool:
    return build_tool(
        name=name,
        description=name,
        args_schema=args_schema,
        func=_noop,
        is_read_only=is_read_only,
        is_destructive=is_destructive,
        rule_matcher=rule_matcher,
    )


@dataclass
class _SpyAsker:
    response: AskerResponse | None = None
    raise_: BaseException | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Rule,
    ) -> AskerResponse:
        self.calls.append({"tool": tool.name, "args": args, "rule_hint": rule_hint})
        if self.raise_ is not None:
            raise self.raise_
        assert self.response is not None
        return self.response


@pytest.fixture
def journal_events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Intercept journal.write from the hook module and collect events in order."""
    events: list[tuple[str, dict[str, Any]]] = []

    def _capture(event: str, /, **fields: Any) -> None:
        events.append((event, fields))

    from aura.core.persistence import journal as journal_mod

    monkeypatch.setattr(journal_mod, "write", _capture)
    return events


# ---------------------------------------------------------------------------
# plan mode — dry-run every tool call
# ---------------------------------------------------------------------------


async def test_plan_mode_blocks_read_file() -> None:
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        # Even with a matching rule, plan mode still dry-runs.
        rules=RuleSet(rules=(Rule(tool="read_file", content=None),)),
        project_root=Path("/tmp"),
        mode="plan",
    )
    tool = _tool("read_file", is_read_only=True, args_schema=_PathArgs)
    result = await hook(
        tool=tool, args={"path": "/tmp/ordinary.txt"}, state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert spy.calls == []


async def test_plan_mode_blocks_write_file() -> None:
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
        mode="plan",
    )
    tool = _tool("write_file", is_destructive=True, args_schema=_PathArgs)
    result = await hook(
        tool=tool, args={"path": "/tmp/new.txt"}, state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert spy.calls == []


async def test_plan_mode_blocks_bash() -> None:
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
        mode="plan",
    )
    tool = _tool("bash", is_destructive=True, args_schema=_BashArgs)
    result = await hook(
        tool=tool, args={"command": "ls"}, state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert spy.calls == []


async def test_plan_mode_error_says_would_have_called() -> None:
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
        mode="plan",
    )
    tool = _tool("bash", is_destructive=True, args_schema=_BashArgs)
    result = await hook(
        tool=tool, args={"command": "ls -la"}, state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    # Must mention plan mode + the tool name that was attempted.
    assert result.error is not None
    assert "plan mode" in result.error
    assert "would have called" in result.error
    assert "bash" in result.error


async def test_plan_mode_respects_safety_floor(tmp_path: Path) -> None:
    """Plan mode does NOT bypass safety. A safety-protected read still
    surfaces as ``safety_blocked`` in the Decision, not ``plan_mode_blocked``.
    This matches the invariant that safety is a separate axis from permission.
    """
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(rules=(Rule(tool="read_file", content=None),)),
        project_root=tmp_path,
        mode="plan",
    )
    tool = _tool("read_file", is_read_only=True, args_schema=_PathArgs)
    state = LoopState()
    result = await hook(
        tool=tool,
        args={"path": str(Path.home() / ".ssh" / "id_rsa")},
        state=state,
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    # Safety wins: the decision must report safety_blocked, not plan_mode_blocked.
    stashed = state.custom.get("_aura_pending_decision")
    assert isinstance(stashed, Decision)
    assert stashed.reason == "safety_blocked"


async def test_plan_mode_stashes_plan_decision_in_state(
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
        mode="plan",
    )
    tool = _tool("write_file", is_destructive=True, args_schema=_PathArgs)
    state = LoopState()
    await hook(
        tool=tool, args={"path": "/tmp/x"}, state=state,
    )
    stashed = state.custom.get("_aura_pending_decision")
    assert isinstance(stashed, Decision)
    assert stashed.reason == "plan_mode_blocked"
    assert stashed.allow is False
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "plan_mode_blocked"
    assert decision_event[1]["mode"] == "plan"


# ---------------------------------------------------------------------------
# accept_edits mode — auto-allow read/edit/write tools, prompt everything else
# ---------------------------------------------------------------------------


async def test_accept_edits_mode_allows_read_file(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),  # no rules — in default mode this would prompt
        project_root=tmp_path,
        mode="accept_edits",
    )
    tool = _tool("read_file", is_read_only=True, args_schema=_PathArgs)
    result = await hook(
        tool=tool, args={"path": str(tmp_path / "ordinary.txt")}, state=LoopState(),
    )
    assert result is None
    assert spy.calls == []
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "mode_accept_edits"


async def test_accept_edits_mode_allows_write_file(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="accept_edits",
    )
    tool = _tool("write_file", is_destructive=True, args_schema=_PathArgs)
    result = await hook(
        tool=tool, args={"path": str(tmp_path / "new.txt")}, state=LoopState(),
    )
    assert result is None
    assert spy.calls == []
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "mode_accept_edits"


async def test_accept_edits_mode_allows_edit_file(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="accept_edits",
    )
    tool = _tool("edit_file", is_destructive=True, args_schema=_PathArgs)
    result = await hook(
        tool=tool, args={"path": str(tmp_path / "edited.txt")}, state=LoopState(),
    )
    assert result is None
    assert spy.calls == []
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "mode_accept_edits"


async def test_accept_edits_mode_still_prompts_bash(tmp_path: Path) -> None:
    """bash is NOT on the edit list — accept_edits falls through to ask.
    An unbounded shell invocation must stay gated behind the prompt."""
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="accept_edits",
    )
    tool = _tool("bash", is_destructive=True, args_schema=_BashArgs)
    result = await hook(
        tool=tool, args={"command": "ls"}, state=LoopState(),
    )
    assert result is None
    # The asker was consulted — proving accept_edits did NOT auto-allow bash.
    assert len(spy.calls) == 1


async def test_accept_edits_respects_safety_floor(tmp_path: Path) -> None:
    """accept_edits does NOT bypass safety. A protected write still
    reports ``safety_blocked`` rather than ``mode_accept_edits``."""
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="accept_edits",
    )
    tool = _tool("write_file", is_destructive=True, args_schema=_PathArgs)
    state = LoopState()
    result = await hook(
        tool=tool,
        args={"path": str(tmp_path / ".git" / "HEAD")},
        state=state,
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    stashed = state.custom.get("_aura_pending_decision")
    assert isinstance(stashed, Decision)
    assert stashed.reason == "safety_blocked"


async def test_accept_edits_respects_user_deny_rule(tmp_path: Path) -> None:
    """If a user explicitly denies (via the asker prompt), accept_edits
    still honors the deny for non-edit tools. (Rule-based denies are not
    modelled in MVP — rules are allow-only — but the ask path exercises
    the same contract: user intent wins when the mode doesn't short-circuit.)
    """
    spy = _SpyAsker(response=AskerResponse(choice="deny"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="accept_edits",
    )
    tool = _tool("bash", is_destructive=True, args_schema=_BashArgs)
    result = await hook(
        tool=tool, args={"command": "rm -rf /tmp/wat"}, state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error == "denied: user"


# ---------------------------------------------------------------------------
# audit_line coverage for the two new reasons
# ---------------------------------------------------------------------------


def test_decision_audit_line_for_accept_edits() -> None:
    line = Decision(allow=True, reason="mode_accept_edits").audit_line()
    assert line == "allowed: mode_accept_edits"


def test_decision_audit_line_for_plan_mode() -> None:
    line = Decision(allow=False, reason="plan_mode_blocked").audit_line()
    assert line == "blocked: plan mode (dry-run)"


# ---------------------------------------------------------------------------
# Preserve existing default / bypass behavior (regression guards).
# ---------------------------------------------------------------------------


async def test_default_mode_unchanged_ask_flow(tmp_path: Path) -> None:
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="default",
    )
    tool = _tool("writer")
    result = await hook(tool=tool, args={}, state=LoopState())
    assert result is None
    assert len(spy.calls) == 1


async def test_bypass_mode_unchanged_auto_allow() -> None:
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
        mode="bypass",
    )
    tool = _tool("writer")
    result = await hook(tool=tool, args={}, state=LoopState())
    assert result is None
    assert spy.calls == []


# Silence "async def not awaited" warnings by letting pytest-asyncio pick up
# the tests in auto mode. The project's pytest config already enables this —
# but we import asyncio above for future use / clarity.
_ = asyncio  # prevent lint "unused import" if pytest mode changes.
