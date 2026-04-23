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
from langchain_core.tools import BaseTool, StructuredTool
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
from aura.schemas.tool import ToolResult, tool_metadata
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


async def test_plan_mode_allows_read_file() -> None:
    # Plan mode keeps the read-tool allow-list reachable so the planner
    # can gather context. A matching default-allow rule covers read_file,
    # so the hook should return None (allow) without prompting.
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(rules=(Rule(tool="read_file", content=None),)),
        project_root=Path("/tmp"),
        mode="plan",
    )
    tool = _tool("read_file", is_read_only=True, args_schema=_PathArgs)
    result = await hook(
        tool=tool, args={"path": "/tmp/ordinary.txt"}, state=LoopState(),
    )
    assert result is None
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


# ---------------------------------------------------------------------------
# Input-aware is_destructive — the safety layer's direction flag must
# honour per-call classifiers (claude-code's isDestructive(input) pattern).
# A static ``metadata.get("is_destructive")`` would see the classifier as
# truthy and misclassify every invocation as destructive.
# ---------------------------------------------------------------------------


def _tool_with_classifier(
    *,
    name: str,
    args_schema: type[BaseModel],
    is_destructive_fn: Any,
) -> BaseTool:
    """Build a StructuredTool with a callable is_destructive in metadata.

    ``build_tool`` accepts only static bools for is_destructive; this
    helper sidesteps it by calling ``tool_metadata`` directly so we
    can exercise the callable branch end-to-end through the hook.
    """
    return StructuredTool.from_function(
        func=lambda **_kw: {},
        name=name,
        description=name,
        args_schema=args_schema,
        metadata=tool_metadata(is_destructive=is_destructive_fn),
    )


async def test_safety_uses_callable_is_destructive_true_branch(tmp_path: Path) -> None:
    """A callable that returns True must route through protected_writes.

    Build a path-bearing tool whose classifier returns True; point it at
    a path that's on the protected_writes list. The hook must block it
    with ``safety_blocked`` — proving the resolver extracted True from
    the callable rather than seeing the function object as truthy and
    coincidentally matching.
    """
    tool = _tool_with_classifier(
        name="destructive_writer",
        args_schema=_PathArgs,
        is_destructive_fn=lambda _args: True,
    )
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="default",
    )
    state = LoopState()
    result = await hook(
        tool=tool,
        args={"path": str(tmp_path / ".git" / "HEAD")},
        state=state,
    )
    assert isinstance(result, ToolResult)
    stashed = state.custom.get("_aura_pending_decision")
    assert isinstance(stashed, Decision)
    assert stashed.reason == "safety_blocked"


async def test_safety_uses_callable_is_destructive_false_branch(tmp_path: Path) -> None:
    """A callable that returns False must route through protected_reads.

    For paths not on the reads list but on the writes list, a read-like
    classification should NOT surface safety_blocked. This is the safe
    ``bash("ls /tmp/.aura/...")`` case — the static True path used to
    block it unnecessarily.
    """
    # Use a path that's considered a "write target" (under .git) but
    # our classifier says this is a read-only operation. .git is on
    # protected_writes but NOT on protected_reads, so the read-direction
    # lookup returns False and we fall through to the ask path.
    tool = _tool_with_classifier(
        name="benign_reader",
        args_schema=_PathArgs,
        is_destructive_fn=lambda _args: False,
    )
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="default",
    )
    state = LoopState()
    result = await hook(
        tool=tool,
        args={"path": str(tmp_path / ".git" / "HEAD")},
        state=state,
    )
    # Not blocked by safety — fell through to the ask path.
    assert result is None
    stashed = state.custom.get("_aura_pending_decision")
    assert isinstance(stashed, Decision)
    assert stashed.reason != "safety_blocked"


async def test_safety_callable_receives_actual_args(tmp_path: Path) -> None:
    """The hook must pass the real args dict to the classifier — not
    ``{}`` or some sentinel. Guards against a refactor that silently
    swaps in an empty dict (which would break every input-aware tool)."""
    received: list[dict[str, Any]] = []

    def classifier(args: dict[str, Any]) -> bool:
        received.append(args)
        # Return True so we land on the protected path and observe the
        # safety branch in action.
        return True

    tool = _tool_with_classifier(
        name="probe",
        args_schema=_PathArgs,
        is_destructive_fn=classifier,
    )
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="default",
    )
    await hook(
        tool=tool,
        args={"path": str(tmp_path / ".git" / "HEAD")},
        state=LoopState(),
    )
    assert received == [{"path": str(tmp_path / ".git" / "HEAD")}]


async def test_safety_callable_exception_fails_safe_to_destructive(
    tmp_path: Path,
) -> None:
    """A throwing classifier must be treated as destructive — NOT
    waved through with a silent False. A broken classifier that fell
    open would be a silent permission regression."""

    def broken(_args: dict[str, Any]) -> bool:
        raise RuntimeError("buggy classifier")

    tool = _tool_with_classifier(
        name="broken",
        args_schema=_PathArgs,
        is_destructive_fn=broken,
    )
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="default",
    )
    state = LoopState()
    result = await hook(
        tool=tool,
        args={"path": str(tmp_path / ".git" / "HEAD")},
        state=state,
    )
    # Fail-safe: treated as destructive → safety_blocked on protected write path.
    assert isinstance(result, ToolResult)
    stashed = state.custom.get("_aura_pending_decision")
    assert isinstance(stashed, Decision)
    assert stashed.reason == "safety_blocked"


async def test_accept_edits_bash_ls_still_prompts_not_auto_allowed(
    tmp_path: Path,
) -> None:
    """Conservative-over-claude-code decision: even though ``bash("ls")``
    now resolves is_destructive=False via the input-aware classifier,
    accept_edits mode still does NOT auto-allow it. The
    ``_ACCEPT_EDITS_TOOLS`` allow-list is keyed on tool NAME
    (read/write/edit_file), not on the resolved destructiveness flag —
    opting into "accept file edits" must not silently opt into shell
    commands just because they happen to be read-like."""
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="accept_edits",
    )
    # Real bash tool carries the callable classifier; "ls" resolves
    # is_destructive=False, but the tool name is "bash" which is NOT
    # on the accept_edits allow-list.
    from aura.tools.bash import bash as real_bash

    result = await hook(
        tool=real_bash, args={"command": "ls /tmp"}, state=LoopState(),
    )
    # Fell through to ask — the asker was consulted.
    assert result is None
    assert len(spy.calls) == 1


# ---------------------------------------------------------------------------
# B1 — audit journal records the LIVE resolved mode, not the captured param.
#
# When ``mode`` is a Callable the journal must record the resolved string,
# not the function object (which would either blow up JSON serialization
# or write an opaque ``"<function ... at 0x...>"`` blob). When the mode
# provider changes mid-session (``Agent.set_mode``) the *next* journal
# event must reflect the new value, not a startup-frozen snapshot.
# ---------------------------------------------------------------------------


async def test_permission_audit_mode_live_read_with_callable(
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    """AC-B1-1: Callable mode → journal records the resolved string, not the fn."""
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
        mode=lambda: "accept_edits",
    )
    # write_file is on the accept_edits allow-list — tool will be auto-allowed,
    # giving us a clean ``permission_decision`` event to inspect.
    tool = _tool("write_file", is_destructive=True, args_schema=_PathArgs)
    await hook(
        tool=tool, args={"path": "/tmp/new.txt"}, state=LoopState(),
    )
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    recorded_mode = decision_event[1]["mode"]
    # Must be the resolved literal string, not the Callable itself nor a repr.
    assert recorded_mode == "accept_edits"
    assert isinstance(recorded_mode, str)
    assert not callable(recorded_mode)


async def test_permission_audit_mode_reflects_mid_turn_switch(
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    """AC-B1-2: mid-turn mode switch → next journal event records the new mode."""
    current_mode: list[Mode] = ["default"]

    def mode_provider() -> Mode:
        return current_mode[0]

    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
        mode=mode_provider,
    )
    tool = _tool("writer")

    # Turn 1 — default mode, asker prompts, decision journaled.
    await hook(tool=tool, args={}, state=LoopState())
    first = next(e for e in journal_events if e[0] == "permission_decision")
    assert first[1]["mode"] == "default"

    # Simulate Agent.set_mode("plan") between turns.
    current_mode[0] = "plan"

    # Turn 2 — plan mode, write-adjacent tool is blocked by plan dry-run.
    write_tool = _tool("write_file", is_destructive=True, args_schema=_PathArgs)
    await hook(
        tool=write_tool, args={"path": "/tmp/x"}, state=LoopState(),
    )
    events_for_decision = [e for e in journal_events if e[0] == "permission_decision"]
    # The *newest* decision event must reflect the switched mode.
    assert events_for_decision[-1][1]["mode"] == "plan"


# Silence "async def not awaited" warnings by letting pytest-asyncio pick up
# the tests in auto mode. The project's pytest config already enables this —
# but we import asyncio above for future use / clarity.
_ = asyncio  # prevent lint "unused import" if pytest mode changes.
