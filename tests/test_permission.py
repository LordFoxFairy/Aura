"""Tests for aura.core.hooks.permission.make_permission_hook.

Covers the post-Plan-B decision flow in spec §5 — order, short-circuits,
journal events, AskerResponse invariants, and the asker-exception-is-deny
defensive path.

Post Plan B (2026-04-21): the old ``is_read_only`` auto-allow branch is
gone. Tools previously auto-allowed by that branch now flow through
``rule_allow`` against ``DEFAULT_ALLOW_RULES``. Safety fires for any
path-arg tool (reads list or writes list depending on direction).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks.permission import (
    AskerResponse,
    PermissionAsker,
    make_permission_hook,
)
from aura.core.permissions.defaults import DEFAULT_ALLOW_RULES
from aura.core.permissions.rule import Rule
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.core.permissions.store import PermissionStoreError
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool


class _P(BaseModel):
    pass


class _PathArgs(BaseModel):
    path: str


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


# --- AskerResponse invariants ---


def test_asker_response_always_requires_rule() -> None:
    with pytest.raises(ValueError):
        AskerResponse(choice="always", rule=None)


def test_asker_response_accept_rejects_rule() -> None:
    with pytest.raises(ValueError):
        AskerResponse(choice="accept", rule=Rule("bash", None))


def test_asker_response_deny_rejects_rule() -> None:
    with pytest.raises(ValueError):
        AskerResponse(choice="deny", rule=Rule("bash", None))


def test_asker_response_is_frozen() -> None:
    resp = AskerResponse(choice="accept")
    with pytest.raises(FrozenInstanceError):
        resp.choice = "deny"  # type: ignore[misc]


def test_permission_asker_runtime_checkable() -> None:
    asker = _SpyAsker(response=AskerResponse(choice="accept"))
    assert isinstance(asker, PermissionAsker)


# --- Decision flow ---


async def test_bypass_mode_short_circuits_even_on_protected_path(
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    spy = _SpyAsker()
    session = SessionRuleSet()
    hook = make_permission_hook(
        asker=spy,
        session=session,
        rules=RuleSet(),
        project_root=Path("/tmp/nope"),
        mode="bypass",
    )
    # Destructive tool writing to a protected path — bypass still wins.
    result = await hook(
        tool=_tool(is_destructive=True, args_schema=_PathArgs),
        args={"path": "/etc/passwd"},
        state=LoopState(),
    )
    assert result is None
    assert spy.calls == []
    names = [e[0] for e in journal_events]
    assert "permission_bypass" in names
    assert "permission_decision" in names
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "mode_bypass"


async def test_safety_blocks_destructive_write_to_protected_path(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
    )
    result = await hook(
        tool=_tool(is_destructive=True, args_schema=_PathArgs),
        args={"path": str(tmp_path / ".git" / "HEAD")},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error == "denied: protected path (safety policy)"
    assert spy.calls == []
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "safety_blocked"


async def test_safety_blocks_read_file_of_ssh_key(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    """The invariant we're closing: read_file(~/.ssh/id_rsa) denies with
    safety_blocked instead of being auto-allowed."""
    # Rule that would otherwise allow read_file — prove safety fires FIRST.
    rules = RuleSet(rules=(Rule(tool="read_file", content=None),))
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=rules,
        project_root=tmp_path,
    )
    tool = _tool("read_file", is_read_only=True, args_schema=_PathArgs)
    result = await hook(
        tool=tool,
        args={"path": str(Path.home() / ".ssh" / "id_rsa")},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error == "denied: protected path (safety policy)"
    assert spy.calls == []
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "safety_blocked"


async def test_read_of_git_path_is_not_blocked_by_safety(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    """Reads of .git/ are legitimate (git log etc); reads list does NOT
    include .git/. With a matching rule in the set, the call auto-allows
    via rule_allow."""
    rules = RuleSet(rules=(Rule(tool="read_file", content=None),))
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=rules,
        project_root=tmp_path,
    )
    tool = _tool("read_file", is_read_only=True, args_schema=_PathArgs)
    result = await hook(
        tool=tool,
        args={"path": str(tmp_path / ".git" / "HEAD")},
        state=LoopState(),
    )
    assert result is None
    assert spy.calls == []
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "rule_allow"


async def test_safety_skipped_on_destructive_tool_without_path_arg(
    tmp_path: Path,
) -> None:
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
    )
    # bash-like: destructive, but no args["path"] — safety convention doesn't fire.
    result = await hook(
        tool=_tool("bash", is_destructive=True),
        args={"command": "rm -rf /"},
        state=LoopState(),
    )
    assert result is None
    assert len(spy.calls) == 1  # asker was consulted


# --- rule match and default rules ---


async def test_read_file_rule_matches_goes_to_rule_allow(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    """Passing the built-in default rule for read_file into the RuleSet
    yields ``rule_allow`` — the new path replacing the old ``read_only``
    branch."""
    rules = RuleSet(rules=DEFAULT_ALLOW_RULES)
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=rules,
        project_root=tmp_path,
    )
    tool = _tool("read_file", is_read_only=True, args_schema=_PathArgs)
    result = await hook(
        tool=tool,
        args={"path": str(tmp_path / "ordinary.txt")},
        state=LoopState(),
    )
    assert result is None
    assert spy.calls == []
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "rule_allow"
    assert decision_event[1]["rule"] == "read_file"


async def test_read_file_with_empty_ruleset_goes_to_ask(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    """Empty RuleSet + no session rule + not on safety list → asker is
    consulted even for read_only tools. Proves the old auto-allow branch
    is gone."""
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
    )
    tool = _tool("read_file", is_read_only=True, args_schema=_PathArgs)
    result = await hook(
        tool=tool,
        args={"path": str(tmp_path / "ordinary.txt")},
        state=LoopState(),
    )
    assert result is None
    assert len(spy.calls) == 1
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "user_accept"


async def test_project_rules_match_takes_priority_over_session(
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    spy = _SpyAsker()
    project_rule = Rule(tool="writer", content=None)
    session_rule = Rule(tool="writer", content=None)
    session = SessionRuleSet()
    session.add(session_rule)
    hook = make_permission_hook(
        asker=spy,
        session=session,
        rules=RuleSet(rules=(project_rule,)),
        project_root=Path("/tmp"),
    )
    result = await hook(tool=_tool(), args={}, state=LoopState())
    assert result is None
    assert spy.calls == []
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "rule_allow"
    assert decision_event[1]["rule"] == project_rule.to_string()


async def test_session_rules_match_when_project_misses() -> None:
    spy = _SpyAsker()
    session = SessionRuleSet()
    session.add(Rule(tool="writer", content=None))
    hook = make_permission_hook(
        asker=spy,
        session=session,
        rules=RuleSet(),  # empty
        project_root=Path("/tmp"),
    )
    result = await hook(tool=_tool(), args={}, state=LoopState())
    assert result is None
    assert spy.calls == []


async def test_ask_accept_allows(
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    result = await hook(tool=_tool(), args={}, state=LoopState())
    assert result is None
    assert len(spy.calls) == 1
    assert spy.calls[0]["rule_hint"] == Rule(tool="writer", content=None)
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "user_accept"


async def test_ask_deny_returns_tool_result(
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    spy = _SpyAsker(response=AskerResponse(choice="deny"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    result = await hook(tool=_tool(), args={}, state=LoopState())
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error == "denied: user"
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "user_deny"


async def test_ask_always_session_scope_adds_rule_to_session() -> None:
    rule = Rule(tool="writer", content=None)
    spy = _SpyAsker(
        response=AskerResponse(choice="always", scope="session", rule=rule),
    )
    session = SessionRuleSet()
    hook = make_permission_hook(
        asker=spy,
        session=session,
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    result = await hook(tool=_tool(), args={}, state=LoopState())
    assert result is None
    assert session.rules() == (rule,)


async def test_ask_always_project_scope_calls_save_rule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rule = Rule(tool="writer", content=None)
    save_calls: list[tuple[Path, Rule, str]] = []

    def _fake_save(project_root: Path, r: Rule, *, scope: str = "project") -> None:
        save_calls.append((project_root, r, scope))

    import aura.core.hooks.permission as permission_module

    monkeypatch.setattr(permission_module, "save_rule", _fake_save)

    spy = _SpyAsker(
        response=AskerResponse(choice="always", scope="project", rule=rule),
    )
    session = SessionRuleSet()
    hook = make_permission_hook(
        asker=spy,
        session=session,
        rules=RuleSet(),
        project_root=tmp_path,
    )
    result = await hook(tool=_tool(), args={}, state=LoopState())
    assert result is None
    assert save_calls == [(tmp_path, rule, "project")]
    assert session.rules() == ()


async def test_ask_always_project_save_failure_degrades_to_session(
    monkeypatch: pytest.MonkeyPatch,
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    rule = Rule(tool="writer", content=None)

    def _boom(project_root: Path, r: Rule, *, scope: str = "project") -> None:
        raise PermissionStoreError(source=str(project_root), detail="disk full")

    import aura.core.hooks.permission as permission_module

    monkeypatch.setattr(permission_module, "save_rule", _boom)

    spy = _SpyAsker(
        response=AskerResponse(choice="always", scope="project", rule=rule),
    )
    session = SessionRuleSet()
    hook = make_permission_hook(
        asker=spy,
        session=session,
        rules=RuleSet(),
        project_root=tmp_path,
    )
    result = await hook(tool=_tool(), args={}, state=LoopState())
    assert result is None
    assert session.rules() == (rule,)
    names = [e[0] for e in journal_events]
    assert "permission_save_failed" in names
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "user_always"


async def test_asker_exception_treated_as_deny(
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    spy = _SpyAsker(raise_=RuntimeError("asker crashed"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    result = await hook(tool=_tool(), args={}, state=LoopState())
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error == "denied: user"
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "user_deny"


@pytest.mark.parametrize(
    "exc_type",
    [KeyboardInterrupt, asyncio.CancelledError, SystemExit],
)
async def test_asker_basexception_propagates_does_not_deny(
    exc_type: type[BaseException],
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    spy = _SpyAsker(raise_=exc_type())
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    with pytest.raises(exc_type):
        await hook(tool=_tool(), args={}, state=LoopState())
    assert not any(e[0] == "permission_decision" for e in journal_events)


async def test_hook_stashes_decision_on_state_custom() -> None:
    """The hook writes Decision to state.custom["_aura_pending_decision"]."""
    from aura.core.permissions.decision import Decision

    rules = RuleSet(rules=(Rule(tool="writer", content=None),))
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=rules,
        project_root=Path("/tmp"),
    )
    state = LoopState()
    await hook(tool=_tool(), args={}, state=state)
    stashed = state.custom.get("_aura_pending_decision")
    assert isinstance(stashed, Decision)
    assert stashed.reason == "rule_allow"
    assert stashed.allow is True


async def test_hook_stash_is_overwritten_across_calls() -> None:
    """Single-slot stash is overwritten each call."""
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    rules = RuleSet(rules=(Rule(tool="writer", content=None),))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=rules,
        project_root=Path("/tmp"),
    )
    state = LoopState()
    # First call: rule_allow (writer rule matches).
    await hook(tool=_tool(), args={}, state=state)
    first = state.custom["_aura_pending_decision"]
    assert first.reason == "rule_allow"
    # Second call: different tool, no matching rule → asker answers accept.
    await hook(tool=_tool("different"), args={}, state=state)
    second = state.custom["_aura_pending_decision"]
    assert second.reason == "user_accept"
    assert second is not first


async def test_every_terminal_decision_emits_permission_decision(
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    """Spec §5: every outcome emits a permission_decision event with required keys."""
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    await hook(tool=_tool(), args={}, state=LoopState())
    decision_events = [e for e in journal_events if e[0] == "permission_decision"]
    assert len(decision_events) == 1
    fields = decision_events[0][1]
    assert set(fields.keys()) >= {"tool", "reason", "rule", "mode"}
    assert fields["tool"] == "writer"
    assert fields["mode"] == "default"
    assert fields["rule"] is None


async def test_user_rule_wins_audit_over_default_when_both_match(
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    """If a user-supplied specific rule and a generic default rule could
    both match, RuleSet iterates in tuple order → first-match wins. The
    CLI composes ``disk_rules + defaults`` (user rules first) so the
    user's explicit intent is what lands in the audit trail."""
    from aura.core.permissions.matchers import path_prefix_on
    # Simulate the CLI's composition: user rule first, default tool-wide last.
    user_rule = Rule(tool="read_file", content="/tmp/specific")
    default_rule = Rule(tool="read_file", content=None)
    # _tool() with a path_prefix_on matcher so the user's pattern rule can fire.
    read_tool = _tool("read_file",
                      is_read_only=True,
                      rule_matcher=path_prefix_on("path"),
                      args_schema=_PathArgs)
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(rules=(user_rule, default_rule)),  # user first
        project_root=Path("/tmp"),
    )
    result = await hook(
        tool=read_tool,
        args={"path": "/tmp/specific/foo.txt"},
        state=LoopState(),
    )
    assert result is None  # allowed
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    # The USER's specific rule should be reported — not the generic default.
    assert decision_event[1]["rule"] == user_rule.to_string()
    assert spy.calls == []
