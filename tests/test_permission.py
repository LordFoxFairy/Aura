"""Tests for aura.core.hooks.permission.make_permission_hook (Task 7 rewrite).

Covers the decision flow in spec §5 — order, short-circuits, journal events,
AskerResponse invariants, and the asker-exception-is-deny defensive path.
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
from aura.core.permissions.rule import Rule
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.core.permissions.store import PermissionStoreError
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool


class _P(BaseModel):
    pass


def _noop() -> dict[str, Any]:
    return {}


def _tool(
    name: str = "writer",
    *,
    is_read_only: bool = False,
    is_destructive: bool = False,
    rule_matcher: Callable[[dict[str, Any], str], bool] | None = None,
) -> BaseTool:
    return build_tool(
        name=name,
        description=name,
        args_schema=_P,
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

    import aura.core.hooks.permission as permission_module

    monkeypatch.setattr(permission_module.journal, "write", _capture)
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
    # Runtime-checkable Protocol: an instance with the right callable attribute
    # must satisfy isinstance. The check is structural, not nominal.
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
        tool=_tool(is_destructive=True),
        args={"path": "/etc/passwd"},
        state=LoopState(),
    )
    assert result is None
    assert spy.calls == []
    # Both the loud bypass event AND the generic decision event are emitted.
    names = [e[0] for e in journal_events]
    assert "permission_bypass" in names
    assert "permission_decision" in names
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "mode_bypass"


async def test_safety_fires_before_read_only_checks_and_skips_asker(
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
    # Destructive tool writing to .git path — safety blocks.
    result = await hook(
        tool=_tool(is_destructive=True),
        args={"path": str(tmp_path / ".git" / "HEAD")},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error == "denied: write to protected path (safety policy)"
    assert spy.calls == []
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "safety_blocked"


async def test_safety_skipped_on_read_only_tool_even_with_protected_path(
    tmp_path: Path,
) -> None:
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
    )
    # read_only tool — must not short-circuit through safety.
    result = await hook(
        tool=_tool(is_read_only=True),
        args={"path": str(tmp_path / ".git" / "HEAD")},
        state=LoopState(),
    )
    assert result is None  # read_only allows
    assert spy.calls == []


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


async def test_read_only_tool_allows_without_asking(
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    result = await hook(
        tool=_tool(is_read_only=True), args={}, state=LoopState(),
    )
    assert result is None
    assert spy.calls == []
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "read_only"


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
    # The project rule is the one recorded — identity check via to_string is fine
    # because both rules have identical content; the key insight is it was reached
    # without asker and without touching session-only rules.
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
    # The MVP rule_hint is tool-wide.
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
    # Project scope did NOT add to session — the disk is the store.
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
    # Current call still succeeds — the user consented; disk failure doesn't revoke it.
    assert result is None
    # Degraded to session scope.
    assert session.rules() == (rule,)
    # Failure journaled.
    names = [e[0] for e in journal_events]
    assert "permission_save_failed" in names
    # Decision is still user_always.
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
    # KeyboardInterrupt / CancelledError / SystemExit must propagate so the
    # outer loop (Agent.astream) handles cancellation cleanly. Swallowing
    # them as "user_deny" would wedge Ctrl+C behavior.
    spy = _SpyAsker(raise_=exc_type())
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    with pytest.raises(exc_type):
        await hook(tool=_tool(), args={}, state=LoopState())
    # No decision should have been journaled — the call never terminated.
    assert not any(e[0] == "permission_decision" for e in journal_events)


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
