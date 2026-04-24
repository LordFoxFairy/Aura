"""Per-turn ResolveOnce dedup for the permission hook's ask branch (V12-H).

Verifies: within a single turn, same-signature ``(tool, args)`` invocations
that hit the ask branch prompt the user ONCE. The cached decision is reused
for subsequent same-signature calls in that turn. Turn boundary clears the
cache so the model's next turn still gets a fresh prompt when behavior might
have drifted.

Cache scope is narrow on purpose:
- Only the ``ask`` branch is deduped; mode/safety/rule branches at the top
  of ``_decide`` are cheap + stateless, no benefit to caching.
- Only ``user_accept`` / ``user_deny`` outcomes are cached; ``user_always``
  installs a persistent rule so subsequent calls hit step-5 ``rule_allow``
  without reaching the ask branch at all.
- Asker failures are NOT cached; a transient widget crash shouldn't deny
  every same-signature call for the rest of the turn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks.permission import (
    AskerResponse,
    make_permission_hook,
)
from aura.core.permissions.rule import Rule
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.schemas.state import LoopState
from aura.tools.base import build_tool


class _AnyArgs(BaseModel):
    path: str = ""
    url: str = ""


def _noop() -> dict[str, Any]:
    return {}


def _tool(name: str = "web_fetch") -> BaseTool:
    return build_tool(
        name=name,
        description=name,
        args_schema=_AnyArgs,
        func=_noop,
    )


@dataclass
class _CountingAsker:
    """Asker that records every call and returns a configured response."""

    response: AskerResponse
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Rule,
    ) -> AskerResponse:
        self.calls.append({"tool": tool.name, "args": dict(args)})
        return self.response


@pytest.fixture
def journal_events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []

    def _capture(event: str, /, **fields: Any) -> None:
        events.append((event, fields))

    from aura.core.persistence import journal as journal_mod
    monkeypatch.setattr(journal_mod, "write", _capture)
    return events


def _hook_factory(
    asker: _CountingAsker, project_root: Path,
) -> Callable[[], Any]:
    def _factory() -> Any:
        return make_permission_hook(
            asker=asker,
            session=SessionRuleSet(),
            rules=RuleSet(),
            project_root=project_root,
        )
    return _factory


@pytest.mark.asyncio
async def test_same_signature_accept_cached_within_turn(
    tmp_path: Path, journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    """User accepts ``web_fetch({url: x})`` once; second same call reuses it."""
    asker = _CountingAsker(response=AskerResponse(choice="accept"))
    hook = _hook_factory(asker, tmp_path)()
    state = LoopState()
    tool = _tool("web_fetch")
    args = {"url": "https://example.com"}

    out1 = await hook(tool=tool, args=args, state=state)
    out2 = await hook(tool=tool, args=args, state=state)

    assert len(asker.calls) == 1
    assert out1.decision is not None
    assert out2.decision is not None
    assert out1.decision.allow is True
    assert out2.decision.allow is True
    assert out2.decision.reason == "user_accept"
    dedup_events = [e for e in journal_events if e[0] == "permission_dedup_hit"]
    assert len(dedup_events) == 1
    assert dedup_events[0][1]["tool"] == "web_fetch"
    assert dedup_events[0][1]["reason"] == "user_accept"


@pytest.mark.asyncio
async def test_same_signature_deny_cached_within_turn(tmp_path: Path) -> None:
    """User denies ``web_fetch({url: x})`` once; second same call stays denied."""
    asker = _CountingAsker(response=AskerResponse(choice="deny"))
    hook = _hook_factory(asker, tmp_path)()
    state = LoopState()
    tool = _tool("web_fetch")
    args = {"url": "https://example.com"}

    out1 = await hook(tool=tool, args=args, state=state)
    out2 = await hook(tool=tool, args=args, state=state)

    assert len(asker.calls) == 1
    assert out1.decision is not None
    assert out2.decision is not None
    assert out1.decision.allow is False
    assert out2.decision.allow is False
    assert out2.decision.reason == "user_deny"
    # Both calls populate the denials sink via their own _hook branch.


@pytest.mark.asyncio
async def test_different_args_prompt_separately(tmp_path: Path) -> None:
    """Same tool, different args → two asker calls (distinct signatures)."""
    asker = _CountingAsker(response=AskerResponse(choice="accept"))
    hook = _hook_factory(asker, tmp_path)()
    state = LoopState()
    tool = _tool("web_fetch")

    await hook(tool=tool, args={"url": "https://a.com"}, state=state)
    await hook(tool=tool, args={"url": "https://b.com"}, state=state)

    assert len(asker.calls) == 2


@pytest.mark.asyncio
async def test_different_tool_prompts_separately(tmp_path: Path) -> None:
    """Same args shape, different tool → two asker calls (distinct signatures)."""
    asker = _CountingAsker(response=AskerResponse(choice="accept"))
    hook = _hook_factory(asker, tmp_path)()
    state = LoopState()

    tool_a = _tool("custom_a")
    tool_b = _tool("custom_b")
    args = {"url": "https://example.com"}

    await hook(tool=tool_a, args=args, state=state)
    await hook(tool=tool_b, args=args, state=state)

    assert len(asker.calls) == 2


@pytest.mark.asyncio
async def test_args_order_independent_cache_key(tmp_path: Path) -> None:
    """Dict key order must NOT affect cache identity — canonical JSON hash."""
    asker = _CountingAsker(response=AskerResponse(choice="accept"))
    hook = _hook_factory(asker, tmp_path)()
    state = LoopState()
    tool = _tool("custom")

    await hook(tool=tool, args={"url": "x", "path": "y"}, state=state)
    await hook(tool=tool, args={"path": "y", "url": "x"}, state=state)

    assert len(asker.calls) == 1


@pytest.mark.asyncio
async def test_cache_cleared_when_state_custom_reset(tmp_path: Path) -> None:
    """Simulates turn boundary — loop pops the cache key, next same-sig call
    re-prompts. Matches ``AgentLoop.run_turn``'s per-turn reset semantics."""
    asker = _CountingAsker(response=AskerResponse(choice="accept"))
    hook = _hook_factory(asker, tmp_path)()
    state = LoopState()
    tool = _tool("web_fetch")
    args = {"url": "https://example.com"}

    await hook(tool=tool, args=args, state=state)
    # Simulate what run_turn does at the start of the next turn.
    state.custom.pop("_perm_turn_ask_cache", None)
    await hook(tool=tool, args=args, state=state)

    assert len(asker.calls) == 2


@pytest.mark.asyncio
async def test_asker_failure_not_cached(
    tmp_path: Path, journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    """A transient asker exception should NOT lock-in a deny for the rest of
    the turn — the next same-signature call must be allowed to prompt."""
    calls: list[dict[str, Any]] = []

    class _FlakyAsker:
        def __init__(self) -> None:
            self.invocations = 0

        async def __call__(
            self,
            *,
            tool: BaseTool,
            args: dict[str, Any],
            rule_hint: Rule,
        ) -> AskerResponse:
            self.invocations += 1
            calls.append({"tool": tool.name, "args": dict(args)})
            if self.invocations == 1:
                raise RuntimeError("widget crashed")
            return AskerResponse(choice="accept")

    asker = _FlakyAsker()
    hook = make_permission_hook(
        asker=asker,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
    )
    state = LoopState()
    tool = _tool("custom")
    args = {"url": "https://example.com"}

    out1 = await hook(tool=tool, args=args, state=state)
    out2 = await hook(tool=tool, args=args, state=state)

    assert asker.invocations == 2
    assert out1.decision is not None
    assert out1.decision.allow is False  # first call: asker failure → deny
    assert out2.decision is not None
    assert out2.decision.allow is True  # second call: fresh prompt succeeded


@pytest.mark.asyncio
async def test_always_decision_not_stored_in_dedup_cache(tmp_path: Path) -> None:
    """``always`` installs a rule as the persistent source of truth — the
    dedup cache deliberately does NOT stash the decision, so there's no
    shadow copy to go stale if the rule is later revoked. Exercises the
    contract that only ``user_accept`` / ``user_deny`` outcomes populate
    the cache (see permission.py ``_decide`` step 6)."""
    asker = _CountingAsker(
        response=AskerResponse(
            choice="always",
            rule=Rule(tool="custom", content=None),
        ),
    )
    hook = make_permission_hook(
        asker=asker,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
    )
    state = LoopState()
    tool = _tool("custom")
    args = {"url": "https://example.com"}

    out = await hook(tool=tool, args=args, state=state)

    assert out.decision is not None
    assert out.decision.reason == "user_always"
    # Cache should be created (by the setdefault in _decide) but the
    # ``always`` branch must NOT have inserted this signature into it.
    cache = state.custom.get("_perm_turn_ask_cache", {})
    assert not cache, f"expected empty cache, got {cache!r}"
