"""Tests for Workstream G5 — structured ``PermissionDenial`` + SDK surface.

Covers the three acceptance criteria from
``docs/specs/2026-04-23-aura-main-channel-parity.md`` § Workstream G5:

- AC-G5-1: denials accumulate within a turn — every deny path
  (``safety_blocked`` / ``plan_mode_blocked`` / ``user_deny``) populates
  a :class:`PermissionDenial` reachable via
  :meth:`Agent.last_turn_denials`.
- AC-G5-2: turn-start clears — turn N+1 opens with an empty list even
  when turn N accumulated denials.
- AC-G5-3: the exposed view is read-only — consumer mutation attempts
  raise ``TypeError`` (tuple is the chosen immutable container).

The hook-level assertions exercise every ``_decide`` deny branch
individually so a future refactor that forgets to populate one of them
fails here instead of silently dropping audit data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.hooks import HookChain
from aura.core.hooks.permission import (
    AskerResponse,
    make_permission_hook,
)
from aura.core.permissions.denials import DENIALS_SINK_KEY, PermissionDenial
from aura.core.permissions.rule import Rule
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.core.persistence.storage import SessionStorage
from aura.schemas.state import LoopState
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel, FakeTurn

# -----------------------------------------------------------------------
# helpers (mirror test_permission.py patterns so the two files read alike)
# -----------------------------------------------------------------------


class _PathArgs(BaseModel):
    path: str


class _NoArgs(BaseModel):
    pass


def _noop() -> dict[str, Any]:
    return {}


def _tool(
    name: str = "writer",
    *,
    is_read_only: bool = False,
    is_destructive: bool = False,
    rule_matcher: Callable[[dict[str, Any], str], bool] | None = None,
    args_schema: type[BaseModel] = _NoArgs,
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
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Rule,
    ) -> AskerResponse:
        self.calls.append({"tool": tool.name, "args": args})
        assert self.response is not None
        return self.response


# -----------------------------------------------------------------------
# PermissionDenial dataclass — shape + immutability
# -----------------------------------------------------------------------


def test_permission_denial_is_frozen() -> None:
    denial = PermissionDenial(
        tool_name="read_file",
        tool_use_id="tc_1",
        tool_input={"path": "/etc/passwd"},
        reason="safety_blocked",
        target="/etc/passwd",
    )
    with pytest.raises(FrozenInstanceError):
        denial.reason = "user_deny"  # type: ignore[misc]


def test_permission_denial_default_timestamp_is_tz_aware() -> None:
    denial = PermissionDenial(
        tool_name="x", tool_use_id="id", tool_input={}, reason="user_deny",
    )
    # UTC-aware so the SDK consumer can compare/serialize safely.
    assert denial.timestamp.tzinfo is not None


# -----------------------------------------------------------------------
# Hook deny branches — every non-allow path populates the sink
# -----------------------------------------------------------------------


async def test_hook_safety_blocked_appends_denial_to_sink() -> None:
    """AC-G5-1 (safety branch): _decide safety_blocked → sink has 1 entry."""
    sink: list[PermissionDenial] = []
    state = LoopState(custom={DENIALS_SINK_KEY: sink})
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    offending = str(Path.home() / ".ssh" / "id_rsa")
    await hook(
        tool=_tool("read_file", is_read_only=True, args_schema=_PathArgs),
        args={"path": offending},
        state=state,
        tool_call_id="tc_safety_1",
    )
    assert len(sink) == 1
    entry = sink[0]
    assert entry.tool_name == "read_file"
    assert entry.reason == "safety_blocked"
    assert entry.target == offending
    assert entry.tool_use_id == "tc_safety_1"
    assert entry.tool_input == {"path": offending}


async def test_hook_plan_mode_blocked_appends_denial_to_sink() -> None:
    """AC-G5-1 (plan-mode branch)."""
    sink: list[PermissionDenial] = []
    state = LoopState(custom={DENIALS_SINK_KEY: sink})
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
        mode="plan",
    )
    await hook(
        tool=_tool("writer"),
        args={"payload": "foo"},
        state=state,
        tool_call_id="tc_plan_1",
    )
    assert len(sink) == 1
    entry = sink[0]
    assert entry.reason == "plan_mode_blocked"
    assert entry.tool_name == "writer"
    assert entry.target is None  # plan_mode carries no path
    assert entry.tool_use_id == "tc_plan_1"


async def test_hook_user_deny_appends_denial_to_sink() -> None:
    """AC-G5-1 (user-deny branch via asker)."""
    sink: list[PermissionDenial] = []
    state = LoopState(custom={DENIALS_SINK_KEY: sink})
    hook = make_permission_hook(
        asker=_SpyAsker(response=AskerResponse(choice="deny")),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    await hook(
        tool=_tool("writer"),
        args={},
        state=state,
        tool_call_id="tc_user_1",
    )
    assert len(sink) == 1
    entry = sink[0]
    assert entry.reason == "user_deny"
    assert entry.tool_use_id == "tc_user_1"


async def test_hook_allow_does_not_append_denial_to_sink() -> None:
    """Allow paths leave the sink untouched (else G5 would double-count)."""
    sink: list[PermissionDenial] = []
    state = LoopState(custom={DENIALS_SINK_KEY: sink})
    hook = make_permission_hook(
        asker=_SpyAsker(response=AskerResponse(choice="accept")),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    outcome = await hook(
        tool=_tool("writer"),
        args={},
        state=state,
        tool_call_id="tc_allow_1",
    )
    assert outcome.short_circuit is None
    assert sink == []


async def test_hook_without_sink_key_is_safe_noop() -> None:
    """Absence of ``_aura_denials_sink`` (e.g. unit test bypassing Loop
    wiring) must not raise — the sink is a best-effort surface."""
    state = LoopState()
    hook = make_permission_hook(
        asker=_SpyAsker(response=AskerResponse(choice="deny")),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    # No DENIALS_SINK_KEY in state.custom — must not crash.
    outcome = await hook(tool=_tool(), args={}, state=state)
    assert outcome.short_circuit is not None  # still a deny
    assert DENIALS_SINK_KEY not in state.custom


async def test_hook_copies_tool_input_defensively() -> None:
    """Post-decision mutation of the caller's ``args`` must not bleed
    into the already-captured denial record (snapshot semantics)."""
    sink: list[PermissionDenial] = []
    state = LoopState(custom={DENIALS_SINK_KEY: sink})
    hook = make_permission_hook(
        asker=_SpyAsker(response=AskerResponse(choice="deny")),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path("/tmp"),
    )
    args: dict[str, Any] = {"k": "v"}
    await hook(
        tool=_tool(),
        args=args,
        state=state,
        tool_call_id="tc_copy",
    )
    args["k"] = "mutated"
    assert sink[0].tool_input == {"k": "v"}


# -----------------------------------------------------------------------
# AC-G5-1 / AC-G5-2 / AC-G5-3 — end-to-end via Agent.astream
# -----------------------------------------------------------------------


def _minimal_config(enabled: list[str]) -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": enabled},
    })


def _ai_with_tool_calls(calls: list[ToolCall]) -> AIMessage:
    return AIMessage(content="", tool_calls=calls)


async def _collect(agent: Agent, prompt: str) -> list[Any]:
    events: list[Any] = []
    async for event in agent.astream(prompt):
        events.append(event)
    return events


def _plan_mode_hooks(project_root: Path) -> HookChain:
    """Build a HookChain whose permission hook denies every tool via
    ``plan`` mode. Gives us three deterministic denies per turn without
    needing to wire an asker or a SafetyPolicy."""
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=project_root,
        mode="plan",
    )
    return HookChain(pre_tool=[hook])


async def test_last_turn_denials_accumulates_within_turn(tmp_path: Path) -> None:
    """AC-G5-1: three denials in a single turn → len == 3, fields correct.

    Scripted model emits three tool_calls in a single AIMessage; plan-mode
    permission hook denies each one (tool names fall outside the plan
    allow-list). After the turn completes, Agent.last_turn_denials()
    surfaces all three in order.
    """
    model = FakeChatModel(turns=[
        FakeTurn(_ai_with_tool_calls([
            {"name": "write_file", "args": {"path": "a.txt", "content": "x"},
             "id": "tc_1", "type": "tool_call"},
            {"name": "write_file", "args": {"path": "b.txt", "content": "y"},
             "id": "tc_2", "type": "tool_call"},
            {"name": "write_file", "args": {"path": "c.txt", "content": "z"},
             "id": "tc_3", "type": "tool_call"},
        ])),
        # After the tool round, the model gives a plain reply so the
        # turn naturally terminates.
        FakeTurn(AIMessage(content="done")),
    ])
    storage = SessionStorage(tmp_path / "aura.db")
    agent = Agent(
        config=_minimal_config(["write_file"]),
        model=model,
        storage=storage,
        hooks=_plan_mode_hooks(tmp_path),
    )
    await _collect(agent, "go")
    denials = agent.last_turn_denials()
    assert len(denials) == 3
    assert [d.tool_use_id for d in denials] == ["tc_1", "tc_2", "tc_3"]
    assert all(d.reason == "plan_mode_blocked" for d in denials)
    assert all(d.tool_name == "write_file" for d in denials)
    # tool_input captured at decision time
    assert denials[0].tool_input == {"path": "a.txt", "content": "x"}
    assert denials[2].tool_input == {"path": "c.txt", "content": "z"}


async def test_last_turn_denials_resets_between_turns(tmp_path: Path) -> None:
    """AC-G5-2: turn 2 opens with an empty list; next deny lands alone."""
    model = FakeChatModel(turns=[
        # Turn 1: two deny tool calls then a final assistant message.
        FakeTurn(_ai_with_tool_calls([
            {"name": "write_file", "args": {"path": "a.txt", "content": ""},
             "id": "t1_1", "type": "tool_call"},
            {"name": "write_file", "args": {"path": "b.txt", "content": ""},
             "id": "t1_2", "type": "tool_call"},
        ])),
        FakeTurn(AIMessage(content="t1 done")),
        # Turn 2: one deny tool call then a final assistant message.
        FakeTurn(_ai_with_tool_calls([
            {"name": "write_file", "args": {"path": "c.txt", "content": ""},
             "id": "t2_1", "type": "tool_call"},
        ])),
        FakeTurn(AIMessage(content="t2 done")),
    ])
    storage = SessionStorage(tmp_path / "aura.db")
    agent = Agent(
        config=_minimal_config(["write_file"]),
        model=model,
        storage=storage,
        hooks=_plan_mode_hooks(tmp_path),
    )
    await _collect(agent, "turn 1")
    assert len(agent.last_turn_denials()) == 2
    await _collect(agent, "turn 2")
    denials = agent.last_turn_denials()
    assert len(denials) == 1
    assert denials[0].tool_use_id == "t2_1"


async def test_last_turn_denials_is_readonly_view(tmp_path: Path) -> None:
    """AC-G5-3: external mutation attempts raise TypeError (tuple)."""
    model = FakeChatModel(turns=[
        FakeTurn(_ai_with_tool_calls([
            {"name": "write_file", "args": {"path": "a.txt", "content": ""},
             "id": "tc", "type": "tool_call"},
        ])),
        FakeTurn(AIMessage(content="done")),
    ])
    storage = SessionStorage(tmp_path / "aura.db")
    agent = Agent(
        config=_minimal_config(["write_file"]),
        model=model,
        storage=storage,
        hooks=_plan_mode_hooks(tmp_path),
    )
    await _collect(agent, "go")
    view = agent.last_turn_denials()
    assert len(view) == 1
    # tuple rejects item assignment AND append
    with pytest.raises(TypeError):
        view[0] = None  # type: ignore[index]
    with pytest.raises(AttributeError):
        view.append(None)  # type: ignore[attr-defined]


async def test_last_turn_denials_empty_before_any_turn(tmp_path: Path) -> None:
    """Defensive: calling before any turn returns the empty view without
    blowing up (common SDK pattern: inspect before first astream call)."""
    storage = SessionStorage(tmp_path / "aura.db")
    agent = Agent(
        config=_minimal_config(["write_file"]),
        model=FakeChatModel(turns=[]),
        storage=storage,
    )
    assert agent.last_turn_denials() == ()
