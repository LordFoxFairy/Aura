"""Tests for audit findings F-04-002 / F-04-005 / F-04-015 / F-04-020."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.config.schema import AuraConfigError
from aura.core.hooks import (
    HookChain,
    PreToolOutcome,
)
from aura.core.hooks.permission import (
    AskerResponse,
    make_permission_hook,
)
from aura.core.permissions.rule import Rule
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.schemas.state import LoopState
from aura.tools.base import build_tool


class _P(BaseModel):
    pass


class _PathArgs(BaseModel):
    path: str


class _UrlArgs(BaseModel):
    url: str


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


# ---------------------------------------------------------------------------
# F-04-002 — PreToolHook returning ``ask`` overrides prior allow → next
# pipeline step is the prompt path (asker invoked).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_f_04_002_ask_demotes_rule_allow_to_asker_call(
    tmp_path: Path,
) -> None:
    """A foreign hook that returns ``ask=True`` must force the permission
    hook to invoke the asker even when a default-allow rule would
    otherwise auto-allow the call."""
    asker = _SpyAsker(response=AskerResponse(choice="accept", scope="session"))
    perm_hook = make_permission_hook(
        asker=asker,
        session=SessionRuleSet(),
        # Default-allow rule that would normally auto-allow read_file.
        rules=RuleSet(rules=(Rule(tool="read_file", content=None),)),
        project_root=tmp_path,
        mode="default",
    )

    async def upstream_ask(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return PreToolOutcome(ask=True)

    chain = HookChain(pre_tool=[upstream_ask, perm_hook])
    tool = _tool("read_file", is_read_only=True, args_schema=_PathArgs)
    state = LoopState()
    outcome = await chain.run_pre_tool(
        tool=tool, args={"path": str(tmp_path / "x.txt")}, state=state,
    )
    # Asker WAS called — auto-allow was demoted.
    assert len(asker.calls) == 1
    # Final decision reflects the asker's accept response.
    assert outcome.decision is not None
    assert outcome.decision.allow is True
    assert outcome.decision.reason == "user_accept"


@pytest.mark.asyncio
async def test_f_04_002_ask_demotes_bypass_to_asker_call(
    tmp_path: Path,
) -> None:
    """Even ``mode='bypass'`` must yield to a downstream ask — the ask
    channel is the operator's "I really need a fresh confirmation"
    override."""
    asker = _SpyAsker(response=AskerResponse(choice="deny", scope="session"))
    perm_hook = make_permission_hook(
        asker=asker,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="bypass",
    )

    async def upstream_ask(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return PreToolOutcome(ask=True)

    chain = HookChain(pre_tool=[upstream_ask, perm_hook])
    tool = _tool("bash", is_destructive=True)
    outcome = await chain.run_pre_tool(
        tool=tool, args={}, state=LoopState(),
    )
    assert len(asker.calls) == 1
    assert outcome.decision is not None
    assert outcome.decision.allow is False
    assert outcome.decision.reason == "user_deny"


@pytest.mark.asyncio
async def test_f_04_002_ask_does_not_override_safety_block(
    tmp_path: Path,
) -> None:
    """An upstream ask must NOT promote a safety-blocked deny back to a
    prompt — safety stays absolute."""
    asker = _SpyAsker(response=AskerResponse(choice="accept", scope="session"))
    perm_hook = make_permission_hook(
        asker=asker,
        session=SessionRuleSet(),
        rules=RuleSet(rules=(Rule(tool="read_file", content=None),)),
        project_root=tmp_path,
        mode="default",
    )

    async def upstream_ask(
        *, tool: BaseTool, args: dict[str, Any], state: LoopState, **_: object,
    ) -> PreToolOutcome:
        return PreToolOutcome(ask=True)

    chain = HookChain(pre_tool=[upstream_ask, perm_hook])
    tool = _tool("read_file", is_read_only=True, args_schema=_PathArgs)
    outcome = await chain.run_pre_tool(
        tool=tool,
        args={"path": str(Path.home() / ".ssh" / "id_rsa")},
        state=LoopState(),
    )
    # Asker NEVER called — safety wins.
    assert asker.calls == []
    assert outcome.decision is not None
    assert outcome.decision.reason == "safety_blocked"


# ---------------------------------------------------------------------------
# F-04-005 — plan-mode + web_fetch → denied (web_fetch removed from
# read-allow list because plan mode = no outbound calls).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_f_04_005_plan_mode_blocks_web_fetch(tmp_path: Path) -> None:
    asker = _SpyAsker()
    hook = make_permission_hook(
        asker=asker,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="plan",
    )
    tool = _tool("web_fetch", args_schema=_UrlArgs)
    outcome = await hook(
        tool=tool, args={"url": "https://example.com"}, state=LoopState(),
    )
    assert outcome.short_circuit is not None
    assert outcome.short_circuit.ok is False
    assert outcome.decision is not None
    assert outcome.decision.reason == "plan_mode_blocked"
    assert asker.calls == []


@pytest.mark.asyncio
async def test_f_04_005_plan_mode_blocks_web_search(tmp_path: Path) -> None:
    asker = _SpyAsker()
    hook = make_permission_hook(
        asker=asker,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode="plan",
    )
    tool = _tool("web_search")
    outcome = await hook(
        tool=tool, args={}, state=LoopState(),
    )
    assert outcome.short_circuit is not None
    assert outcome.short_circuit.ok is False
    assert outcome.decision is not None
    assert outcome.decision.reason == "plan_mode_blocked"


def test_f_04_005_plan_mode_read_tools_excludes_outbound() -> None:
    from aura.core.hooks.permission import _PLAN_MODE_READ_TOOLS

    assert "web_fetch" not in _PLAN_MODE_READ_TOOLS
    assert "web_search" not in _PLAN_MODE_READ_TOOLS
    # Local read tools still allowed.
    assert "read_file" in _PLAN_MODE_READ_TOOLS
    assert "grep" in _PLAN_MODE_READ_TOOLS
    assert "glob" in _PLAN_MODE_READ_TOOLS


# ---------------------------------------------------------------------------
# F-04-015 — disable_bypass=True + runtime mode="bypass" → clamped to
# default, warning emitted.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_f_04_015_disable_bypass_clamps_runtime_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    def _capture(event: str, /, **fields: Any) -> None:
        events.append((event, fields))

    from aura.core.persistence import journal as journal_mod

    monkeypatch.setattr(journal_mod, "write", _capture)

    asker = _SpyAsker(response=AskerResponse(choice="accept", scope="session"))
    # Provider returns "bypass" — simulating a runtime mode-set that
    # would otherwise auto-allow everything.
    hook = make_permission_hook(
        asker=asker,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode=lambda: "bypass",
        disable_bypass=True,
    )
    tool = _tool("bash", is_destructive=True)
    outcome = await hook(
        tool=tool, args={}, state=LoopState(),
    )
    # The hook must NOT have taken the bypass auto-allow path; instead
    # it falls through to the asker (clamped to default mode).
    assert len(asker.calls) == 1
    assert outcome.decision is not None
    assert outcome.decision.reason != "mode_bypass"
    assert outcome.decision.reason == "user_accept"

    # The clamp must journal a warning the first time it fires.
    clamp_events = [e for e in events if e[0] == "bypass_clamped"]
    assert len(clamp_events) == 1
    assert clamp_events[0][1]["requested"] == "bypass"
    assert clamp_events[0][1]["effective"] == "default"
    assert clamp_events[0][1]["reason"] == "disable_bypass"


@pytest.mark.asyncio
async def test_f_04_015_disable_bypass_warning_is_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    def _capture(event: str, /, **fields: Any) -> None:
        events.append((event, fields))

    from aura.core.persistence import journal as journal_mod

    monkeypatch.setattr(journal_mod, "write", _capture)

    asker = _SpyAsker(response=AskerResponse(choice="accept", scope="session"))
    hook = make_permission_hook(
        asker=asker,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode=lambda: "bypass",
        disable_bypass=True,
    )
    tool = _tool("bash", is_destructive=True)
    state = LoopState()
    await hook(tool=tool, args={}, state=state)
    await hook(tool=tool, args={"x": 1}, state=state)
    await hook(tool=tool, args={"x": 2}, state=state)
    # One warning emission across the three calls.
    clamp_events = [e for e in events if e[0] == "bypass_clamped"]
    assert len(clamp_events) == 1


@pytest.mark.asyncio
async def test_f_04_015_disable_bypass_false_lets_bypass_through(
    tmp_path: Path,
) -> None:
    """Regression guard: with the kill switch off, ``mode='bypass'``
    still produces ``mode_bypass`` auto-allow (no clamp)."""
    asker = _SpyAsker()
    hook = make_permission_hook(
        asker=asker,
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=tmp_path,
        mode=lambda: "bypass",
        disable_bypass=False,
    )
    tool = _tool("bash", is_destructive=True)
    outcome = await hook(
        tool=tool, args={}, state=LoopState(),
    )
    assert outcome.decision is not None
    assert outcome.decision.reason == "mode_bypass"
    assert asker.calls == []


# ---------------------------------------------------------------------------
# F-04-020 — safety_exempt=["~/.ssh/**"] at config load → AuraConfigError.
# ---------------------------------------------------------------------------


def test_f_04_020_safety_exempt_overlaps_ssh_raises(tmp_path: Path) -> None:
    from aura.core.permissions.store import load

    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({
        "permissions": {"safety_exempt": ["~/.ssh/**"]},
    }))
    with pytest.raises(AuraConfigError) as exc_info:
        load(tmp_path)
    msg = str(exc_info.value)
    assert "safety_exempt" in msg
    assert "~/.ssh/**" in msg
    # Error must name a built-in protected entry the user tried to disarm.
    assert ".ssh" in msg


def test_f_04_020_safety_exempt_overlaps_git_raises(tmp_path: Path) -> None:
    from aura.core.permissions.store import load

    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({
        "permissions": {"safety_exempt": ["**/.git/**"]},
    }))
    with pytest.raises(AuraConfigError) as exc_info:
        load(tmp_path)
    assert "safety_exempt" in str(exc_info.value)
    assert ".git" in str(exc_info.value)


def test_f_04_020_safety_exempt_overlaps_etc_raises(tmp_path: Path) -> None:
    from aura.core.permissions.store import load

    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({
        "permissions": {"safety_exempt": ["/etc/**"]},
    }))
    with pytest.raises(AuraConfigError) as exc_info:
        load(tmp_path)
    assert "/etc" in str(exc_info.value)


def test_f_04_020_safety_exempt_overlaps_bashrc_raises(tmp_path: Path) -> None:
    from aura.core.permissions.store import load

    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({
        "permissions": {"safety_exempt": ["~/.bashrc"]},
    }))
    with pytest.raises(AuraConfigError) as exc_info:
        load(tmp_path)
    assert ".bashrc" in str(exc_info.value)


def test_f_04_020_safety_exempt_unrelated_path_allowed(tmp_path: Path) -> None:
    """Non-overlapping safety_exempt entries continue to load normally."""
    from aura.core.permissions.store import load

    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({
        "permissions": {
            "safety_exempt": ["project/build/**", "tmp/scratch.json"],
        },
    }))
    cfg = load(tmp_path)
    assert cfg.safety_exempt == ["project/build/**", "tmp/scratch.json"]


def test_f_04_020_safety_exempt_local_file_attribution(tmp_path: Path) -> None:
    """When the offending pattern lives in settings.local.json the
    error message must point at THAT file, not settings.json."""
    from aura.core.permissions.store import load

    aura_dir = tmp_path / ".aura"
    aura_dir.mkdir()
    (aura_dir / "settings.json").write_text(json.dumps({
        "permissions": {"safety_exempt": ["build/**"]},
    }))
    (aura_dir / "settings.local.json").write_text(json.dumps({
        "permissions": {"safety_exempt": ["~/.ssh/**"]},
    }))
    with pytest.raises(AuraConfigError) as exc_info:
        load(tmp_path)
    assert "settings.local.json" in str(exc_info.value)
