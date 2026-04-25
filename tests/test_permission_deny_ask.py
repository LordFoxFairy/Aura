"""F-04-001: deny / ask rule tests.

Covers the precedence ladder ``deny > ask > allow > default`` and the
new bypass-immune behavior of deny rules. Pairs with ``test_permission.py``
which still owns the safety / mode / allow rule branches; this file
focuses on the new layers.
"""

from __future__ import annotations

import json
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
from aura.core.permissions.store import (
    load,
    load_ask_ruleset,
    load_deny_ruleset,
)
from aura.schemas.permissions import PermissionsConfig
from aura.schemas.state import LoopState
from aura.tools.base import build_tool


class _P(BaseModel):
    pass


def _noop() -> dict[str, Any]:
    return {}


def _tool(
    name: str = "bash",
    *,
    rule_matcher: Callable[[dict[str, Any], str], bool] | None = None,
    args_schema: type[BaseModel] = _P,
) -> BaseTool:
    return build_tool(
        name=name,
        description=name,
        args_schema=args_schema,
        func=_noop,
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
        self.calls.append({"tool": tool.name, "args": args, "rule_hint": rule_hint})
        assert self.response is not None
        return self.response


@pytest.fixture
def journal_events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []

    def _capture(event: str, /, **fields: Any) -> None:
        events.append((event, fields))

    from aura.core.persistence import journal as journal_mod

    monkeypatch.setattr(journal_mod, "write", _capture)
    return events


# ---------------------------------------------------------------------------
# Schema — backward compatibility + new fields
# ---------------------------------------------------------------------------


def test_permissions_config_deny_ask_default_to_empty_lists() -> None:
    cfg = PermissionsConfig()
    assert cfg.deny == []
    assert cfg.ask == []


def test_permissions_config_accepts_deny_and_ask() -> None:
    cfg = PermissionsConfig(
        deny=["bash(rm:*)"],
        ask=["bash(git push:*)"],
    )
    assert cfg.deny == ["bash(rm:*)"]
    assert cfg.ask == ["bash(git push:*)"]


def test_load_settings_with_only_allow_does_not_break(tmp_path: Path) -> None:
    """Backward compat: a settings.json from before deny/ask shipped must
    keep loading clean — deny/ask default to []."""
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"permissions": {"allow": ["bash"]}}))
    cfg = load(tmp_path)
    assert cfg.allow == ["bash"]
    assert cfg.deny == []
    assert cfg.ask == []


def test_load_concatenates_deny_and_ask_across_project_and_local(
    tmp_path: Path,
) -> None:
    project = tmp_path / ".aura" / "settings.json"
    local = tmp_path / ".aura" / "settings.local.json"
    project.parent.mkdir()
    project.write_text(json.dumps({
        "permissions": {
            "deny": ["bash(rm:*)"],
            "ask": ["bash(git push:*)"],
        },
    }))
    local.write_text(json.dumps({
        "permissions": {
            "deny": ["bash(curl:*)"],
            "ask": ["read_file(/etc/passwd)"],
        },
    }))
    cfg = load(tmp_path)
    # Project first, local appended — same shape as allow.
    assert cfg.deny == ["bash(rm:*)", "bash(curl:*)"]
    assert cfg.ask == ["bash(git push:*)", "read_file(/etc/passwd)"]


# ---------------------------------------------------------------------------
# Loader — load_deny_ruleset / load_ask_ruleset
# ---------------------------------------------------------------------------


def test_load_deny_ruleset_parses_kind_deny(tmp_path: Path) -> None:
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({
        "permissions": {"deny": ["bash(rm:*)"]},
    }))
    rs = load_deny_ruleset(tmp_path)
    assert len(rs.rules) == 1
    assert rs.rules[0].kind == "deny"
    assert rs.rules[0].tool == "bash"


def test_load_ask_ruleset_parses_kind_ask(tmp_path: Path) -> None:
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({
        "permissions": {"ask": ["bash(git push:*)"]},
    }))
    rs = load_ask_ruleset(tmp_path)
    assert len(rs.rules) == 1
    assert rs.rules[0].kind == "ask"


def test_invalid_rule_pattern_in_deny_logs_and_skips(
    tmp_path: Path,
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    """F-04-001 contract: malformed deny entries journal + skip,
    they DO NOT raise. A typo in deny must not crash startup — safety
    + asker still cover the floor."""
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({
        "permissions": {
            "deny": ["bash(unclosed", "bash(npm publish)"],
        },
    }))
    rs = load_deny_ruleset(tmp_path)
    # Bad rule skipped, good rule loaded.
    assert len(rs.rules) == 1
    assert rs.rules[0].content == "npm publish"
    parse_events = [
        e for e in journal_events if e[0] == "permission_rule_parse_failed"
    ]
    assert len(parse_events) == 1
    assert parse_events[0][1]["kind"] == "deny"
    assert parse_events[0][1]["rule"] == "bash(unclosed"


def test_invalid_rule_pattern_in_ask_logs_and_skips(
    tmp_path: Path,
    journal_events: list[tuple[str, dict[str, Any]]],
) -> None:
    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({
        "permissions": {"ask": ["bash(unclosed"]},
    }))
    rs = load_ask_ruleset(tmp_path)
    assert rs.rules == ()
    parse_events = [
        e for e in journal_events if e[0] == "permission_rule_parse_failed"
    ]
    assert len(parse_events) == 1
    assert parse_events[0][1]["kind"] == "ask"


# ---------------------------------------------------------------------------
# Decision pipeline — deny rules
# ---------------------------------------------------------------------------


async def test_deny_rule_blocks_under_default_mode(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    """Baseline: a deny rule blocks even the simplest call."""
    from aura.core.permissions.matchers import exact_match_on

    deny_rules = RuleSet(rules=(
        Rule(tool="bash", content="rm -rf /tmp/x", kind="deny"),
    ))
    spy = _SpyAsker()
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        deny_rules=deny_rules,
        project_root=tmp_path,
    )
    bash_tool = _tool("bash", rule_matcher=exact_match_on("command"))
    outcome = await hook(
        tool=bash_tool,
        args={"command": "rm -rf /tmp/x"},
        state=LoopState(),
    )
    assert outcome.short_circuit is not None
    assert outcome.short_circuit.ok is False
    assert outcome.short_circuit.error is not None
    assert "deny rule" in outcome.short_circuit.error
    assert outcome.decision is not None
    assert outcome.decision.reason == "rule_deny"
    assert outcome.decision.allow is False
    # Asker must NOT be consulted — deny rules short-circuit hard.
    assert spy.calls == []


async def test_deny_rule_blocks_even_under_bypass(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    """F-04-001 + F-04-006: deny rules are bypass-immune. ``mode=bypass``
    is the user's "skip the prompt UX" consent, not a license to run a
    forever-forbidden command."""
    from aura.core.permissions.matchers import exact_match_on

    deny_rules = RuleSet(rules=(
        Rule(tool="bash", content="rm -rf /tmp/x", kind="deny"),
    ))
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        deny_rules=deny_rules,
        project_root=tmp_path,
        mode="bypass",
    )
    outcome = await hook(
        tool=_tool("bash", rule_matcher=exact_match_on("command")),
        args={"command": "rm -rf /tmp/x"},
        state=LoopState(),
    )
    assert outcome.short_circuit is not None
    assert outcome.decision is not None
    assert outcome.decision.reason == "rule_deny"
    # ``permission_bypass`` MUST NOT fire — deny short-circuited first.
    assert "permission_bypass" not in [e[0] for e in journal_events]


async def test_deny_overrides_allow_when_both_match(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    """``deny > allow``. If the same pattern is in both lists, deny wins."""
    from aura.core.permissions.matchers import exact_match_on

    pattern = "rm -rf /tmp/x"
    allow_rules = RuleSet(rules=(
        Rule(tool="bash", content=pattern, kind="allow"),
    ))
    deny_rules = RuleSet(rules=(
        Rule(tool="bash", content=pattern, kind="deny"),
    ))
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=allow_rules,
        deny_rules=deny_rules,
        project_root=tmp_path,
    )
    outcome = await hook(
        tool=_tool("bash", rule_matcher=exact_match_on("command")),
        args={"command": pattern},
        state=LoopState(),
    )
    assert outcome.decision is not None
    assert outcome.decision.reason == "rule_deny"
    assert outcome.decision.allow is False


async def test_journal_records_rule_pattern_and_kind_for_deny(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    """``permission_decision`` event surfaces the matched rule string +
    its kind so audit consumers can filter without re-parsing."""
    from aura.core.permissions.matchers import exact_match_on

    deny_rule = Rule(tool="bash", content="rm -rf /tmp/x", kind="deny")
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        deny_rules=RuleSet(rules=(deny_rule,)),
        project_root=tmp_path,
    )
    await hook(
        tool=_tool("bash", rule_matcher=exact_match_on("command")),
        args={"command": "rm -rf /tmp/x"},
        state=LoopState(),
    )
    decision_event = next(e for e in journal_events if e[0] == "permission_decision")
    assert decision_event[1]["reason"] == "rule_deny"
    assert decision_event[1]["rule"] == deny_rule.to_string()
    assert decision_event[1]["rule_kind"] == "deny"


# ---------------------------------------------------------------------------
# Decision pipeline — ask rules
# ---------------------------------------------------------------------------


async def test_ask_rule_forces_prompt_when_allow_rule_would_match(
    tmp_path: Path,
) -> None:
    """``ask > allow``. A sibling allow rule that would have auto-allowed
    is overridden — the asker is consulted instead."""
    from aura.core.permissions.matchers import exact_match_on

    pattern = "git push origin main"
    allow_rules = RuleSet(rules=(
        Rule(tool="bash", content=pattern, kind="allow"),
    ))
    ask_rules = RuleSet(rules=(
        Rule(tool="bash", content=pattern, kind="ask"),
    ))
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=allow_rules,
        ask_rules=ask_rules,
        project_root=tmp_path,
    )
    outcome = await hook(
        tool=_tool("bash", rule_matcher=exact_match_on("command")),
        args={"command": pattern},
        state=LoopState(),
    )
    assert outcome.short_circuit is None
    # Asker WAS consulted — the allow rule was overridden by the ask rule.
    assert len(spy.calls) == 1
    assert outcome.decision is not None
    assert outcome.decision.reason == "user_accept"


async def test_ask_rule_forces_prompt_even_after_session_allow(
    tmp_path: Path,
) -> None:
    """The user said "always" once → session has the rule. A sibling ask
    rule pattern still forces every subsequent call to re-prompt."""
    from aura.core.permissions.matchers import exact_match_on

    pattern = "git push origin main"
    session = SessionRuleSet()
    session.add(Rule(tool="bash", content=pattern, kind="allow"))
    ask_rules = RuleSet(rules=(
        Rule(tool="bash", content=pattern, kind="ask"),
    ))
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=session,
        rules=RuleSet(),
        ask_rules=ask_rules,
        project_root=tmp_path,
    )
    outcome = await hook(
        tool=_tool("bash", rule_matcher=exact_match_on("command")),
        args={"command": pattern},
        state=LoopState(),
    )
    assert outcome.short_circuit is None
    assert len(spy.calls) == 1


async def test_ask_rule_bypasses_per_turn_dedup_cache(
    tmp_path: Path,
) -> None:
    """The dedup cache silences repeat prompts within a turn for non-ask
    paths. Ask rules MUST bypass the cache — the contract is "always
    prompt", and that includes within a turn."""
    from aura.core.permissions.matchers import exact_match_on

    pattern = "git push origin main"
    ask_rules = RuleSet(rules=(
        Rule(tool="bash", content=pattern, kind="ask"),
    ))
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        ask_rules=ask_rules,
        project_root=tmp_path,
    )
    state = LoopState()
    bash_tool = _tool("bash", rule_matcher=exact_match_on("command"))
    args = {"command": pattern}
    # Two identical calls — both must hit the asker because the rule says "ask".
    await hook(tool=bash_tool, args=args, state=state)
    await hook(tool=bash_tool, args=args, state=state)
    assert len(spy.calls) == 2


async def test_ask_rule_emits_audit_event(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    """A ``permission_ask_rule_forced`` journal event records the rule
    pattern that promoted the call to the prompt path — audit consumers
    can see WHY a previously allow-ruled tool started asking again."""
    from aura.core.permissions.matchers import exact_match_on

    ask_rule = Rule(tool="bash", content="git push origin main", kind="ask")
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        ask_rules=RuleSet(rules=(ask_rule,)),
        project_root=tmp_path,
    )
    await hook(
        tool=_tool("bash", rule_matcher=exact_match_on("command")),
        args={"command": "git push origin main"},
        state=LoopState(),
    )
    forced = [e for e in journal_events if e[0] == "permission_ask_rule_forced"]
    assert len(forced) == 1
    assert forced[0][1]["rule"] == ask_rule.to_string()


# ---------------------------------------------------------------------------
# Precedence — full ladder deny > ask > allow > default
# ---------------------------------------------------------------------------


async def test_deny_overrides_ask_when_both_match(
    journal_events: list[tuple[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    """``deny > ask``. Even if the same pattern is on the ask list, deny
    must win — the call is hard-blocked, not prompted."""
    from aura.core.permissions.matchers import exact_match_on

    pattern = "rm -rf /tmp/x"
    deny_rules = RuleSet(rules=(
        Rule(tool="bash", content=pattern, kind="deny"),
    ))
    ask_rules = RuleSet(rules=(
        Rule(tool="bash", content=pattern, kind="ask"),
    ))
    spy = _SpyAsker(response=AskerResponse(choice="accept"))
    hook = make_permission_hook(
        asker=spy,
        session=SessionRuleSet(),
        rules=RuleSet(),
        deny_rules=deny_rules,
        ask_rules=ask_rules,
        project_root=tmp_path,
    )
    outcome = await hook(
        tool=_tool("bash", rule_matcher=exact_match_on("command")),
        args={"command": pattern},
        state=LoopState(),
    )
    assert outcome.decision is not None
    assert outcome.decision.reason == "rule_deny"
    assert spy.calls == []  # never prompted — deny short-circuits


async def test_no_deny_no_ask_falls_through_to_allow_path(
    tmp_path: Path,
) -> None:
    """Sanity: when neither deny nor ask matches, the existing allow
    pipeline is unchanged."""
    from aura.core.permissions.matchers import exact_match_on

    allow_rules = RuleSet(rules=(
        Rule(tool="bash", content="ls -la", kind="allow"),
    ))
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=allow_rules,
        deny_rules=RuleSet(),
        ask_rules=RuleSet(),
        project_root=tmp_path,
    )
    outcome = await hook(
        tool=_tool("bash", rule_matcher=exact_match_on("command")),
        args={"command": "ls -la"},
        state=LoopState(),
    )
    assert outcome.decision is not None
    assert outcome.decision.reason == "rule_allow"
    assert outcome.decision.allow is True


# ---------------------------------------------------------------------------
# End-to-end via store loader — confirms the wiring
# ---------------------------------------------------------------------------


async def test_e2e_settings_json_deny_rule_blocks_call(
    tmp_path: Path,
) -> None:
    """Settings.json -> load_deny_ruleset -> hook produces rule_deny."""
    from aura.core.permissions.matchers import exact_match_on

    settings = tmp_path / ".aura" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({
        "permissions": {"deny": ["bash(npm publish)"]},
    }))
    deny_rules = load_deny_ruleset(tmp_path)
    hook = make_permission_hook(
        asker=_SpyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        deny_rules=deny_rules,
        project_root=tmp_path,
    )
    outcome = await hook(
        tool=_tool("bash", rule_matcher=exact_match_on("command")),
        args={"command": "npm publish"},
        state=LoopState(),
    )
    assert outcome.decision is not None
    assert outcome.decision.reason == "rule_deny"
