"""Tests for aura.core.permissions.session — RuleSet + SessionRuleSet."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.permissions.rule import Rule
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.tools.base import build_tool


def _fake_tool(
    name: str,
    *,
    rule_matcher: Callable[[dict[str, Any], str], bool] | None = None,
) -> BaseTool:
    class _P(BaseModel):
        pass

    tool = build_tool(name=name, description=name, args_schema=_P, func=lambda: {})
    if rule_matcher is not None:
        assert tool.metadata is not None
        tool.metadata["rule_matcher"] = rule_matcher
    return tool


def test_empty_ruleset_matches_nothing() -> None:
    rs = RuleSet()
    assert rs.matches("bash", {}, _fake_tool("bash")) is None


def test_ruleset_returns_first_matching_rule_in_order() -> None:
    # Three rules all for "bash"; tool-wide rules all match. First-in-order wins.
    r1 = Rule(tool="bash", content=None)
    r2 = Rule(tool="bash", content=None)
    r3 = Rule(tool="bash", content=None)
    rs = RuleSet(rules=(r1, r2, r3))
    matched = rs.matches("bash", {"command": "ls"}, _fake_tool("bash"))
    assert matched is r1


def test_ruleset_skips_non_matching_rules_to_find_the_match() -> None:
    # r1 is for read_file (no match), r2 is for bash tool-wide (matches).
    r1 = Rule(tool="read_file", content=None)
    r2 = Rule(tool="bash", content=None)
    r3 = Rule(tool="bash", content=None)
    rs = RuleSet(rules=(r1, r2, r3))
    matched = rs.matches("bash", {}, _fake_tool("bash"))
    assert matched is r2


def test_session_ruleset_add_makes_matches_find_the_rule() -> None:
    s = SessionRuleSet()
    assert s.matches("bash", {}, _fake_tool("bash")) is None
    rule = Rule(tool="bash", content=None)
    s.add(rule)
    assert s.matches("bash", {}, _fake_tool("bash")) is rule


def test_session_ruleset_add_is_idempotent_on_rule_equality() -> None:
    s = SessionRuleSet()
    r1 = Rule(tool="bash", content="npm test")
    r2 = Rule(tool="bash", content="npm test")  # equal by value
    assert r1 == r2
    s.add(r1)
    s.add(r2)
    assert len(s.rules()) == 1


def test_session_ruleset_rules_returns_immutable_tuple_snapshot() -> None:
    s = SessionRuleSet()
    s.add(Rule(tool="bash", content=None))
    snapshot = s.rules()
    assert isinstance(snapshot, tuple)
    # Snapshot is decoupled from future mutations.
    s.add(Rule(tool="read_file", content=None))
    assert len(snapshot) == 1
    assert len(s.rules()) == 2


def test_session_ruleset_clear_drops_all_rules() -> None:
    s = SessionRuleSet()
    rule = Rule(tool="bash", content="npm test")
    s.add(rule)
    assert s.matches("bash", {"command": "npm test"}, _fake_tool(
        "bash", rule_matcher=lambda args, content: args.get("command") == content,
    )) is rule
    s.clear()
    assert s.matches("bash", {"command": "npm test"}, _fake_tool(
        "bash", rule_matcher=lambda args, content: args.get("command") == content,
    )) is None


def test_session_ruleset_clear_leaves_rules_empty_tuple() -> None:
    s = SessionRuleSet()
    s.add(Rule(tool="bash", content=None))
    s.add(Rule(tool="read_file", content=None))
    s.clear()
    assert s.rules() == ()


def test_session_ruleset_clear_on_empty_is_noop() -> None:
    s = SessionRuleSet()
    s.clear()  # must not raise
    assert s.rules() == ()
