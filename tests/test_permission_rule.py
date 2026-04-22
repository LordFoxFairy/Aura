"""Tests for aura.core.permissions.rule — Rule parse, to_string, matches."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.permissions.rule import InvalidRuleError, Rule
from aura.errors import AuraError
from aura.tools.base import build_tool


def _fake_tool(
    name: str,
    *,
    rule_matcher: Callable[[dict[str, Any], str], bool] | None = None,
) -> BaseTool:
    """Minimal BaseTool for matcher tests; optionally carries ``rule_matcher``
    in its metadata dict (the slot ``Rule.matches`` consults)."""

    class _P(BaseModel):
        pass

    tool = build_tool(name=name, description=name, args_schema=_P, func=lambda: {})
    if rule_matcher is not None:
        assert tool.metadata is not None
        tool.metadata["rule_matcher"] = rule_matcher
    return tool


def test_parse_bare_tool_name_yields_tool_wide_rule() -> None:
    assert Rule.parse("bash") == Rule(tool="bash", content=None)


def test_parse_paren_content_splits_tool_and_content() -> None:
    assert Rule.parse("bash(npm test)") == Rule(tool="bash", content="npm test")


def test_parse_escaped_parens_in_content() -> None:
    assert Rule.parse(r"bash(echo \(x\))") == Rule(tool="bash", content="echo (x)")


def test_parse_escaped_backslash_in_content() -> None:
    assert Rule.parse(r"bash(path\\sep)") == Rule(tool="bash", content=r"path\sep")


def test_to_string_tool_wide() -> None:
    assert Rule(tool="bash", content=None).to_string() == "bash"


def test_to_string_with_simple_content() -> None:
    assert Rule(tool="bash", content="npm test").to_string() == "bash(npm test)"


def test_to_string_escapes_parens_and_backslash() -> None:
    # Backslashes must be escaped before parens, else the escape for ( leaks.
    assert Rule(tool="bash", content=r"a\(b)c").to_string() == r"bash(a\\\(b\)c)"


def test_parse_to_string_round_trip() -> None:
    for raw in [
        "bash",
        "bash(npm test)",
        r"bash(echo \(x\))",
        r"bash(path\\sep)",
        "read_file(/tmp)",
    ]:
        assert Rule.parse(raw).to_string() == raw


def test_parse_empty_string_raises() -> None:
    with pytest.raises(InvalidRuleError) as exc_info:
        Rule.parse("")
    assert "empty" in str(exc_info.value).lower()


def test_parse_missing_close_paren_raises() -> None:
    with pytest.raises(InvalidRuleError) as exc_info:
        Rule.parse("bash(unclosed")
    assert "unclosed" in str(exc_info.value).lower() or "close" in str(exc_info.value).lower()


def test_parse_trailing_chars_after_close_raises() -> None:
    with pytest.raises(InvalidRuleError) as exc_info:
        Rule.parse("bash(abc)xyz")
    assert "trailing" in str(exc_info.value).lower() or "xyz" in str(exc_info.value)


def test_parse_missing_tool_name_raises() -> None:
    with pytest.raises(InvalidRuleError):
        Rule.parse("(content)")


def test_invalid_rule_error_is_aura_error_subclass() -> None:
    assert issubclass(InvalidRuleError, AuraError)


# ---------------------------------------------------------------------------
# Rule.matches
# ---------------------------------------------------------------------------


def test_matches_returns_false_on_tool_name_mismatch() -> None:
    rule = Rule(tool="bash", content=None)
    assert rule.matches("read_file", {}, _fake_tool("read_file")) is False


def test_tool_wide_rule_matches_any_args_without_consulting_matcher() -> None:
    # Tool-wide rule (content=None) short-circuits before touching any matcher.
    rule = Rule(tool="bash", content=None)
    tool = _fake_tool("bash")  # no rule_matcher on this tool
    assert rule.matches("bash", {"command": "rm -rf /"}, tool) is True


def test_pattern_rule_delegates_to_tool_rule_matcher() -> None:
    rule = Rule(tool="bash", content="npm test")
    seen: list[tuple[dict[str, Any], str]] = []

    def matcher(args: dict[str, Any], content: str) -> bool:
        seen.append((args, content))
        return args.get("command") == content

    tool = _fake_tool("bash", rule_matcher=matcher)
    assert rule.matches("bash", {"command": "npm test"}, tool) is True
    assert seen == [({"command": "npm test"}, "npm test")]


def test_pattern_rule_on_tool_without_matcher_returns_false() -> None:
    # Conservative default: a pattern rule cannot fire on a tool whose class
    # never declared how to match arg patterns. Prevents accidental allow.
    rule = Rule(tool="bash", content="npm test")
    tool = _fake_tool("bash")  # no rule_matcher
    assert rule.matches("bash", {"command": "npm test"}, tool) is False


def test_pattern_rule_returns_false_when_matcher_rejects() -> None:
    rule = Rule(tool="bash", content="npm test")
    tool = _fake_tool("bash", rule_matcher=lambda _args, _content: False)
    assert rule.matches("bash", {"command": "npm test"}, tool) is False


# ---------------------------------------------------------------------------
# Wildcard tool-name matching — MCP use case (tool names namespaced per server)
# ---------------------------------------------------------------------------


def test_wildcard_matches_mcp_namespaced_tool_name() -> None:
    # `allow: ["mcp__github__*"]` was silently not working before the
    # fnmatch path because Rule.matches did exact equality on tool name.
    # Regression guard: make sure wildcard coverage does match.
    rule = Rule(tool="mcp__github__*", content=None)
    tool = _fake_tool("mcp__github__issue_search")
    assert rule.matches("mcp__github__issue_search", {}, tool) is True


def test_wildcard_does_not_match_different_server() -> None:
    rule = Rule(tool="mcp__github__*", content=None)
    tool = _fake_tool("mcp__gitlab__issue_search")
    assert rule.matches("mcp__gitlab__issue_search", {}, tool) is False


def test_wildcard_pattern_rule_still_requires_matcher() -> None:
    # A wildcard rule with content (pattern rule) still needs rule_matcher
    # on the tool. Conservative: missing matcher → reject.
    rule = Rule(tool="mcp__github__*", content="foo")
    tool = _fake_tool("mcp__github__tool")  # no rule_matcher metadata
    assert rule.matches("mcp__github__tool", {"x": 1}, tool) is False


def test_non_wildcard_rule_unchanged_by_fnmatch_path() -> None:
    # Regression guard: the wildcard branch must not alter existing exact
    # matching for rules that don't contain "*".
    rule = Rule(tool="bash", content=None)
    tool = _fake_tool("bash")
    assert rule.matches("bash", {}, tool) is True
    assert rule.matches("grep", {}, tool) is False


def test_single_star_wildcard_matches_any_tool() -> None:
    # Edge case: "*" alone covers every tool. Useful in test fixtures but
    # users are unlikely to write it at the rule level.
    rule = Rule(tool="*", content=None)
    assert rule.matches("bash", {}, _fake_tool("bash")) is True
    assert rule.matches("mcp__anything__tool", {}, _fake_tool("mcp__anything__tool")) is True
