"""Tests for ``aura.core.permissions.rule_hint.derive_rule_hint`` — spec §8.2.

``derive_rule_hint`` returns the most specific pattern rule the CLI can show
on prompt option 2, or ``None`` when the tool/args can't support a pattern
rule. The caller (CLI) owns the fallback (tool-wide + session scope).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from aura.core.permissions.matchers import exact_match_on, path_prefix_on
from aura.core.permissions.rule import Rule
from aura.core.permissions.rule_hint import derive_rule_hint
from aura.tools.base import build_tool


class _P(BaseModel):
    pass


def _noop() -> dict[str, Any]:
    return {}


def _tool_with_matcher(
    name: str, matcher: Any = None,
) -> Any:
    return build_tool(
        name=name,
        description=name,
        args_schema=_P,
        func=_noop,
        rule_matcher=matcher,
    )


def test_bash_command_arg_yields_pattern_rule() -> None:
    tool = _tool_with_matcher("bash", exact_match_on("command"))
    hint = derive_rule_hint(tool, {"command": "npm test"})
    assert hint == Rule(tool="bash", content="npm test")


def test_read_file_path_arg_yields_pattern_rule() -> None:
    tool = _tool_with_matcher("read_file", path_prefix_on("path"))
    hint = derive_rule_hint(tool, {"path": "/tmp/a"})
    assert hint == Rule(tool="read_file", content="/tmp/a")


def test_tool_without_rule_matcher_returns_none() -> None:
    tool = _tool_with_matcher("todo_write", None)
    assert derive_rule_hint(tool, {"todos": []}) is None


def test_custom_matcher_without_key_attr_returns_none() -> None:
    # Hand-rolled matcher (not from matchers module) has no .key attribute
    # → helper can't know which arg to inspect → returns None.
    def custom(args: dict[str, Any], content: str) -> bool:
        return True

    tool = _tool_with_matcher("custom", custom)
    assert derive_rule_hint(tool, {"command": "npm test"}) is None


def test_args_missing_key_returns_none() -> None:
    tool = _tool_with_matcher("bash", exact_match_on("command"))
    assert derive_rule_hint(tool, {}) is None


def test_args_empty_string_returns_none() -> None:
    tool = _tool_with_matcher("bash", exact_match_on("command"))
    assert derive_rule_hint(tool, {"command": ""}) is None


def test_args_non_string_value_returns_none() -> None:
    tool = _tool_with_matcher("bash", exact_match_on("command"))
    assert derive_rule_hint(tool, {"command": 42}) is None


def test_tool_with_no_metadata_returns_none() -> None:
    # Defensive: metadata can be None or missing the key — helper must not crash.
    class _Bare:
        name = "bare"
        metadata = None

    assert derive_rule_hint(_Bare(), {"command": "x"}) is None  # type: ignore[arg-type]
