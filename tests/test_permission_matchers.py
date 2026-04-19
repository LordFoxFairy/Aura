"""Tests for aura.core.permissions.matchers — exact_match_on, path_prefix_on."""

from __future__ import annotations

from aura.core.permissions.matchers import exact_match_on, path_prefix_on

# ---------------------------------------------------------------------------
# exact_match_on
# ---------------------------------------------------------------------------


def test_exact_match_returns_true_on_identical_value() -> None:
    m = exact_match_on("command")
    assert m({"command": "npm test"}, "npm test") is True


def test_exact_match_returns_false_on_different_value() -> None:
    m = exact_match_on("command")
    assert m({"command": "rm -rf /"}, "npm test") is False


def test_exact_match_returns_false_on_missing_key() -> None:
    m = exact_match_on("command")
    assert m({}, "npm test") is False


def test_exact_match_returns_false_on_non_string_value() -> None:
    m = exact_match_on("command")
    assert m({"command": 42}, "42") is False


# ---------------------------------------------------------------------------
# path_prefix_on
# ---------------------------------------------------------------------------


def test_path_prefix_matches_exact_path() -> None:
    m = path_prefix_on("path")
    assert m({"path": "/tmp/foo.txt"}, "/tmp/foo.txt") is True


def test_path_prefix_matches_descendant() -> None:
    m = path_prefix_on("path")
    assert m({"path": "/tmp/sub/foo.txt"}, "/tmp") is True


def test_path_prefix_does_not_match_sibling_with_shared_prefix_chars() -> None:
    # Classic bug: "/tmp" must not cover "/tmpfoo". Component boundary matters.
    m = path_prefix_on("path")
    assert m({"path": "/tmpfoo/bar"}, "/tmp") is False


def test_path_prefix_does_not_match_parent() -> None:
    # A rule "/tmp/foo" must NOT allow a call on "/tmp".
    m = path_prefix_on("path")
    assert m({"path": "/tmp"}, "/tmp/foo") is False


def test_path_prefix_handles_relative_paths() -> None:
    m = path_prefix_on("path")
    assert m({"path": "src/app.py"}, "src") is True


def test_path_prefix_returns_false_on_missing_key() -> None:
    m = path_prefix_on("path")
    assert m({}, "/tmp") is False


def test_path_prefix_returns_false_on_non_string_value() -> None:
    m = path_prefix_on("path")
    assert m({"path": 123}, "/tmp") is False
