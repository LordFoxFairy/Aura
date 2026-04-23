"""Tests for ``aura.schemas.tool`` metadata resolvers.

Covers the input-aware ``is_destructive`` / ``is_read_only`` path added
to match claude-code's ``isDestructive(input)`` method pattern. Each
test names the concrete failure mode it guards — a regression here
would silently misclassify every bash invocation.
"""

from __future__ import annotations

from typing import Any

import pytest

from aura.schemas.tool import (
    resolve_is_destructive,
    resolve_is_read_only,
    tool_metadata,
)

# ---------------------------------------------------------------------------
# resolve_is_destructive
# ---------------------------------------------------------------------------


def test_resolve_is_destructive_none_metadata_returns_false() -> None:
    # No metadata at all → no claim → not destructive. Prevents a tool
    # without metadata from being spuriously routed through the
    # protected_writes list.
    assert resolve_is_destructive(None, {}) is False


def test_resolve_is_destructive_static_true() -> None:
    meta = tool_metadata(is_destructive=True)
    assert resolve_is_destructive(meta, {}) is True


def test_resolve_is_destructive_static_false() -> None:
    meta = tool_metadata(is_destructive=False)
    assert resolve_is_destructive(meta, {}) is False


def test_resolve_is_destructive_missing_key_returns_false() -> None:
    # Empty dict passed directly — no ``is_destructive`` key. Must
    # default to False, same as the None-metadata path.
    assert resolve_is_destructive({}, {}) is False


def test_resolve_is_destructive_callable_returns_classifier_result() -> None:
    def classifier(args: dict[str, Any]) -> bool:
        return bool(args.get("command", "").startswith("rm"))

    meta = tool_metadata(is_destructive=classifier)
    # Static flag lookup would see the function object as truthy and
    # return True for EVERY invocation. The resolver must actually call
    # the classifier and honour its per-args answer.
    assert resolve_is_destructive(meta, {"command": "rm -rf /"}) is True
    assert resolve_is_destructive(meta, {"command": "ls"}) is False


def test_resolve_is_destructive_callable_receives_args_verbatim() -> None:
    captured: list[dict[str, Any]] = []

    def classifier(args: dict[str, Any]) -> bool:
        captured.append(args)
        return False

    meta = tool_metadata(is_destructive=classifier)
    resolve_is_destructive(meta, {"command": "echo hi", "timeout": 5})
    assert captured == [{"command": "echo hi", "timeout": 5}]


def test_resolve_is_destructive_callable_exception_fails_safe_true() -> None:
    def broken(_args: dict[str, Any]) -> bool:
        raise RuntimeError("classifier bug")

    meta = tool_metadata(is_destructive=broken)
    # Fail-safe direction: an unclassifiable command must be treated as
    # destructive. Better to re-prompt the user than to route past the
    # protected_writes list.
    assert resolve_is_destructive(meta, {}) is True


def test_resolve_is_destructive_static_false_then_callable_true_resolves_correctly() -> None:
    # Guards against a refactor that conflates the "raw bool" branch and
    # the "callable" branch. Build two metadatas differing ONLY in the
    # is_destructive slot — the resolver must return different bools.
    static_meta = tool_metadata(is_destructive=False)
    callable_meta = tool_metadata(is_destructive=lambda _args: True)
    assert resolve_is_destructive(static_meta, {}) is False
    assert resolve_is_destructive(callable_meta, {}) is True


def test_resolve_is_destructive_truthy_non_bool_coerces() -> None:
    # Hypothetical legacy caller passes a truthy int. ``bool()`` cast
    # is the documented contract — guard against someone "optimizing"
    # it to a raw return.
    meta = {"is_destructive": 1}
    assert resolve_is_destructive(meta, {}) is True


# ---------------------------------------------------------------------------
# resolve_is_read_only
# ---------------------------------------------------------------------------


def test_resolve_is_read_only_none_metadata_returns_false() -> None:
    assert resolve_is_read_only(None, {}) is False


def test_resolve_is_read_only_static_true() -> None:
    meta = tool_metadata(is_read_only=True)
    assert resolve_is_read_only(meta, {}) is True


def test_resolve_is_read_only_callable_returns_classifier_result() -> None:
    def classifier(args: dict[str, Any]) -> bool:
        return bool(args.get("command", "").startswith("cat"))

    meta = tool_metadata(is_read_only=classifier)
    assert resolve_is_read_only(meta, {"command": "cat /etc/hosts"}) is True
    assert resolve_is_read_only(meta, {"command": "rm -rf /"}) is False


def test_resolve_is_read_only_callable_exception_fails_closed_false() -> None:
    # For is_read_only, the fail-safe direction is False — a tool that
    # CAN'T prove it's read-only should not be treated as read-only.
    # Contrast with is_destructive, where ambiguity ≙ True.
    def broken(_args: dict[str, Any]) -> bool:
        raise ValueError("boom")

    meta = tool_metadata(is_read_only=broken)
    assert resolve_is_read_only(meta, {}) is False


# ---------------------------------------------------------------------------
# tool_metadata accepts both shapes
# ---------------------------------------------------------------------------


def test_tool_metadata_accepts_static_bool() -> None:
    meta = tool_metadata(is_destructive=True, is_read_only=False)
    assert meta["is_destructive"] is True
    assert meta["is_read_only"] is False


def test_tool_metadata_accepts_callable() -> None:
    def classifier(_args: dict[str, Any]) -> bool:
        return False

    meta = tool_metadata(is_destructive=classifier, is_read_only=classifier)
    # The raw slot stores the callable — it's the caller's job to go
    # through the resolver. Direct lookups are the exact foot-gun the
    # resolver exists to prevent.
    assert meta["is_destructive"] is classifier
    assert meta["is_read_only"] is classifier


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("rm -rf /", True),
        ("ls", False),
    ],
)
def test_resolve_is_destructive_integrates_with_tool_metadata(
    command: str, expected: bool,
) -> None:
    """End-to-end: tool_metadata(...) → resolve_is_destructive(metadata, args).

    Mirrors how the permission hook calls the resolver in production.
    """
    def classifier(args: dict[str, Any]) -> bool:
        return bool(args.get("command", "").startswith("rm"))

    meta = tool_metadata(is_destructive=classifier)
    assert resolve_is_destructive(meta, {"command": command}) is expected
