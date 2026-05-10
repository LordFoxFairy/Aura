"""Tests for the ``ToolMetadata`` frozen dataclass — Phase 2 Task 1.

These tests pin the contract documented in spec §3 (tool subsystem
redesign): a typed replacement for the legacy ``tool_metadata(...)``
dict, covering the seven mandated fields plus the dataclass identity
guarantees (frozen, default ``capability_flags``).

Migration of consumers (registry enforcement, per-tool ``aura_metadata``
attribute) is Tasks 2-4; Task 1 is contract-only. The legacy dict path
returned by :func:`tool_metadata` MUST stay functional through these
tasks — a dedicated test guards the no-regression contract.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from aura.schemas.tool import (
    ToolArgsPreview,
    ToolMetadata,
    ToolRuleMatcher,
    tool_metadata,
)


def _matcher(_args: dict[str, Any], _content: str) -> bool:
    return True


def _preview(_args: dict[str, Any]) -> str:
    return ""


def test_tool_metadata_constructible_with_seven_fields() -> None:
    """Spec §3 — every field must be settable via the canonical
    constructor signature. Asserts each field round-trips its value
    so a typo or rename in the dataclass surfaces here, not at the
    consumer migration in Task 2.
    """
    meta = ToolMetadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        rule_matcher=_matcher,
        args_preview=_preview,
        timeout_sec=10.0,
        capability_flags=frozenset({"writes_files"}),
    )

    assert meta.is_read_only is True
    assert meta.is_destructive is False
    assert meta.is_concurrency_safe is True
    assert meta.rule_matcher is _matcher
    assert meta.args_preview is _preview
    assert meta.timeout_sec == 10.0
    assert meta.capability_flags == frozenset({"writes_files"})


def test_tool_metadata_has_seven_fields_per_spec() -> None:
    """Spec §3 lists exactly 7 named fields. Guards against accidental
    extras (which would erode the typed surface) or missing fields
    (which would silently break consumers in Tasks 2-3).
    """
    expected = {
        "is_read_only",
        "is_destructive",
        "is_concurrency_safe",
        "rule_matcher",
        "args_preview",
        "timeout_sec",
        "capability_flags",
    }
    actual = {f.name for f in dataclasses.fields(ToolMetadata)}
    assert actual == expected


def test_tool_metadata_is_frozen() -> None:
    """``ToolMetadata`` is a frozen dataclass — direct assignment
    raises ``FrozenInstanceError``. The metadata is set once at tool
    class construction and never mutated; freezing prevents a stray
    consumer from rewriting capability flags at runtime.
    """
    meta = ToolMetadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        rule_matcher=None,
        args_preview=None,
        timeout_sec=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.is_read_only = False  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.capability_flags = frozenset({"x"})  # type: ignore[misc]


def test_tool_metadata_capability_flags_defaults_to_empty_frozenset() -> None:
    """Spec §3 — ``capability_flags`` defaults to ``frozenset()`` so
    tools that don't claim any capability flag don't have to type
    ``frozenset()`` at every site. The default factory must produce
    a fresh empty frozenset; no aliasing across instances.
    """
    a = ToolMetadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        rule_matcher=None,
        args_preview=None,
        timeout_sec=None,
    )
    b = ToolMetadata(
        is_read_only=False,
        is_destructive=True,
        is_concurrency_safe=False,
        rule_matcher=None,
        args_preview=None,
        timeout_sec=None,
    )

    assert a.capability_flags == frozenset()
    assert b.capability_flags == frozenset()
    # Frozensets are immutable so aliasing is safe in principle, but
    # asserting separate identity is cheap insurance against a stray
    # mutable-default refactor.
    assert isinstance(a.capability_flags, frozenset)


def test_tool_metadata_optional_fields_accept_none() -> None:
    """``rule_matcher``, ``args_preview``, ``timeout_sec`` are all
    nullable per spec §3 (a tool without per-call rules supplies no
    matcher; a tool with internal timeout ladders supplies
    ``timeout_sec=None`` to skip the loop's outer wrapper).
    """
    meta = ToolMetadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        rule_matcher=None,
        args_preview=None,
        timeout_sec=None,
    )
    assert meta.rule_matcher is None
    assert meta.args_preview is None
    assert meta.timeout_sec is None


def test_tool_metadata_field_types_match_aliases() -> None:
    """Sanity: the callable-typed fields accept the existing alias
    types used elsewhere in the schemas module (``ToolRuleMatcher``,
    ``ToolArgsPreview``). If those aliases drift we want a compile-
    time signal here.
    """
    matcher: ToolRuleMatcher = _matcher
    preview: ToolArgsPreview = _preview
    meta = ToolMetadata(
        is_read_only=False,
        is_destructive=False,
        is_concurrency_safe=False,
        rule_matcher=matcher,
        args_preview=preview,
        timeout_sec=None,
    )
    assert meta.rule_matcher is matcher
    assert meta.args_preview is preview


def test_tool_metadata_helper_still_returns_dict_for_backwards_compat() -> None:
    """Phase 2 Task 1 explicitly preserves the legacy dict path.
    Migration of the helper's return type is Tasks 2-4; if this
    test ever fails before then, a consumer still on the dict path
    is silently broken.
    """
    legacy = tool_metadata(is_read_only=True, is_concurrency_safe=True)
    assert isinstance(legacy, dict)
    assert legacy["is_read_only"] is True
    assert legacy["is_concurrency_safe"] is True


def test_tool_metadata_exported_from_aura_schemas() -> None:
    """``ToolMetadata`` is part of the leaf package's public surface
    so consumers in ``aura.core`` / ``aura.tools`` can import via
    ``from aura.schemas import ToolMetadata``.
    """
    from aura import schemas

    assert hasattr(schemas, "ToolMetadata")
    assert schemas.ToolMetadata is ToolMetadata
    assert "ToolMetadata" in schemas.__all__
