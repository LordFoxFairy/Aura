"""Tests for aura.core.skills.registry.SkillRegistry."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.core.skills.registry import SkillRegistry
from aura.core.skills.types import Skill


def _skill(name: str, *, desc: str = "d", body: str = "b") -> Skill:
    return Skill(
        name=name,
        description=desc,
        body=body,
        source_path=Path(f"/tmp/{name}.md"),
        layer="user",
    )


def test_register_then_get() -> None:
    reg = SkillRegistry()
    s = _skill("foo")
    reg.register(s)
    assert reg.get("foo") is s
    assert reg.get("nope") is None


def test_duplicate_name_rejected() -> None:
    reg = SkillRegistry()
    reg.register(_skill("dup"))
    with pytest.raises(ValueError, match="dup"):
        reg.register(_skill("dup"))


def test_list_returns_sorted_by_name() -> None:
    reg = SkillRegistry()
    reg.register(_skill("zeta"))
    reg.register(_skill("alpha"))
    reg.register(_skill("mu"))
    assert [s.name for s in reg.list()] == ["alpha", "mu", "zeta"]


def test_empty_registry_list_is_empty() -> None:
    reg = SkillRegistry()
    assert reg.list() == []


def test_init_from_iterable() -> None:
    a = _skill("a")
    b = _skill("b")
    reg = SkillRegistry([a, b])
    assert [s.name for s in reg.list()] == ["a", "b"]
