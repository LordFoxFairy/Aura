"""paths:-conditional skill activation state machine — match, stickiness, path scoping."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aura.domain.skill import Skill
from aura.infrastructure.skills import skills_conditional as sc


@pytest.fixture(autouse=True)
def _isolate_module_state() -> Iterator[None]:
    """Module-global state persists for process lifetime; reset around each test."""
    sc.clear_conditional_state()
    yield
    sc.clear_conditional_state()


def _skill(name: str, paths: tuple[str, ...] = ()) -> Skill:
    return Skill(
        name=name,
        description="d",
        body="b",
        source_path=Path("/skills") / name / "SKILL.md",
        layer="user",
        paths=frozenset(paths),
    )


def test_stash_holds_skill_unactivated(tmp_path: Path) -> None:
    """A stashed conditional skill must stay dormant (not yet matched) until a file touch."""
    sc.stash_conditional(_skill("rust", ("*.rs",)))
    assert [s.name for s in sc.get_conditional_skills()] == ["rust"]
    assert sc.is_activated("rust") is False


def test_activate_with_empty_bucket_returns_empty(tmp_path: Path) -> None:
    assert sc.activate_conditional_skills_for_paths(["a.py"], tmp_path) == []


def test_no_paths_skill_activates_unconditionally(tmp_path: Path) -> None:
    """A conditional skill with no paths: filter must fire on the first activation sweep."""
    sc.stash_conditional(_skill("always"))
    activated = sc.activate_conditional_skills_for_paths([], tmp_path)
    assert activated == ["always"]
    assert sc.is_activated("always") is True
    assert sc.get_conditional_skills() == []  # consumed out of the bucket


def test_paths_skill_activates_on_gitignore_match(tmp_path: Path) -> None:
    sc.stash_conditional(_skill("py", ("*.py",)))
    activated = sc.activate_conditional_skills_for_paths(["pkg/mod.py"], tmp_path)
    assert activated == ["py"]
    assert sc.is_activated("py") is True


def test_paths_skill_stays_stashed_on_no_match(tmp_path: Path) -> None:
    """No matching touch must leave the skill dormant — premature activation pollutes context."""
    sc.stash_conditional(_skill("rust", ("*.rs",)))
    assert sc.activate_conditional_skills_for_paths(["mod.py"], tmp_path) == []
    assert sc.is_activated("rust") is False
    assert [s.name for s in sc.get_conditional_skills()] == ["rust"]


def test_activation_is_sticky_and_idempotent(tmp_path: Path) -> None:
    """Activation sticks for the process; a second matching sweep must not re-yield it."""
    sc.stash_conditional(_skill("py", ("*.py",)))
    first = sc.activate_conditional_skills_for_paths(["a.py"], tmp_path)
    second = sc.activate_conditional_skills_for_paths(["b.py"], tmp_path)
    assert first == ["py"]
    assert second == []  # already consumed, never double-activates
    assert sc.is_activated("py") is True


def test_malformed_pathspec_is_skipped_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skill with an unparseable paths: pattern must be skipped, never crash the sweep."""
    def _boom(*_a: object, **_k: object) -> object:
        raise ValueError("bad gitignore pattern")

    monkeypatch.setattr("pathspec.PathSpec.from_lines", _boom)
    sc.stash_conditional(_skill("broken", ("[",)))
    assert sc.activate_conditional_skills_for_paths(["a.py"], tmp_path) == []
    assert sc.is_activated("broken") is False


@pytest.mark.parametrize(
    ("raw", "inside"),
    [
        ("a.py", True),  # relative, inside cwd
        ("sub/dir/a.py", True),  # relative nested, inside
        ("../escape.py", False),  # relative escaping cwd -> rejected
    ],
)
def test_relative_to_cwd_scopes_to_tree(
    tmp_path: Path, raw: str, inside: bool,
) -> None:
    """Only paths under cwd may key a match; an escaping ../ path must resolve to None."""
    cwd = tmp_path.resolve()
    result = sc._relative_to_cwd(raw, cwd)
    if inside:
        assert result is not None and not result.startswith("..")
    else:
        assert result is None


def test_relative_to_cwd_absolute_inside_and_outside(tmp_path: Path) -> None:
    cwd = tmp_path.resolve()
    inside = str(cwd / "pkg" / "x.py")
    assert sc._relative_to_cwd(inside, cwd) == "pkg/x.py"
    assert sc._relative_to_cwd("/var/elsewhere/y.py", cwd) is None


def test_clear_resets_both_buckets(tmp_path: Path) -> None:
    sc.stash_conditional(_skill("a"))
    sc.activate_conditional_skills_for_paths([], tmp_path)
    sc.stash_conditional(_skill("b", ("*.rs",)))
    sc.clear_conditional_state()
    assert sc.get_conditional_skills() == []
    assert sc.is_activated("a") is False
