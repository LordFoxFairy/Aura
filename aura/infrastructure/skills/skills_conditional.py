"""Process-lifetime state machine for ``paths:``-conditional skills."""

from __future__ import annotations

from pathlib import Path

import pathspec

from aura.domain.skill import Skill

# Process-lifetime state so activation sticks across ``load_skills`` calls.
_conditional_skills: dict[str, Skill] = {}
_activated_conditional_names: set[str] = set()


def stash_conditional(skill: Skill) -> None:
    """Hold ``skill`` in the lazy bucket until a matching file touch."""
    _conditional_skills[skill.name] = skill


def is_activated(name: str) -> bool:
    """Whether the conditional skill ``name`` was activated this session."""
    return name in _activated_conditional_names


def activate_conditional_skills_for_paths(
    paths: list[str],
    cwd: Path,
) -> list[str]:
    """Activate and return stashed skills whose ``paths:`` match (gitignore semantics)."""
    if not _conditional_skills:
        return []
    cwd_resolved = cwd.resolve()
    activated: list[str] = []
    for name in list(_conditional_skills.keys()):
        skill = _conditional_skills[name]
        if not skill.paths:
            _activated_conditional_names.add(name)
            del _conditional_skills[name]
            activated.append(name)
            continue
        try:
            spec = pathspec.PathSpec.from_lines("gitignore", skill.paths)
        except Exception:  # noqa: BLE001 — pathspec raises many exc types
            continue
        for raw_path in paths:
            rel = _relative_to_cwd(raw_path, cwd_resolved)
            if rel is not None and spec.match_file(rel):
                _activated_conditional_names.add(name)
                del _conditional_skills[name]
                activated.append(name)
                break
    return activated


def get_conditional_skills() -> list[Skill]:
    """Return all skills currently stashed as conditional."""
    return list(_conditional_skills.values())


def clear_conditional_state() -> None:
    """Reset the module-global conditional state (test hook)."""
    _conditional_skills.clear()
    _activated_conditional_names.clear()


def _relative_to_cwd(raw_path: str, cwd: Path) -> str | None:
    """Return ``raw_path`` as a cwd-relative POSIX string, or None if outside."""
    p = Path(raw_path)
    if p.is_absolute():
        try:
            return p.resolve().relative_to(cwd).as_posix()
        except (ValueError, OSError):
            return None
    try:
        return (cwd / p).resolve().relative_to(cwd).as_posix()
    except ValueError:
        return None
