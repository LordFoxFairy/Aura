"""SkillRegistry — in-memory name-indexed store of loaded Skills.

Populated once at Agent construction from :func:`load_skills`. Not mutated
during a session in the MVP (no hot reload).
"""

from __future__ import annotations

from collections.abc import Iterable

from aura.core.skills.types import Skill


class SkillRegistry:
    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills: dict[str, Skill] = {}
        for s in skills:
            self.register(s)

    def register(self, skill: Skill) -> None:
        """Add ``skill``.

        Raises:
            ValueError: if ``skill.name`` is already registered.
        """
        if skill.name in self._skills:
            raise ValueError(
                f"skill {skill.name!r} is already registered"
            )
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list(self) -> list[Skill]:
        """Return all registered skills sorted by name."""
        return [self._skills[k] for k in sorted(self._skills)]
