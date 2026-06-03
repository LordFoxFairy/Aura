"""SkillRegistry — in-memory name-indexed store of loaded Skills.

Populated once at AgentSession construction from :func:`load_skills`. Conditional
skills enter the registry only after activation moves them out of the
loader's lazy bucket.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterable

from aura.domain.skill import Skill

# Alias so method annotations resolve to the builtin rather than the
# ``list()`` method (mypy would complain ``Function is not a type``).
_List = builtins.list


class SkillRegistry:
    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills: dict[str, Skill] = {}
        for s in skills:
            self.register(s)

    def register(self, skill: Skill) -> None:
        """Add ``skill``; raises ``ValueError`` on duplicate name."""
        if skill.name in self._skills:
            raise ValueError(f"skill {skill.name!r} is already registered")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list(self) -> _List[Skill]:
        """Return all registered skills sorted by name."""
        return [self._skills[k] for k in sorted(self._skills)]

    def model_visible(self) -> _List[Skill]:
        """Skills the LLM sees in ``<skills-available>``."""
        return [s for s in self.list() if not s.disable_model_invocation]

    def user_invocable(self) -> _List[Skill]:
        """Skills that register a ``/<name>`` slash command."""
        return [s for s in self.list() if s.user_invocable]
