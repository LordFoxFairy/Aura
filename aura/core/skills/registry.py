"""SkillRegistry — in-memory name-indexed store of loaded Skills.

Populated once at Agent construction from :func:`load_skills`. Conditional
skills (``paths:`` frontmatter) are held in ``loader._conditional_skills``
and only enter this registry after a matching file is touched — see
:func:`aura.core.skills.loader.activate_conditional_skills_for_paths`.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterable

from aura.core.skills.types import Skill

# Alias so method annotations inside the class body resolve to the builtin
# rather than our ``list()`` method. ``list[Skill]`` in the class namespace
# would otherwise point at ``SkillRegistry.list`` and mypy rightly complains
# ("Function is not valid as a type"). Fully-qualifying via ``_List[Skill]``
# keeps the method name (which is part of the public API — the CLI + tests
# call ``registry.list()``) untouched.
_List = builtins.list


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

    def list(self) -> _List[Skill]:
        """Return all registered skills sorted by name."""
        return [self._skills[k] for k in sorted(self._skills)]

    def model_visible(self) -> _List[Skill]:
        """Return skills the LLM should see in ``<skills-available>``.

        Excludes ``disable_model_invocation=True`` skills (hidden from the
        model by the user, still invocable via ``/<name>``) and the lazy
        bucket of conditional skills (which don't enter the registry until
        activated). Mirrors claude-code's ``isModelVisible`` filter on
        ``<skills-available>``.
        """
        return [s for s in self.list() if not s.disable_model_invocation]

    def user_invocable(self) -> _List[Skill]:
        """Return skills that should register a ``/<name>`` slash command."""
        return [s for s in self.list() if s.user_invocable]
