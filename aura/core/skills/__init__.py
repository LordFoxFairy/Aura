"""Skills — command-mode markdown prompts injected on user-typed slash commands.

v0.2.0 MVP scope:
- Flat ``<root>/<name>.md`` format only (no directory-based ``SKILL.md``).
- Discovered from ``~/.aura/skills/`` (user) + ``<cwd>/.aura/skills/`` (project).
- Name collisions resolve to the user layer (project-layer dup dropped).
- Invocation is user-driven via ``/<name>`` slash commands; the model does not
  select skills autonomously yet.

Context integration lives in :mod:`aura.core.memory.context` via two tags:
``<skills-available>`` (eager list of names+descriptions) and
``<skill-invoked>`` (per-invocation body injection).
"""

from aura.core.skills.command import SkillCommand
from aura.core.skills.loader import load_skills
from aura.core.skills.registry import SkillRegistry
from aura.core.skills.types import Skill

__all__ = [
    "Skill",
    "SkillCommand",
    "SkillRegistry",
    "load_skills",
]
