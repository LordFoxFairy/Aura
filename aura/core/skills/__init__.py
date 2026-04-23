"""Skills — directory-per-skill markdown prompts (claude-code v2.1.88 format).

Disk layout:

- User   → ``~/.aura/skills/<name>/SKILL.md``
- Project → ``<dir>/.aura/skills/<name>/SKILL.md`` (walks up cwd→$HOME exclusive)

Each skill dir can sibling resources (``examples/``, scripts) referenced
from the SKILL.md body via ``${AURA_SKILL_DIR}``. Frontmatter supports
``description`` (required), ``name``/``when_to_use``/``allowed-tools``/
``arguments``/``argument-hint``/``version``/``paths``/``user-invocable``/
``disable-model-invocation``.

Plain top-level ``.md`` files under ``.aura/skills/`` (the pre-0.8 Aura
layout) are NO LONGER loaded; the loader emits a
``skill_legacy_format_detected`` journal event listing the offending files
so users can migrate explicitly.

Context integration lives in :mod:`aura.core.memory.context` via two tags:
``<skills-available>`` (eager catalogue) and ``<skill-invoked>`` (per-
invocation body injection, body pre-rendered with variable substitution).
"""

from aura.core.skills.command import SkillCommand
from aura.core.skills.loader import (
    activate_conditional_skills_for_paths,
    activated_conditional_names,
    clear_conditional_state,
    get_conditional_skills,
    load_skills,
    render_skill_body,
)
from aura.core.skills.registry import SkillRegistry
from aura.core.skills.types import Skill, SkillLayer

__all__ = [
    "Skill",
    "SkillCommand",
    "SkillLayer",
    "SkillRegistry",
    "activate_conditional_skills_for_paths",
    "activated_conditional_names",
    "clear_conditional_state",
    "get_conditional_skills",
    "load_skills",
    "render_skill_body",
]
