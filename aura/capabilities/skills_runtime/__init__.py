"""Skills runtime implementation.

This package owns skill loading, registry, command/tool wiring helpers,
restriction leases, and shared error formatting.
"""

from aura.capabilities.skills_runtime.command import SkillCommand, install_skill_allow_rules
from aura.capabilities.skills_runtime.errors import format_missing_args_error
from aura.capabilities.skills_runtime.loader import (
    activate_conditional_skills_for_paths,
    activated_conditional_names,
    clear_conditional_state,
    get_conditional_skills,
    load_skills,
    render_skill_body,
)
from aura.capabilities.skills_runtime.registry import SkillRegistry
from aura.capabilities.skills_runtime.types import Skill, SkillLayer

__all__ = [
    "Skill",
    "SkillCommand",
    "SkillLayer",
    "SkillRegistry",
    "activate_conditional_skills_for_paths",
    "activated_conditional_names",
    "clear_conditional_state",
    "format_missing_args_error",
    "get_conditional_skills",
    "install_skill_allow_rules",
    "load_skills",
    "render_skill_body",
]
