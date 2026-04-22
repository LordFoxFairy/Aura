"""Skill value type — frozen, hashable, identified by ``source_path``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SkillLayer = Literal["user", "project"]


@dataclass(frozen=True)
class Skill:
    """A parsed skill file.

    ``source_path`` is the resolved absolute path to the ``.md`` file and is
    used for dedup (a skill invoked twice yields a single
    ``<skill-invoked>`` tag). ``name`` is unique across both layers after
    collision resolution — see :func:`aura.core.skills.loader.load_skills`.
    """

    name: str
    description: str
    body: str  # Markdown body with frontmatter stripped, raw (no truncation).
    source_path: Path
    layer: SkillLayer
