"""Skill value type — directory-per-skill, identified by resolved SKILL.md path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SkillLayer = Literal["user", "project", "managed"]


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    source_path: Path
    layer: SkillLayer

    base_dir: Path | None = None
    when_to_use: str | None = None
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    # Lease-scoped strict whitelist (empty = no restriction); distinct from allowed_tools.
    restrict_tools: frozenset[str] = field(default_factory=frozenset)
    arguments: tuple[str, ...] = ()
    argument_hint: str | None = None
    version: str | None = None
    # Non-empty paths → conditional skill (activated lazily on file touch).
    paths: frozenset[str] = field(default_factory=frozenset)
    # False → skill won't register a ``/<name>`` slash command.
    user_invocable: bool = True
    # True → hidden from ``<skills-available>`` (still runnable via slash).
    disable_model_invocation: bool = False
    # Flipped True once activation triggers a conditional skill.
    activated: bool = False

    def __post_init__(self) -> None:
        if self.base_dir is None:
            object.__setattr__(self, "base_dir", self.source_path.parent)

    def is_conditional(self) -> bool:
        """True iff this skill is waiting on a conditional trigger."""
        return len(self.paths) > 0 and not self.activated
