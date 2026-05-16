"""Skill value type — directory-per-skill, claude-code v2.1.88 compatible.

Design reference: ``loadSkillsDir.ts`` `parseSkillFrontmatterFields` +
`createSkillCommand`. Aura-side adaptations:

- No MCP skill builders / plugin layer / analytics / policy settings —
  those slots exist in claude-code to thread through a commercial SaaS
  harness. Aura's MVP skills are pure file-backed prompts.
- ``${CLAUDE_SKILL_DIR}`` / ``${CLAUDE_SESSION_ID}`` renamed to
  ``${AURA_SKILL_DIR}`` / ``${AURA_SESSION_ID}`` — our product, our namespace.
- Inline shell-command execution (`` !`cmd` ``) is deliberately NOT supported;
  claude-code executes these via ``executeShellCommandsInPrompt`` but that's
  a security surface we don't need in v1 (users who want shell can pipe
  bash + skill together at the model level).

Identity: the resolved absolute path to the ``SKILL.md`` file
(``source_path``). Two discovered skill dirs that resolve to the same file
(via symlinks) are deduped by realpath in the loader; first-seen wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# "managed" is reserved for the future policy-settings layer (claude-code's
# policySettings → /etc-style skills). Not loaded today; kept so callers can
# pattern-match on the literal without surprises when we wire it up.
SkillLayer = Literal["user", "project", "managed"]


@dataclass(frozen=True)
class Skill:
    """A parsed skill file — dir-per-skill (``<dir>/SKILL.md``).

    Required at construction: ``name``, ``description``, ``body``,
    ``source_path`` (resolved absolute path to SKILL.md), ``layer``.
    All frontmatter-driven knobs (``when_to_use``, ``allowed_tools``, etc.)
    default to the "minimal skill" configuration so tests constructing a
    Skill with only the legacy five fields keep compiling.

    ``base_dir`` defaults to ``source_path.parent`` — every on-disk skill
    lives at ``<base_dir>/SKILL.md``, so the derivation is lossless. Tests
    constructing synthetic Skills from ``/tmp/foo.md`` get ``/tmp`` as
    base_dir automatically, matching the contract at invocation time.
    """

    # Required identity fields.
    name: str
    description: str
    body: str  # Markdown body with frontmatter stripped, pre-substitution.
    source_path: Path  # Resolved absolute path to the SKILL.md file.
    layer: SkillLayer

    # Directory resources sit next to SKILL.md; body may reference them via
    # ``${AURA_SKILL_DIR}/examples/...`` which the render helper substitutes.
    # Frozen dataclass + default=None + __post_init__ is the idiomatic Python
    # way to compute a derived field.
    base_dir: Path | None = None

    # Optional frontmatter — all default to the "no special config" values.
    when_to_use: str | None = None
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    # ``restrict_tools`` (V14): strict whitelist — while this skill's lease is
    # active, ONLY tools in this set may be invoked by the model; everything
    # else short-circuits to ``restrict_tools_blocked`` BEFORE bypass / safety
    # / rule resolution. Distinct from ``allowed_tools`` (permissive auto-
    # allow): restrict_tools does not auto-allow declared tools, it just
    # leaves them unblocked. Empty = no restriction.
    restrict_tools: frozenset[str] = field(default_factory=frozenset)
    arguments: tuple[str, ...] = ()
    argument_hint: str | None = None
    version: str | None = None
    # Non-empty paths → conditional skill. Activated lazily by
    # ``activate_conditional_skills_for_paths`` when a matching file is
    # touched by a tool call. Empty frozenset → unconditional (always in
    # <skills-available>).
    paths: frozenset[str] = field(default_factory=frozenset)
    # ``/skill-name`` slash command registered iff True (default). Matches
    # claude-code: user-invocable=false skills are hidden from slash-command
    # completion but still loadable by the model (unless also
    # disable_model_invocation=True, which is the "fully hidden" case).
    user_invocable: bool = True
    # Hidden from the LLM's <skills-available> block when True. Combined with
    # user_invocable=True, this models the "user-only skill" case — a
    # workflow shortcut the user wants but the model shouldn't auto-invoke.
    disable_model_invocation: bool = False
    # Flipped to True by ``activate_conditional_skills_for_paths`` when a
    # conditional skill's paths match a touched file. Used by
    # :meth:`is_conditional` so render-time filters distinguish "still
    # waiting to be triggered" (filtered out) from "activated, now visible
    # in <skills-available>" (rendered). Unconditional skills default to
    # False and are unaffected (``is_conditional()`` short-circuits on
    # empty ``paths``).
    activated: bool = False

    def __post_init__(self) -> None:
        # Frozen dataclasses forbid regular attribute assignment; the
        # official escape hatch is object.__setattr__ in __post_init__.
        if self.base_dir is None:
            object.__setattr__(self, "base_dir", self.source_path.parent)

    def is_conditional(self) -> bool:
        """True iff this skill is waiting on a conditional trigger.

        ``paths`` empty → unconditional (always False).
        ``paths`` non-empty + ``activated=False`` → conditional, lazy-waiting → True.
        ``paths`` non-empty + ``activated=True`` → was conditional, now triggered,
        behaves like unconditional → False.
        """
        return len(self.paths) > 0 and not self.activated
