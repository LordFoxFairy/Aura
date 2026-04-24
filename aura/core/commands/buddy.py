"""``/buddy`` — render the user's companion as ASCII art.

Static frame-0 portrait only; there is no animation loop. Future v0.14
work may wire a 500ms tick via ``application.invalidate()``, but for
v0.13 the command is a one-shot print so it composes with the
existing str-only command renderer.

Opt-out honors the same precedence as the status-bar fragment: env var
``AURA_NO_BUDDY=1`` wins, then ``config.ui.buddy_enabled=False``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aura.cli.buddy import (
    _MOOD_LABEL,
    _env_opt_out,
    current_user_seed,
    generate_buddy,
    get_mood,
)
from aura.cli.buddy_sprites import compose_sprite
from aura.core.commands.types import CommandResult, CommandSource

if TYPE_CHECKING:
    from aura.core.agent import Agent


class BuddyCommand:
    """``/buddy`` — print a 5-line ASCII portrait of the user's buddy."""

    name = "/buddy"
    description = "show your companion's ASCII portrait"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        # Opt-out precedence mirrors ``buddy_status_fragment`` — env var
        # first (cheapest), then config flag. Either one disabled → the
        # command prints a single-line "disabled" marker instead of the
        # portrait so the user gets feedback (rather than silence) that
        # their opt-out is in effect.
        if _env_opt_out():
            return CommandResult(
                handled=True, kind="print", text="buddy disabled",
            )
        enabled = _read_buddy_enabled(agent)
        if not enabled:
            return CommandResult(
                handled=True, kind="print", text="buddy disabled",
            )

        seed = current_user_seed()
        buddy = generate_buddy(seed)
        # ``_MOOD_LABEL`` maps the 4 moods to their one-word suffix
        # (idle / thinking / happy / worried) — same table the status-bar
        # fragment uses, so ``/buddy`` and the bar can never disagree.
        mood = get_mood(agent._state)
        mood_word = _MOOD_LABEL[mood]

        # Compose the 5-line sprite: frame 0 (static), with the rolled
        # eye + hat stamped in. Hats skip frames whose line 0 is already
        # occupied — frame 0 is always hat-ready by construction of the
        # BODIES table, so this always renders the hat if one was rolled.
        sprite = compose_sprite(
            species=buddy.species,
            frame_idx=0,
            eye=buddy.eye or "·",  # defensive: fall back to the first EYE
            hat=buddy.hat or "none",
        )

        lines = [
            f"Your buddy — {buddy.emoji} {buddy.species} ({buddy.rarity})",
            f"  Eye: {buddy.eye}   Hat: {buddy.hat}",
            "",
            *sprite,
            "",
            f"  Mood: {mood_word}",
        ]
        return CommandResult(
            handled=True, kind="print", text="\n".join(lines),
        )


def _read_buddy_enabled(agent: Agent) -> bool:
    """Read ``config.ui.buddy_enabled`` defensively.

    Mirrors the ``getattr`` chain in ``aura.cli.repl._buddy_suffix_html``
    so a bare Agent (no UI config, no public ``config`` property) still
    resolves to the documented default of ``True``. Prefers the public
    ``agent.config`` attribute if the agent exposes one; falls back to
    the private ``agent._config`` since that's where ``AuraConfig`` is
    actually stashed by ``Agent.__init__``.
    """
    cfg = getattr(agent, "config", None)
    if cfg is None:
        cfg = getattr(agent, "_config", None)
    if cfg is None:
        return True
    ui = getattr(cfg, "ui", None)
    if ui is None:
        return True
    return bool(getattr(ui, "buddy_enabled", True))
