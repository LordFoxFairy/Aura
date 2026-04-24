"""``/buddy`` — render the user's companion as ASCII art.

Animated in TTY contexts: the command owns the 5-line sprite area for
about 3 seconds, cycling through claude-code's ``IDLE_SEQUENCE`` of
frame indices with 250 ms ticks. Non-TTY contexts (pipes, programmatic
callers, tests) fall through to a static frame-0 print so the output is
still correct when captured.

The animation is self-contained (no ``application.invalidate()``,
no background task) — it writes ANSI ``\\033[F`` cursor-up escapes
to rewrite its own output region, finishes within a bounded ~3 s
window, and leaves the cursor below the final frame so the REPL's
next prompt renders cleanly. Matches claude-code's ``CompanionSprite``
behaviour without the React-ink machinery.

Opt-out honors the same precedence as the status-bar fragment: env var
``AURA_NO_BUDDY=1`` wins, then ``config.ui.buddy_enabled=False``.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import TYPE_CHECKING

from aura.core.commands.types import CommandResult, CommandSource

if TYPE_CHECKING:
    from aura.core.agent import Agent

# Frame-index sequence claude-code's ``CompanionSprite`` cycles through for
# idle fidget — see ``src/buddy/CompanionSprite.tsx:23``. -1 is a "blink"
# slot; Aura's sprites don't have a separate eyes-closed frame, so we map
# -1 → 0 for visual continuity (the body is identical; the only thing
# that'd change is eye glyph, and losing it for one tick isn't worth a
# separate frame table).
_IDLE_SEQUENCE: tuple[int, ...] = (0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 2, 0, 0, 0)

# Per-tick pause. Claude-code uses 500 ms; at 250 ms the 15-tick cycle
# finishes in ~3.75 s which feels lively without holding the REPL prompt
# hostage. Tunable via ``AURA_BUDDY_TICK_MS`` for dogfood / demos.
_DEFAULT_TICK_MS: int = 250

# NB: ``aura.cli.buddy`` / ``aura.cli.buddy_sprites`` are imported lazily
# inside ``handle`` to break a circular import. The package chain
# ``aura.cli/__init__.py`` imports ``aura.cli.commands`` (the façade),
# which in turn imports this module to register ``BuddyCommand``. If we
# imported ``aura.cli.buddy`` at module top, ``aura.cli.__init__`` would
# re-enter itself before its exports were assigned, raising
# ``ImportError: cannot import name 'BuddyCommand' ...``. The deferred
# import runs at handle-time when ``aura.cli`` is fully initialised.
# Pattern mirrors ``aura.core.hooks.must_read_first`` + ``bash_safety`` —
# journal writes deferred to avoid module-load-time side-effects.


class BuddyCommand:
    """``/buddy`` — print a 5-line ASCII portrait of the user's buddy.

    TTY + default args: animates through the IDLE_SEQUENCE fidget cycle.
    Non-TTY or ``/buddy still``: single static print of frame 0.
    """

    name = "/buddy"
    description = "show your companion's ASCII portrait"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[still]"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        # Deferred import: see module-level note. Must run at handle-time so
        # ``aura.cli``'s __init__ has finished loading before we pull
        # ``aura.cli.buddy`` symbols.
        from aura.cli.buddy import (
            _MOOD_LABEL,
            _env_opt_out,
            current_user_seed,
            generate_buddy,
            get_mood,
        )
        from aura.cli.buddy_sprites import compose_sprite

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
        mood_word = _MOOD_LABEL[get_mood(agent._state)]

        def _frame(frame_idx: int) -> tuple[str, ...]:
            """Compose a 5-line sprite with the rolled eye + hat stamped in.

            Unknown species (if any slipped past ``generate_buddy``'s
            10-species gen) falls back to a single-line placeholder so
            the command never crashes mid-animation.
            """
            try:
                return compose_sprite(
                    species=buddy.species,
                    frame_idx=frame_idx,
                    eye=buddy.eye or "·",
                    hat=buddy.hat or "none",
                )
            except KeyError:
                return (f"  ({buddy.species} — no sprite)",)

        header = [
            f"Your buddy — {buddy.emoji} {buddy.species} ({buddy.rarity})",
            f"  Eye: {buddy.eye}   Hat: {buddy.hat}",
            "",
        ]
        footer = ["", f"  Mood: {mood_word}"]

        # Decide animate vs static. Non-TTY (pipe, programmatic, test
        # capture) and explicit ``/buddy still`` both skip the animation;
        # the animation writes ANSI escapes that show up as literal
        # ``\\033[...`` bytes when stdout is captured, which breaks tests
        # and log files. Static path matches the original v0.13 behaviour.
        static_mode = (
            arg.strip() == "still"
            or not getattr(sys.stdout, "isatty", lambda: False)()
        )
        if static_mode:
            lines = [*header, *_frame(0), *footer]
            return CommandResult(
                handled=True, kind="print", text="\n".join(lines),
            )

        # Animated path. Own the 5 sprite lines + 2 footer lines by
        # rewriting them in place with cursor-up ANSI. The first render is
        # a plain print (cursor ends below the last line); subsequent
        # frames prepend a ``\\033[{N}F`` to jump back to the start of the
        # sprite block and ``\\033[2K`` to clear each line before writing.
        await _animate(
            header=header,
            footer=footer,
            frame_picker=_frame,
            tick_ms=_tick_ms(),
        )
        # Nothing left for the REPL renderer to print — the animation
        # already produced all the output. An empty ``kind="print"`` still
        # keeps ``handled=True`` so dispatch knows the command ran.
        return CommandResult(handled=True, kind="print", text="")


def _tick_ms() -> int:
    """Read the per-frame delay. Env override lets demos speed it up."""
    import os
    raw = os.environ.get("AURA_BUDDY_TICK_MS")
    if raw:
        with contextlib.suppress(ValueError):
            parsed = int(raw)
            if parsed > 0:
                return parsed
    return _DEFAULT_TICK_MS


async def _animate(
    *,
    header: list[str],
    footer: list[str],
    frame_picker: _FramePicker,
    tick_ms: int,
) -> None:
    """Run the IDLE_SEQUENCE fidget with in-place sprite rewrites.

    Contract:
    - The first frame is printed with regular ``print`` — the cursor ends
      immediately below ``footer[-1]``.
    - Subsequent frames cursor-up back to the first sprite line, clear
      each line, and overwrite with the new frame's content.
    - On ``CancelledError`` (user hit Ctrl+C mid-animation) we suppress
      the exception and just stop — the REPL's own cancel handling
      re-prompts on the next line, and the partially-animated output is
      not load-bearing state.
    """
    first_frame = frame_picker(0)
    sprite_lines = len(first_frame)
    footer_blank_line = 1  # the "" between sprite and "Mood: …"
    mood_line = 1
    jumpback = sprite_lines + footer_blank_line + mood_line

    # Initial full render.
    for line in header:
        sys.stdout.write(line + "\n")
    for line in first_frame:
        sys.stdout.write(line + "\n")
    for line in footer:
        sys.stdout.write(line + "\n")
    sys.stdout.flush()

    with contextlib.suppress(asyncio.CancelledError):
        # Skip index 0 since we already printed frame 0. Iterate the
        # remaining 14 ticks; clamp -1 → 0 (Aura has no blink frame).
        for tick_frame_idx in _IDLE_SEQUENCE[1:]:
            await asyncio.sleep(tick_ms / 1000)
            idx = 0 if tick_frame_idx == -1 else tick_frame_idx
            frame = frame_picker(idx)
            # Jump cursor up to the first sprite line and rewrite the
            # sprite + footer area. ``\033[{n}F`` = cursor up N lines
            # AND move to column 1; ``\033[2K`` = clear entire current
            # line. Leaving the cursor at end-of-footer so the REPL
            # re-prompt lands on a fresh line below.
            sys.stdout.write(f"\033[{jumpback}F")
            for line in frame:
                sys.stdout.write("\033[2K" + line + "\n")
            for line in footer:
                sys.stdout.write("\033[2K" + line + "\n")
            sys.stdout.flush()


# Typing alias for the frame picker callback (declared after the function
# so the runtime alias doesn't forward-ref an unresolved generic).
from collections.abc import Callable as _Callable  # noqa: E402

_FramePicker = _Callable[[int], tuple[str, ...]]


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
