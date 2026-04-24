"""ASCII sprite art for the ``/buddy`` slash command.

Ported verbatim from claude-code v2.1.88's
``src/buddy/sprites.ts`` (``BODIES`` + ``HAT_LINES`` + ``EYES``). Each
sprite is 5 lines tall, 12 columns wide AFTER ``{E}`` → one-char eye
substitution. Pre-substitution widths are also 12 because ``{E}`` is a
3-char placeholder replacing a 1-char slot plus 2 chars of surrounding
whitespace/glyphs — the TS source is carefully balanced so the invariant
holds both before and after substitution.

Composition contract (mirrors ``renderSprite`` in the TS source):

- Line 0 of every frame is the hat slot. Must be blank whitespace in
  frames 0/1 of every species so the hat can be stamped in; frame 2 is
  allowed to use line 0 for idle flourishes (e.g. dragon's ``~    ~``)
  and the hat is SKIPPED on those frames.
- ``{E}`` is the eye placeholder — substituted with one of the 6
  ``EYES`` characters.
- Hats are 12-char full-width strings; ``none`` → 12 spaces.

Aura-specific additions:

- ``dog`` sprite (not present in claude-code; Aura's species list
  includes dog) — designed to match the ground-level critter style
  (frame 2 wagging a tail).
"""

from __future__ import annotations

from collections.abc import Mapping

# ---------------------------------------------------------------------------
# BODIES — per-species 3-frame sprites. 5 lines × 12 cols each.
# ---------------------------------------------------------------------------

#: Species → 3-tuple of frames; each frame is a 5-tuple of 12-char lines.
#: ``{E}`` is the eye placeholder (substituted at render time).
BODIES: Mapping[str, tuple[tuple[str, str, str, str, str], ...]] = {
    "duck": (
        (
            "            ",
            "    __      ",
            "  <({E} )___  ",
            "   (  ._>   ",
            "    `--´    ",
        ),
        (
            "            ",
            "    __      ",
            "  <({E} )___  ",
            "   (  ._>   ",
            "    `--´~   ",
        ),
        (
            "            ",
            "    __      ",
            "  <({E} )___  ",
            "   (  .__>  ",
            "    `--´    ",
        ),
    ),
    "cat": (
        (
            "            ",
            "   /\\_/\\    ",
            "  ( {E}   {E})  ",
            "  (  ω  )   ",
            "  (\")_(\")   ",
        ),
        (
            "            ",
            "   /\\_/\\    ",
            "  ( {E}   {E})  ",
            "  (  ω  )   ",
            "  (\")_(\")~  ",
        ),
        (
            "            ",
            "   /\\-/\\    ",
            "  ( {E}   {E})  ",
            "  (  ω  )   ",
            "  (\")_(\")   ",
        ),
    ),
    "dog": (
        (
            "            ",
            "   __       ",
            "  ({E}.{E})  U  ",
            "  (  _)     ",
            "   ^^       ",
        ),
        (
            "            ",
            "   __       ",
            "  ({E}.{E})  U  ",
            "  (  _)     ",
            "   ^^~      ",
        ),
        (
            "            ",
            "   __       ",
            "  ({E}.{E})  U  ",
            "  (  o)     ",
            "   ^^       ",
        ),
    ),
    "owl": (
        (
            "            ",
            "   /\\  /\\   ",
            "  (({E})({E}))  ",
            "  (  ><  )  ",
            "   `----´   ",
        ),
        (
            "            ",
            "   /\\  /\\   ",
            "  (({E})({E}))  ",
            "  (  ><  )  ",
            "   .----.   ",
        ),
        (
            "            ",
            "   /\\  /\\   ",
            "  (({E})(-))  ",
            "  (  ><  )  ",
            "   `----´   ",
        ),
    ),
    "turtle": (
        (
            "            ",
            "   _,--._   ",
            "  ( {E}  {E} )  ",
            " /[______]\\ ",
            "  ``    ``  ",
        ),
        (
            "            ",
            "   _,--._   ",
            "  ( {E}  {E} )  ",
            " /[______]\\ ",
            "   ``  ``   ",
        ),
        (
            "            ",
            "   _,--._   ",
            "  ( {E}  {E} )  ",
            " /[======]\\ ",
            "  ``    ``  ",
        ),
    ),
    "rabbit": (
        (
            "            ",
            "   (\\__/)   ",
            "  ( {E}  {E} )  ",
            " =(  ..  )= ",
            "  (\")__(\")  ",
        ),
        (
            "            ",
            "   (|__/)   ",
            "  ( {E}  {E} )  ",
            " =(  ..  )= ",
            "  (\")__(\")  ",
        ),
        (
            "            ",
            "   (\\__/)   ",
            "  ( {E}  {E} )  ",
            " =( .  . )= ",
            "  (\")__(\")  ",
        ),
    ),
    "octopus": (
        (
            "            ",
            "   .----.   ",
            "  ( {E}  {E} )  ",
            "  (______)  ",
            "  /\\/\\/\\/\\  ",
        ),
        (
            "            ",
            "   .----.   ",
            "  ( {E}  {E} )  ",
            "  (______)  ",
            "  \\/\\/\\/\\/  ",
        ),
        (
            "     o      ",
            "   .----.   ",
            "  ( {E}  {E} )  ",
            "  (______)  ",
            "  /\\/\\/\\/\\  ",
        ),
    ),
    "penguin": (
        (
            "            ",
            "  .---.     ",
            "  ({E}>{E})     ",
            " /(   )\\    ",
            "  `---´     ",
        ),
        (
            "            ",
            "  .---.     ",
            "  ({E}>{E})     ",
            " |(   )|    ",
            "  `---´     ",
        ),
        (
            "  .---.     ",
            "  ({E}>{E})     ",
            " /(   )\\    ",
            "  `---´     ",
            "   ~ ~      ",
        ),
    ),
    "dragon": (
        (
            "            ",
            "  /^\\  /^\\  ",
            " <  {E}  {E}  > ",
            " (   ~~   ) ",
            "  `-vvvv-´  ",
        ),
        (
            "            ",
            "  /^\\  /^\\  ",
            " <  {E}  {E}  > ",
            " (        ) ",
            "  `-vvvv-´  ",
        ),
        (
            "   ~    ~   ",
            "  /^\\  /^\\  ",
            " <  {E}  {E}  > ",
            " (   ~~   ) ",
            "  `-vvvv-´  ",
        ),
    ),
    "axolotl": (
        (
            "            ",
            "}~(______)~{",
            "}~({E} .. {E})~{",
            "  ( .--. )  ",
            "  (_/  \\_)  ",
        ),
        (
            "            ",
            "~}(______){~",
            "~}({E} .. {E}){~",
            "  ( .--. )  ",
            "  (_/  \\_)  ",
        ),
        (
            "            ",
            "}~(______)~{",
            "}~({E} .. {E})~{",
            "  (  --  )  ",
            "  ~_/  \\_~  ",
        ),
    ),
}


# ---------------------------------------------------------------------------
# HAT_LINES — 12-char strings stamped into line 0 when the hat slot is free.
# ---------------------------------------------------------------------------

#: Hat name → 12-char line stamped into sprite line 0. ``"none"`` → blank.
#: Ported verbatim from claude-code ``sprites.ts`` (``HAT_LINES`` record)
#: with ``none`` filled out to 12 spaces (the TS source leaves it ``""``
#: and the renderer short-circuits; we pre-fill so every hat satisfies
#: the 12-char invariant without a special case).
HAT_LINES: Mapping[str, str] = {
    "none": "            ",
    "crown": "   \\^^^/    ",
    "tophat": "   [___]    ",
    "propeller": "    -+-     ",
    "halo": "   (   )    ",
    "wizard": "    /^\\     ",
    "beanie": "   (___)    ",
    "tinyduck": "    ,>      ",
}


# ---------------------------------------------------------------------------
# EYES — 6 single-char eye glyphs, verbatim from types.ts line 76.
# ---------------------------------------------------------------------------

#: Eye characters. Each is exactly one column wide when printed — any
#: wide-char replacement here would break the 12-col invariant.
EYES: tuple[str, ...] = ("·", "✦", "×", "◉", "@", "°")


# ---------------------------------------------------------------------------
# compose_sprite — public render helper.
# ---------------------------------------------------------------------------


def compose_sprite(
    species: str,
    frame_idx: int,
    eye: str,
    hat: str,
) -> tuple[str, str, str, str, str]:
    """Build a rendered 5-line sprite with eyes + hat applied.

    Steps (mirrors ``renderSprite`` in the TS source, minus the
    "drop blank hat line" optimization — we always return 5 lines so
    the ``/buddy`` card layout is fixed-height):

    1. Look up ``BODIES[species][frame_idx]``.
    2. Substitute every ``{E}`` → ``eye``.
    3. If ``hat != "none"`` AND line 0 is all whitespace, overwrite
       line 0 with ``HAT_LINES[hat]``. Otherwise leave line 0 alone
       (frame 2 of dragon / octopus / penguin uses line 0 for idle
       flourishes; stamping the hat would erase them).

    Raises ``KeyError`` for an unknown species / hat. Frame index is
    taken modulo the frame count so callers can hand in a tick counter
    without bounds checking.
    """
    frames = BODIES[species]
    frame = frames[frame_idx % len(frames)]
    body: list[str] = [line.replace("{E}", eye) for line in frame]
    if hat != "none" and body[0].strip() == "":
        body[0] = HAT_LINES[hat]
    return (body[0], body[1], body[2], body[3], body[4])
