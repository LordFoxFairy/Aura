"""Status-bar pet buddy — deterministic species + mood observer.

Aura ships a complete, opt-out-able version of claude-code's half-wired
"buddy" feature (see ``claude-code-source/src/buddy/companion.ts``). The
public surface is intentionally narrow:

- :func:`generate_buddy` — pure function, ``seed → Buddy`` (species,
  rarity, emoji). Mulberry32-seeded PRNG; same seed always yields the
  same buddy.
- :func:`buddy_status_fragment` — text for the pt/rich status bar,
  e.g. ``"🦆 happy"``. Empty string when the user opted out
  (``AURA_NO_BUDDY=1`` env var or ``config.ui.buddy_enabled=False``).
- :func:`observe_post_model` / :func:`observe_post_tool` — the two
  observer callbacks wired from :mod:`aura.core.hooks.budget` that
  update the mood state machine.
- :func:`reset` — invoked from ``/clear`` so the mood flips back to
  idle on session reset.

Design constraints (see T2-B task spec):

- **No I/O, no env reads in the core gen.** ``generate_buddy`` is a
  deterministic pure function so tests can pin a seed and assert a
  specific species — unlike claude-code which reads config + hashes
  an OAuth UUID inside the generator.
- **No prompt_toolkit dependency.** Status fragment returns a plain
  ``str``; the REPL glues it onto the HTML bar in
  :mod:`aura.cli.repl`.
- **Mood state lives on ``LoopState.custom["_buddy_state"]``.** Same
  per-session scratchpad that the token-stats hook uses, so
  ``LoopState.reset()`` (called by ``/clear``) wipes our mood along
  with everything else without special-casing.

Mulberry32 reference: claude-code v2.1.88 ``companion.ts`` lines
16–25; this module ports the same 4-step integer mixer so the species
distribution is bit-for-bit compatible with operators who've eyeballed
their claude-code pet.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult

# ---------------------------------------------------------------------------
# Species + rarity tables
# ---------------------------------------------------------------------------

#: 10 species. Order is load-bearing — index into this tuple is what
#: ``generate_buddy`` picks via ``int(rng() * len(SPECIES))``. Reordering
#: would change which species every existing seed resolves to. Append,
#: never insert.
SPECIES: tuple[str, ...] = (
    "duck",
    "cat",
    "dog",
    "owl",
    "turtle",
    "rabbit",
    "octopus",
    "penguin",
    "dragon",
    "axolotl",
)


#: Species → emoji. Axolotl has no official emoji in older fonts, so we
#: pick the lizard 🦎 which is the standard fallback operators see in
#: every terminal that renders emoji at all.
SPECIES_EMOJI: Mapping[str, str] = {
    "duck": "🦆",
    "cat": "🐱",
    "dog": "🐶",
    "owl": "🦉",
    "turtle": "🐢",
    "rabbit": "🐰",
    "octopus": "🐙",
    "penguin": "🐧",
    "dragon": "🐉",
    "axolotl": "🦎",
}


Rarity = Literal["common", "uncommon", "rare", "legendary"]

#: Order matters: rarity rolling walks the tuple in this order and
#: stops at the first non-negative bucket. Reordering would re-skew
#: every existing seed.
RARITIES: tuple[Rarity, ...] = ("common", "uncommon", "rare", "legendary")

#: Weights must sum to 100. Spec-pinned (T2-B §Design §Deterministic):
#: common 60 / uncommon 25 / rare 12 / legendary 3.
RARITY_WEIGHTS: Mapping[Rarity, int] = {
    "common": 60,
    "uncommon": 25,
    "rare": 12,
    "legendary": 3,
}


# ---------------------------------------------------------------------------
# Mulberry32 — deterministic PRNG
# ---------------------------------------------------------------------------


_UINT32 = 0xFFFFFFFF
_POW32 = 0x100000000  # 2**32, denominator for the 0..1 float conversion.


def _imul32(a: int, b: int) -> int:
    """32-bit multiply with wraparound, matching JS ``Math.imul``.

    Python ints don't overflow, so we mask to 32 bits after the multiply.
    This matters bit-for-bit: mulberry32's mixing relies on the
    wraparound, and dropping the mask produces a completely different
    sequence.
    """
    return (a * b) & _UINT32


def _mulberry32(seed: int) -> Callable[[], float]:
    """Port of claude-code's companion.ts mulberry32.

    Returns a closure yielding floats in ``[0.0, 1.0)``. Tiny state
    (one int) so we never need to reseed mid-sequence. Each call:

    1. ``a += 0x6d2b79f5`` (32-bit)
    2. ``t = imul32(a XOR (a >>> 15), 1 | a)``
    3. ``t += imul32(t XOR (t >>> 7), 61 | t)`` then XOR with t
    4. return ``((t XOR (t >>> 14)) >>> 0) / 2**32``
    """
    state = [seed & _UINT32]

    def _next() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & _UINT32
        a = state[0]
        t = _imul32(a ^ (a >> 15), 1 | a)
        t = (t + _imul32(t ^ (t >> 7), 61 | t)) ^ t
        t = t & _UINT32
        return ((t ^ (t >> 14)) & _UINT32) / _POW32

    return _next


def _fnv1a(s: str) -> int:
    """FNV-1a 32-bit string hash — same algorithm as companion.ts'
    non-Bun branch (line 31–36). Deterministic across platforms.
    """
    h = 2166136261
    for ch in s:
        h ^= ord(ch) & 0xFF
        h = _imul32(h, 16777619)
    return h & _UINT32


# Salt mirrors companion.ts line 84 — distinguishes Aura buddies from
# any claude-code buddy an operator might have generated with the same
# username, so the two systems don't resolve to the same duck.
_SALT = "aura-buddy-2026-v1"


# ---------------------------------------------------------------------------
# Public: Buddy + generate_buddy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Buddy:
    """Immutable snapshot of a rolled buddy.

    ``species`` is a member of :data:`SPECIES`; ``rarity`` is a member
    of :data:`RARITIES`; ``emoji`` is the corresponding entry from
    :data:`SPECIES_EMOJI`. Frozen so callers can safely stash it on
    state without worrying about accidental mutation.

    ``eye`` / ``hat`` (T2B-sprites, v0.13) drive the ASCII portrait
    rendered by ``/buddy``. Defaults are empty strings so older pickled
    states (pre-sprite) still construct cleanly — generate_buddy always
    fills them with real values from
    :data:`aura.cli.buddy_sprites.EYES` /
    :data:`aura.cli.buddy_sprites.HAT_LINES`.
    """

    species: str
    rarity: Rarity
    emoji: str
    eye: str = ""
    hat: str = ""


def _roll_rarity(rng: Callable[[], float]) -> Rarity:
    """Walk the rarity tuple subtracting weights until the roll underflows.

    Total weight = 100 (:data:`RARITY_WEIGHTS` invariant). The roll is
    in ``[0, 100)`` so the walk always terminates before the ``common``
    fallback — the fallback is defensive belt-and-braces in case a
    future maintainer breaks the sum-to-100 invariant.
    """
    total = sum(RARITY_WEIGHTS.values())
    roll = rng() * total
    for rarity in RARITIES:
        roll -= RARITY_WEIGHTS[rarity]
        if roll < 0:
            return rarity
    return "common"


def generate_buddy(seed: str) -> Buddy:
    """Pure ``seed → Buddy`` function.

    Empty seed falls back to ``"default"`` (pathological case: a
    ``Path.home().name`` that's the empty string). This guarantees
    every caller gets a valid buddy — no ``None`` returns, no
    exceptions.

    The seed is concatenated with :data:`_SALT` before hashing so
    Aura's namespace of buddies is disjoint from claude-code's, even
    for operators who happen to share the same username. Hashing goes
    FNV-1a → mulberry32 to match the claude-code port bit-for-bit.

    Roll order (matches claude-code ``companion.ts:92-100``): rarity →
    species → eye → hat. Rarity-first so the hat picker can short-
    circuit to ``"none"`` on common buddies without spending an rng
    tick that would perturb downstream draws.
    """
    # Local import avoids a module-init cycle: buddy_sprites imports nothing
    # from this file, but keeping the import lazy mirrors the style the rest
    # of this module uses for ``aura.schemas`` and keeps the top-of-file
    # dependency list focused on the core status-bar surface.
    from aura.cli.buddy_sprites import EYES, HAT_LINES

    effective = seed if seed else "default"
    rng = _mulberry32(_fnv1a(effective + _SALT))

    rarity = _roll_rarity(rng)
    species_idx = int(rng() * len(SPECIES))
    # Guard against rng() returning exactly 1.0 (mathematically
    # impossible since we divide by 2**32, but belt-and-braces).
    species_idx = min(species_idx, len(SPECIES) - 1)
    species = SPECIES[species_idx]

    eye_idx = min(int(rng() * len(EYES)), len(EYES) - 1)
    eye = EYES[eye_idx]

    # Common buddies always go bare-headed — mirrors claude-code
    # ``companion.ts:97`` (``rarity === 'common' ? 'none' : pick(...)``).
    # For non-common buddies we pick uniformly from the 7 non-``none``
    # hats so the hat is always present and the roll never lands on
    # ``none`` twice (which would waste the rarity upgrade).
    if rarity == "common":
        hat = "none"
    else:
        # Build the non-``none`` hat list lazily so reordering
        # HAT_LINES doesn't silently re-skew every existing seed.
        non_none_hats = tuple(h for h in HAT_LINES if h != "none")
        hat_idx = min(int(rng() * len(non_none_hats)), len(non_none_hats) - 1)
        hat = non_none_hats[hat_idx]

    return Buddy(
        species=species,
        rarity=rarity,
        emoji=SPECIES_EMOJI[species],
        eye=eye,
        hat=hat,
    )


def current_user_seed() -> str:
    """Stable per-user seed — ``Path.home().name`` (e.g. ``"nako"``).

    Falls back to ``"default"`` for pathological empty home paths. No
    PII beyond what the shell already exposes via ``$HOME``.
    """
    name = Path.home().name
    return name if name else "default"


# ---------------------------------------------------------------------------
# Mood observer — state machine on LoopState.custom["_buddy_state"]
# ---------------------------------------------------------------------------


Mood = Literal["idle", "thinking", "happy", "worried"]

_MOOD_KEY = "_buddy_state"

#: Per-mood glyph shown AFTER the species emoji in the status fragment.
#: Operators scan the bar at a glance — a text suffix ("happy") reads
#: faster than a secondary emoji and stays single-line in any terminal.
_MOOD_LABEL: Mapping[Mood, str] = {
    "idle": "idle",
    "thinking": "thinking",
    "happy": "happy",
    "worried": "worried",
}


def _buddy_state(state: LoopState) -> dict[str, object]:
    """Lazy-init the buddy state slot on the loop's per-session scratchpad."""
    slot = state.custom.get(_MOOD_KEY)
    if not isinstance(slot, dict):
        slot = {"mood": "idle", "last_event_ts": 0.0, "had_recent_error": False}
        state.custom[_MOOD_KEY] = slot
    return slot


def get_mood(state: LoopState) -> Mood:
    """Read the current mood — ``idle`` when no events have fired yet.

    Safe to call before any observer has run; lazy-init keeps the state
    slot absent until we actually need it (avoids polluting ``/stats``
    output with an empty buddy stub on bare agents)."""
    slot = state.custom.get(_MOOD_KEY)
    if not isinstance(slot, dict):
        return "idle"
    mood = slot.get("mood")
    if mood == "idle":
        return "idle"
    if mood == "thinking":
        return "thinking"
    if mood == "happy":
        return "happy"
    if mood == "worried":
        return "worried"
    return "idle"


def reset(state: LoopState) -> None:
    """Wipe the buddy state — called by ``/clear`` to undo accumulated mood.

    Delete rather than rewriting the dict so a subsequent ``get_mood``
    naturally returns ``idle`` via the lazy-init path. Matches the
    ``LoopState.reset()`` contract: in-place mutation preserves the
    caller's reference.
    """
    state.custom.pop(_MOOD_KEY, None)


async def observe_pre_model(*, state: LoopState, **_: object) -> None:
    """pre_model observer: flip to ``thinking`` while the model is busy.

    Fires AFTER the user's HumanMessage has been appended to history but
    BEFORE ``ainvoke`` returns. From the operator's perspective the
    buddy "starts thinking" the instant they hit Enter on a prompt.
    The transition out happens automatically: ``observe_post_model``
    overwrites mood to ``happy`` (or ``worried`` if a tool errored)
    when the model reply lands.

    Worry preservation: if ``had_recent_error`` is set we DON'T overwrite
    the mood. A user who just saw a tool fail expects continuity —
    seeing the buddy switch to "thinking" between "worried" and the
    next "worried" would be more confusing than helpful. Worry sticks
    until a successful tool clears the flag.
    """
    slot = _buddy_state(state)
    slot["last_event_ts"] = time.time()
    if slot.get("had_recent_error"):
        return
    slot["mood"] = "thinking"


async def observe_post_model(*, state: LoopState, **_: object) -> None:
    """post_model observer: flip to ``happy`` unless a recent tool error
    kept us in ``worried``.

    The state machine is "sticky worried": once a tool error has fired,
    subsequent model replies stay worried until a *successful* tool
    clears the flag. This matches claude-code's
    ``useBuddyNotification.tsx`` flow where worry lingers across turns
    so the operator notices something went sideways even if the next
    reply looks fine.
    """
    slot = _buddy_state(state)
    slot["last_event_ts"] = time.time()
    if slot.get("had_recent_error"):
        slot["mood"] = "worried"
    else:
        slot["mood"] = "happy"


async def observe_post_tool(
    *,
    state: LoopState,
    result: ToolResult,
    **_: object,
) -> None:
    """post_tool observer: track worry on failed tools, clear it on success.

    Successful tools DON'T immediately flip the mood to "happy" — we
    leave mood as-is and let the next post_model decide. That way a
    single successful post_tool between a failed one and the reply
    doesn't mask the worry: mood only turns happy after BOTH the tool
    chain has recovered AND the model has spoken again.
    """
    slot = _buddy_state(state)
    slot["last_event_ts"] = time.time()
    if not result.ok:
        slot["had_recent_error"] = True
        slot["mood"] = "worried"
    else:
        # Success clears the worry flag; mood itself waits for the
        # next post_model to transition (see above).
        slot["had_recent_error"] = False


# ---------------------------------------------------------------------------
# Status-bar fragment — the one surface the REPL glues in
# ---------------------------------------------------------------------------


def _env_opt_out() -> bool:
    """``AURA_NO_BUDDY=1`` (case-insensitive; also ``"true"``/``"yes"``) opts out.

    We parse the value liberally because the env-var pattern in
    claude-code / Aura's codebase has historically accepted both "1"
    and truthy strings; keeping our parse aligned avoids surprises for
    operators copy-pasting their shell rc between the two tools.
    """
    val = os.environ.get("AURA_NO_BUDDY", "").strip().lower()
    return val in ("1", "true", "yes", "on")


#: Animation glyph cycle for the time-aware status fragment. Same family
#: as :data:`aura.cli.spinner._GLYPHS` and ``_BANNER_SPINNER_FRAMES`` in
#: :mod:`aura.cli.repl` so the buddy's idle motion shares visual language
#: with the welcome banner and the in-turn thinking spinner — every
#: animated surface in Aura draws from the same character set.
#:
#: Palindromic so the glyph "bounces" back and forth instead of resetting,
#: matching claude-code's ``CompanionStatusBar.tsx`` (v2.1.88) which uses
#: the same darwin spinner set with the same bounce pattern.
BUDDY_FRAMES: tuple[str, ...] = (
    "·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢",
)

#: Seconds per glyph. 0.5s is the sweet spot — fast enough to read as
#: animated, slow enough that pt's ``refresh_interval=0.5`` repaint cost
#: stays negligible (one full re-render of the bottom toolbar). Aligned
#: with the REPL's chosen ``refresh_interval`` so each pt tick advances
#: the animation by exactly one frame.
BUDDY_FRAME_INTERVAL: float = 0.5


def _frame_for_now(now: float) -> str:
    """Pick the cycle frame for the given wall-clock time.

    ``now / interval`` floored to int gives a stable index that bumps by
    one every :data:`BUDDY_FRAME_INTERVAL` seconds. Modulo the frame
    count to wrap. Same ``now`` always yields the same frame (no
    randomness) so pt invocations within the same tick produce
    identical fragments and the bar doesn't visibly flicker.
    """
    if BUDDY_FRAME_INTERVAL <= 0.0:
        return BUDDY_FRAMES[0]
    idx = int(now / BUDDY_FRAME_INTERVAL) % len(BUDDY_FRAMES)
    return BUDDY_FRAMES[idx]


def time_aware_status_fragment(
    *,
    state: LoopState,
    seed: str | None = None,
    enabled: bool = True,
    now: float,
) -> str:
    """Animated variant of :func:`buddy_status_fragment`.

    Same opt-out precedence and same shape (``"<emoji> <mood>"``) plus a
    leading cycling glyph drawn from :data:`BUDDY_FRAMES`. The glyph
    advances every :data:`BUDDY_FRAME_INTERVAL` seconds of ``now`` —
    callers pass ``time.time()`` (or a fake in tests) and the function
    is otherwise pure.

    Wired from :mod:`aura.cli.repl` where pt's ``refresh_interval``
    forces the bottom_toolbar callable to run on a timer; without that
    timer the glyph would only advance on user keystrokes.
    """
    if not enabled:
        return ""
    if _env_opt_out():
        return ""
    effective_seed = seed if seed is not None else current_user_seed()
    b = generate_buddy(effective_seed)
    mood = get_mood(state)
    glyph = _frame_for_now(now)
    return f"{glyph} {b.emoji} {_MOOD_LABEL[mood]}"


def buddy_status_fragment(
    *,
    state: LoopState,
    seed: str | None = None,
    enabled: bool = True,
) -> str:
    """Return the buddy's status-bar fragment, or ``""`` when opted out.

    Returns a plain string (not pt HTML) so this module stays
    pt-dependency-free; :mod:`aura.cli.repl` wraps the string in the
    existing HTML bar. Format: ``"<emoji> <mood>"`` e.g. ``"🦆 happy"``.

    Opt-out precedence (first wins, earliest cheapest):

    1. ``enabled=False`` (from ``config.ui.buddy_enabled``).
    2. ``AURA_NO_BUDDY=1`` env var.

    The seed defaults to :func:`current_user_seed` when the caller
    doesn't pass one — tests pin it explicitly so they don't depend on
    the autouse ``_isolated_home`` fixture producing a predictable
    basename.
    """
    if not enabled:
        return ""
    if _env_opt_out():
        return ""
    effective_seed = seed if seed is not None else current_user_seed()
    buddy = generate_buddy(effective_seed)
    mood = get_mood(state)
    return f"{buddy.emoji} {_MOOD_LABEL[mood]}"
