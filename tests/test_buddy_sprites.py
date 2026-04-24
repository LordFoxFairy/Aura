"""Tests for aura.cli.buddy_sprites — ASCII sprite port.

Verifies the 12-col × 5-line invariant holds for every sprite frame, that
``{E}`` eye substitution and hat stamping work as specced, and that
frame-2 flourishes (where line 0 is non-blank) don't get clobbered by
the hat.
"""

from __future__ import annotations

from aura.cli.buddy_sprites import BODIES, EYES, HAT_LINES, compose_sprite

# ---------------------------------------------------------------------------
# Shape invariants — every species has 3 frames of 5 lines.
# ---------------------------------------------------------------------------


def test_all_species_have_3_frames_of_5_lines() -> None:
    """Every species must have exactly 3 frames, each a 5-tuple."""
    for species, frames in BODIES.items():
        assert len(frames) == 3, (
            f"{species}: expected 3 frames, got {len(frames)}"
        )
        for i, frame in enumerate(frames):
            assert len(frame) == 5, (
                f"{species}[{i}]: expected 5 lines, got {len(frame)}"
            )


def test_all_frames_12_chars_wide_pre_substitution() -> None:
    """Every line is 12 columns after substituting ``{E}`` with a 1-char sample.

    Pre-substitution widths vary because ``{E}`` is a 3-char placeholder;
    the invariant the renderer depends on is post-substitution width.
    """
    for species, frames in BODIES.items():
        for i, frame in enumerate(frames):
            for j, line in enumerate(frame):
                substituted = line.replace("{E}", "X")
                assert len(substituted) == 12, (
                    f"{species}[{i}][{j}] width={len(substituted)} != 12: "
                    f"{substituted!r} (raw: {line!r})"
                )


def test_all_hats_exactly_12_chars() -> None:
    """Every hat line is 12 columns — including the ``none`` sentinel."""
    for name, line in HAT_LINES.items():
        assert len(line) == 12, (
            f"HAT_LINES[{name!r}] width={len(line)} != 12: {line!r}"
        )


def test_eyes_table_has_6_single_char_glyphs() -> None:
    """6 eye glyphs; each is a single Python character.

    (Display width can still be off for full-width glyphs on some
    terminals — we only enforce the Python-string-length invariant here,
    matching how the TS source treats ``Eye`` as a single codepoint.)
    """
    assert len(EYES) == 6
    for eye in EYES:
        assert len(eye) == 1, f"eye {eye!r} is {len(eye)} chars"


# ---------------------------------------------------------------------------
# compose_sprite — eye + hat composition.
# ---------------------------------------------------------------------------


def test_eye_substitution() -> None:
    """``{E}`` placeholders are replaced with the given eye character."""
    rendered = compose_sprite(
        species="duck", frame_idx=0, eye="·", hat="none",
    )
    for line in rendered:
        assert "{E}" not in line, f"unreplaced placeholder in {line!r}"
    # Duck frame 0 has {E} on line 2 — should contain the eye char now.
    assert "·" in rendered[2]


def test_hat_substitution_when_line_0_empty() -> None:
    """When line 0 is whitespace, the hat line is stamped in."""
    rendered = compose_sprite(
        species="duck", frame_idx=0, eye="·", hat="crown",
    )
    assert rendered[0] == HAT_LINES["crown"]


def test_hat_skipped_when_line_0_occupied() -> None:
    """Dragon frame 2 uses line 0 for idle flourish (``   ~    ~   ``) —
    the hat must NOT overwrite it."""
    # Sanity: frame 2 of dragon has non-blank line 0.
    assert BODIES["dragon"][2][0].strip() != ""
    rendered = compose_sprite(
        species="dragon", frame_idx=2, eye="·", hat="crown",
    )
    # Line 0 stays as the original (with {E} substituted — there's no
    # {E} on that line, so it's unchanged).
    assert rendered[0] == BODIES["dragon"][2][0]
    assert rendered[0] != HAT_LINES["crown"]


def test_compose_returns_5_line_tuple() -> None:
    """Rendered output is always a 5-tuple of strings."""
    rendered = compose_sprite(
        species="cat", frame_idx=0, eye="✦", hat="wizard",
    )
    assert isinstance(rendered, tuple)
    assert len(rendered) == 5
    for line in rendered:
        assert isinstance(line, str)


def test_hat_none_leaves_line_0_alone() -> None:
    """``hat="none"`` never stamps — line 0 stays whatever the frame said."""
    rendered = compose_sprite(
        species="duck", frame_idx=0, eye="·", hat="none",
    )
    assert rendered[0] == BODIES["duck"][0][0]
