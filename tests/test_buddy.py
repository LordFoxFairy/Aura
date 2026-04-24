"""Tests for aura.cli.buddy — deterministic pet + mood observer.

TDD for T2-B: a status-bar pet whose species + rarity are a pure function
of the user's ``Path.home().name``, and whose mood reflects the four
stock turn-cycle events (idle / thinking / happy / worried).

Tests only touch the pure-logic module — no prompt_toolkit dependencies,
no REPL wiring, no journal I/O. Wiring tests live in test_repl /
test_event_logger_hooks.
"""

from __future__ import annotations

from typing import Any

import pytest

from aura.cli import buddy
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult

# ---------------------------------------------------------------------------
# Deterministic generation
# ---------------------------------------------------------------------------


def test_buddy_gen_is_deterministic() -> None:
    """Same seed → same (species, rarity, emoji). No randomness leak."""
    a = buddy.generate_buddy("alice")
    b = buddy.generate_buddy("alice")
    assert a.species == b.species
    assert a.rarity == b.rarity
    assert a.emoji == b.emoji


def test_buddy_gen_different_users_different_species() -> None:
    """Across 10 distinct seeds we should see >= 2 distinct species.

    Statistical (not exhaustive) — with a 10-species table and mulberry32
    hashing, 10 distinct seeds collapsing to one species would be a
    spectacular hash failure (~1e-10).
    """
    seeds = [f"user-{i}" for i in range(10)]
    species = {buddy.generate_buddy(s).species for s in seeds}
    assert len(species) >= 2, (
        f"expected >=2 distinct species across 10 seeds, got {species}"
    )


def test_rarity_distribution_roughly_matches_weights() -> None:
    """1000 distinct seeds; common>=50%, legendary<=7%.

    Loose bounds so the test is non-flaky — weights are common=60,
    uncommon=25, rare=12, legendary=3. Even on a streak, 1000 draws
    shouldn't blow through these bounds.
    """
    counts: dict[str, int] = {}
    for i in range(1000):
        r = buddy.generate_buddy(f"seed-{i}").rarity
        counts[r] = counts.get(r, 0) + 1
    assert counts.get("common", 0) >= 500, counts
    assert counts.get("legendary", 0) <= 70, counts


def test_generate_buddy_handles_empty_seed() -> None:
    """Pathological empty home basename falls back to ``"default"``; still
    produces a valid buddy (no exception, species in the known set)."""
    b = buddy.generate_buddy("")
    assert b.species in buddy.SPECIES
    assert b.rarity in buddy.RARITIES
    assert b.emoji  # non-empty


# ---------------------------------------------------------------------------
# Opt-out
# ---------------------------------------------------------------------------


def test_opt_out_via_env_returns_empty_status_frag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AURA_NO_BUDDY=1`` disables the status fragment entirely."""
    monkeypatch.setenv("AURA_NO_BUDDY", "1")
    state = LoopState()
    assert buddy.buddy_status_fragment(state=state, seed="alice") == ""


def test_settings_json_opt_out() -> None:
    """``ui.buddy_enabled=False`` suppresses the fragment even without the env var."""
    state = LoopState()
    assert buddy.buddy_status_fragment(
        state=state, seed="alice", enabled=False,
    ) == ""


def test_status_fragment_present_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without opt-out, fragment is non-empty and contains the species emoji."""
    monkeypatch.delenv("AURA_NO_BUDDY", raising=False)
    state = LoopState()
    frag = buddy.buddy_status_fragment(state=state, seed="alice")
    assert frag
    b = buddy.generate_buddy("alice")
    assert b.emoji in frag


# ---------------------------------------------------------------------------
# Mood observer
# ---------------------------------------------------------------------------


def _fake_ai_message(*, had_errors: bool = False) -> Any:
    """Minimal duck-typed AIMessage stand-in — observe_post_model only cares
    that the argument exists; it reads mood signal from ``state.custom``."""
    class _M:
        pass
    return _M()


def test_mood_idle_by_default() -> None:
    state = LoopState()
    assert buddy.get_mood(state) == "idle"


async def test_mood_happy_after_successful_turn() -> None:
    state = LoopState()
    await buddy.observe_post_model(state=state)
    assert buddy.get_mood(state) == "happy"


async def test_mood_worried_after_tool_error() -> None:
    state = LoopState()
    failing = ToolResult(ok=False, error="boom")
    await buddy.observe_post_tool(state=state, result=failing)
    assert buddy.get_mood(state) == "worried"


async def test_mood_persists_until_next_event() -> None:
    """Worried mood survives multiple post_model calls if no new tool error
    arrives — claude-code's observer has the same stickiness so the mood
    doesn't flicker between 'happy' and 'worried' every turn after an error."""
    state = LoopState()
    failing = ToolResult(ok=False, error="boom")
    await buddy.observe_post_tool(state=state, result=failing)
    assert buddy.get_mood(state) == "worried"
    # Two consecutive post_model events don't reset — they just keep worrying.
    await buddy.observe_post_model(state=state)
    assert buddy.get_mood(state) == "worried"
    await buddy.observe_post_model(state=state)
    assert buddy.get_mood(state) == "worried"


async def test_successful_tool_clears_worry() -> None:
    """A successful post_tool after an error should let the next post_model
    flip back to happy — otherwise worry would be permanent."""
    state = LoopState()
    bad = ToolResult(ok=False, error="boom")
    await buddy.observe_post_tool(state=state, result=bad)
    good = ToolResult(ok=True, output={"done": True})
    await buddy.observe_post_tool(state=state, result=good)
    await buddy.observe_post_model(state=state)
    assert buddy.get_mood(state) == "happy"


def test_clear_session_resets_mood() -> None:
    state = LoopState()
    state.custom["_buddy_state"] = {"mood": "worried", "last_event_ts": 1.0}
    buddy.reset(state)
    assert buddy.get_mood(state) == "idle"


def test_status_fragment_includes_species_emoji_and_mood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendered status fragment contains the species emoji AND a mood glyph."""
    monkeypatch.delenv("AURA_NO_BUDDY", raising=False)
    state = LoopState()
    # Put the buddy in worried mood so we can look for its glyph.
    state.custom["_buddy_state"] = {"mood": "worried", "last_event_ts": 0.0}
    frag = buddy.buddy_status_fragment(state=state, seed="alice")
    b = buddy.generate_buddy("alice")
    assert b.emoji in frag
    # Mood indicator is some recognizable suffix per-mood; "worried" must
    # produce a glyph different from the default happy/idle one.
    idle_state = LoopState()
    idle_frag = buddy.buddy_status_fragment(state=idle_state, seed="alice")
    assert frag != idle_frag


# ---------------------------------------------------------------------------
# Species table sanity — the spec pins species + emoji count.
# ---------------------------------------------------------------------------


def test_species_table_has_at_least_10_entries() -> None:
    assert len(buddy.SPECIES) >= 10
    assert len(buddy.SPECIES) == len(buddy.SPECIES_EMOJI)
    for sp in buddy.SPECIES:
        assert sp in buddy.SPECIES_EMOJI
        assert buddy.SPECIES_EMOJI[sp]  # non-empty


def test_rarity_weights_sum_to_100() -> None:
    assert sum(buddy.RARITY_WEIGHTS.values()) == 100
    assert set(buddy.RARITY_WEIGHTS) == set(buddy.RARITIES)
