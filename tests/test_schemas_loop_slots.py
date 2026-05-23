"""Tests for ``LoopSlots`` and its supporting frozen dataclasses.

Phase 1 Task 1 — schema contracts only. Migration of legacy untyped
scratchpad consumers landed in Tasks 3-6; Task 7 deleted the dict
outright. These tests assert the type shape, defaults, frozenness,
and the ``replace`` ergonomic.
"""

from __future__ import annotations

import dataclasses

import pytest

from aura.schemas.state import (
    BuddyState,
    LoopSlots,
    SkillRestrictLease,
    TokenStats,
)
from aura.schemas.todos import TodoItem


def test_loop_slots_constructible_with_defaults() -> None:
    """``LoopSlots()`` returns a usable, fully-defaulted instance.

    Spec §3.1 — every field has a default that yields an empty/zero
    starting state, so the loop can construct fresh ``LoopSlots``
    without arguments at session start.
    """
    slots = LoopSlots()

    assert slots.token_stats == TokenStats()
    assert slots.turn_denials == []
    assert slots.todos == []
    assert slots.perm_dedup_cache == {}
    assert slots.preserved_invoked_skills == []
    assert slots.invoked_skills == []
    assert slots.consecutive_compact_failures == 0
    assert slots.active_team is None
    assert slots.buddy == BuddyState()
    assert slots.skill_restrict_leases == []


def test_loop_slots_has_ten_fields_per_spec() -> None:
    """Spec §3.1 — every named field is cross-turn state. No extras.

    ``mood`` from the spec sketch is realised as ``buddy: BuddyState``
    (a frozen dataclass packing ``mood`` + ``last_event_ts`` +
    ``had_recent_error`` so the buddy state machine has the room it
    needs without spreading three coupled fields across the slot bag).
    Likewise ``skill_restrict_lease`` is plural (``skill_restrict_leases``):
    multiple skills can stack leases, so the slot is a list.
    """
    expected = {
        "token_stats",
        "turn_denials",
        "todos",
        "perm_dedup_cache",
        "preserved_invoked_skills",
        "invoked_skills",
        "consecutive_compact_failures",
        "active_team",
        "buddy",
        "skill_restrict_leases",
    }
    actual = {f.name for f in dataclasses.fields(LoopSlots)}
    assert actual == expected


def test_loop_slots_is_frozen() -> None:
    """``LoopSlots`` is a frozen dataclass — direct field assignment
    raises ``FrozenInstanceError``. Mutation flows through
    ``dataclasses.replace`` (or in-place mutation of contained
    mutable collections, which is intentional)."""
    slots = LoopSlots()
    with pytest.raises(dataclasses.FrozenInstanceError):
        slots.active_team = "team-a"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        slots.buddy = BuddyState(mood="happy")  # type: ignore[misc]


def test_loop_slots_replace_returns_new_instance() -> None:
    """``dataclasses.replace`` is the supported way to override a field
    while keeping the frozen invariant. Returns a NEW instance; old
    instance is untouched."""
    original = LoopSlots()
    updated = dataclasses.replace(
        original, active_team="team-a", buddy=BuddyState(mood="busy"),
    )

    assert updated.active_team == "team-a"
    assert updated.buddy.mood == "busy"
    # Original untouched.
    assert original.active_team is None
    assert original.buddy.mood == "idle"
    # Different instances.
    assert updated is not original


def test_loop_slots_default_factories_isolate_per_instance() -> None:
    """Mutable defaults (lists, dicts) must use ``field(default_factory=...)``;
    if they used a shared default, two ``LoopSlots()`` instances would
    alias their containers and mutating one would leak into the other."""
    a = LoopSlots()
    b = LoopSlots()

    a.turn_denials.append("sentinel")  # type: ignore[arg-type]
    a.todos.append(TodoItem(content="x", status="pending", active_form="x"))
    a.perm_dedup_cache["k"] = "v"  # type: ignore[assignment]
    a.invoked_skills.append("skill_a")  # type: ignore[arg-type]
    a.preserved_invoked_skills.append("preserved_a")  # type: ignore[arg-type]

    assert b.turn_denials == []
    assert b.todos == []
    assert b.perm_dedup_cache == {}
    assert b.invoked_skills == []
    assert b.preserved_invoked_skills == []


def test_loop_slots_collections_mutable_in_place_under_frozen() -> None:
    """``frozen=True`` blocks rebinding the slot itself but does NOT
    block mutating the contained list/dict. The loop relies on this:
    ``slots.turn_denials.clear()`` (§4 step 1) must work even though
    ``LoopSlots`` is frozen.
    """
    slots = LoopSlots()
    slots.turn_denials.append("d1")  # type: ignore[arg-type]
    slots.turn_denials.clear()
    assert slots.turn_denials == []


def test_tokenstats_is_frozen_and_zero_default() -> None:
    """``TokenStats`` is a frozen dataclass; default instance is all zeros.

    Field shape mirrors the per-turn token-usage counters
    ``make_usage_tracking_hook`` writes; wire/status_bar consumers read
    these same fields off ``state.slots.token_stats`` (Task 3 migration).
    """
    ts = TokenStats()
    assert ts.last_input_tokens == 0
    assert ts.last_output_tokens == 0
    assert ts.last_cache_read_tokens == 0
    assert ts.total_input_tokens == 0
    assert ts.total_output_tokens == 0
    assert ts.total_cache_read_tokens == 0
    assert ts.turn_count == 0

    with pytest.raises(dataclasses.FrozenInstanceError):
        ts.last_input_tokens = 5  # type: ignore[misc]


def test_tokenstats_replace_yields_new_instance() -> None:
    ts = TokenStats()
    bumped = dataclasses.replace(ts, last_input_tokens=42, turn_count=1)
    assert bumped.last_input_tokens == 42
    assert bumped.turn_count == 1
    assert ts.last_input_tokens == 0
    assert ts.turn_count == 0


def test_skill_restrict_lease_is_frozen_with_required_fields() -> None:
    """``SkillRestrictLease`` mirrors the runtime shape used today
    (``install_turn: int, tools: frozenset[str]``). Frozen so once a
    lease is recorded the audit shape can't be retroactively edited.
    """
    lease = SkillRestrictLease(install_turn=3, tools=frozenset({"read_file"}))
    assert lease.install_turn == 3
    assert lease.tools == frozenset({"read_file"})

    with pytest.raises(dataclasses.FrozenInstanceError):
        lease.install_turn = 5  # type: ignore[misc]


def test_loop_slots_exported_from_schemas_state_module() -> None:
    """``LoopSlots`` must be importable from ``aura.schemas.state``
    (its canonical home per spec §3.1)."""
    from aura.schemas import state as state_mod

    assert hasattr(state_mod, "LoopSlots")
    assert hasattr(state_mod, "TokenStats")
    assert hasattr(state_mod, "SkillRestrictLease")
    assert hasattr(state_mod, "BuddyState")


def test_buddy_state_defaults_match_idle_observer() -> None:
    """``BuddyState()`` is the "no events yet" shape — :func:`get_mood`
    on a fresh :class:`LoopState` must return ``"idle"`` because the
    default mood is ``"idle"`` (matches the pre-migration lazy-init
    contract)."""
    bs = BuddyState()
    assert bs.mood == "idle"
    assert bs.last_event_ts == 0.0
    assert bs.had_recent_error is False


def test_buddy_state_is_frozen() -> None:
    bs = BuddyState()
    with pytest.raises(dataclasses.FrozenInstanceError):
        bs.mood = "happy"  # type: ignore[misc]
