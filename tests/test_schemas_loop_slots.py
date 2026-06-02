"""Tests for ``LoopSlots`` and its supporting frozen dataclasses.

Phase 1 Task 1 — schema contracts only. Migration of legacy untyped
scratchpad consumers landed in Tasks 3-6; Task 7 deleted the dict
outright. These tests assert the type shape, defaults, frozenness,
and the ``replace`` ergonomic.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from aura.application.loop_state import LoopSlots
from aura.domain.state_values import BuddyState, SkillRestrictLease, TokenStats
from aura.domain.todos import TodoItem


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
    obj: Any = slots
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.active_team = "team-a"
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.buddy = BuddyState(mood="happy")


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

    sentinel: Any = "sentinel"
    a.turn_denials.append(sentinel)
    a.todos.append(TodoItem(content="x", status="pending", active_form="x"))
    dedup_key: Any = "k"
    dedup_val: Any = "v"
    a.perm_dedup_cache[dedup_key] = dedup_val
    skill_a: Any = "skill_a"
    a.invoked_skills.append(skill_a)
    preserved_a: Any = "preserved_a"
    a.preserved_invoked_skills.append(preserved_a)

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
    d1: Any = "d1"
    slots.turn_denials.append(d1)
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

    ts_obj: Any = ts
    with pytest.raises(dataclasses.FrozenInstanceError):
        ts_obj.last_input_tokens = 5


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

    lease_obj: Any = lease
    with pytest.raises(dataclasses.FrozenInstanceError):
        lease_obj.install_turn = 5


def test_loop_slots_and_pure_values_at_canonical_homes() -> None:
    """``LoopSlots`` lives in ``aura.application.loop_state``; the pure
    value types it composes live in ``aura.domain.state_values``."""
    from aura.application import loop_state as loop_mod
    from aura.domain import state_values as values_mod

    assert hasattr(loop_mod, "LoopSlots")
    assert hasattr(values_mod, "TokenStats")
    assert hasattr(values_mod, "SkillRestrictLease")
    assert hasattr(values_mod, "BuddyState")


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
    bs_obj: Any = bs
    with pytest.raises(dataclasses.FrozenInstanceError):
        bs_obj.mood = "happy"
