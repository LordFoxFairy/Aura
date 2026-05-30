"""Invariant guard — ``LoopState`` must NOT carry an untyped ``custom``
scratchpad dict.

Phase 1 Task 7 deleted the legacy ``LoopState.custom: dict[str, Any]``
field after Tasks 3-6 migrated every consumer onto a typed
:class:`aura.application.loop_state.LoopSlots` slot. This test pins the deletion
so a future refactor can't silently re-introduce the dict (which would
re-open the untyped escape hatch the migration spent five tasks
closing).

If this fails: do NOT add ``custom`` back. Add a typed field on
:class:`LoopSlots` and route the new state through there.
"""

from __future__ import annotations

import dataclasses

from aura.application.loop_state import LoopState


def test_loop_state_has_no_custom_attribute() -> None:
    """No instance-level ``custom`` attribute exists on a fresh LoopState.

    Guards against the legacy untyped scratchpad slipping back in via
    a default-factory dict. The typed ``state.slots`` slot bag is the
    only sanctioned home for transient per-session state.
    """
    state = LoopState()
    assert not hasattr(state, "custom"), (
        "LoopState.custom was deleted in Phase 1 Task 7. New transient "
        "state must land on a typed LoopSlots field, not on a dict."
    )


def test_loop_state_dataclass_has_no_custom_field() -> None:
    """No dataclass field named ``custom`` is declared on LoopState.

    Belt-and-suspenders against ``hasattr`` returning False because the
    default factory raised — the field declaration itself must be gone.
    """
    field_names = {f.name for f in dataclasses.fields(LoopState)}
    assert "custom" not in field_names, (
        f"LoopState dataclass still declares a 'custom' field: "
        f"{field_names!r}. Route new state through LoopSlots instead."
    )
