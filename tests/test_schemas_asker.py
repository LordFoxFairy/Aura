"""Tests for ``AskerPrompt`` + ``AskerResponse`` in ``aura.schemas.permissions``.

Phase 5 Task 3 — schema contracts only. The CLI / IPC / subagent askers
migrate to consume / produce these in Tasks 4-6.

Spec §6 — frozen dataclass shapes:

- ``AskerPrompt(tool, args_preview, rule_hint, is_destructive, request_id)``
- ``AskerResponse(choice, request_id)`` where
  ``choice: Literal["yes", "yes-always", "no", "no-always"]``

Invariants enforced at construction so bugs surface at the asker
boundary, not later when the gate tries to translate the response into
a ``Decision``:

- ``request_id`` non-empty (IPC correlation requires a stable id).
- ``AskerPrompt.tool`` non-empty (the asker widget needs a label).
- ``AskerResponse.choice`` is constrained to the four literal values
  by static typing; runtime construction with a foreign value raises.
"""

from __future__ import annotations

import dataclasses

import pytest

from aura.schemas.permissions import AskerPrompt, AskerResponse


def test_asker_prompt_constructs_with_all_fields() -> None:
    prompt = AskerPrompt(
        tool="bash",
        args_preview="rm -rf /tmp/foo",
        rule_hint="bash(rm:*)",
        is_destructive=True,
        request_id="req-123",
    )
    assert prompt.tool == "bash"
    assert prompt.args_preview == "rm -rf /tmp/foo"
    assert prompt.rule_hint == "bash(rm:*)"
    assert prompt.is_destructive is True
    assert prompt.request_id == "req-123"


def test_asker_prompt_is_frozen() -> None:
    prompt = AskerPrompt(
        tool="bash",
        args_preview="ls",
        rule_hint="bash(ls)",
        is_destructive=False,
        request_id="req-1",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        prompt.tool = "other"  # type: ignore[misc]  # rebinding/mutating frozen field for test


def test_asker_prompt_rejects_empty_request_id() -> None:
    """``request_id`` is the IPC correlation key — empty would mean the
    desktop frontend can't match the response back to the request."""
    with pytest.raises(ValueError, match="request_id"):
        AskerPrompt(
            tool="bash",
            args_preview="ls",
            rule_hint="bash(ls)",
            is_destructive=False,
            request_id="",
        )


def test_asker_prompt_rejects_empty_tool() -> None:
    """The asker widget renders ``tool`` as the prompt label — empty
    would produce an unrenderable prompt."""
    with pytest.raises(ValueError, match="tool"):
        AskerPrompt(
            tool="",
            args_preview="ls",
            rule_hint="bash(ls)",
            is_destructive=False,
            request_id="req-1",
        )


@pytest.mark.parametrize("choice", ["yes", "yes-always", "no", "no-always"])
def test_asker_response_accepts_each_literal_choice(choice: str) -> None:
    """All four ``Literal`` values must construct cleanly. Spec §6 maps
    them onto Decision factories in the gate (Task 8)."""
    # deliberately off-type arg to exercise path
    response = AskerResponse(choice=choice, request_id="req-1")  # type: ignore[arg-type]
    assert response.choice == choice
    assert response.request_id == "req-1"


def test_asker_response_is_frozen() -> None:
    response = AskerResponse(choice="yes", request_id="req-1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        response.choice = "no"  # type: ignore[misc]  # rebinding/mutating frozen field for test


def test_asker_response_rejects_empty_request_id() -> None:
    """``request_id`` must echo the prompt's id for IPC correlation —
    an empty echo means the response can't be routed."""
    with pytest.raises(ValueError, match="request_id"):
        AskerResponse(choice="yes", request_id="")


def test_asker_response_rejects_unknown_choice() -> None:
    """Static typing constrains ``choice`` to four literals; runtime
    construction with a foreign string raises so a malformed IPC
    payload surfaces at deserialization, not at the gate."""
    with pytest.raises(ValueError, match="choice"):
        # deliberately off-type arg to exercise path
        AskerResponse(choice="maybe", request_id="req-1")  # type: ignore[arg-type]


def test_asker_types_exported_from_schemas_permissions() -> None:
    from aura.schemas import permissions as perm_mod

    assert hasattr(perm_mod, "AskerPrompt")
    assert hasattr(perm_mod, "AskerResponse")


def test_asker_types_exported_from_schemas_init() -> None:
    """``aura.schemas`` is the leaf layer — re-exporting keeps the
    asker boundary discoverable from the top-level data namespace."""
    import aura.schemas as schemas_mod

    assert hasattr(schemas_mod, "AskerPrompt")
    assert hasattr(schemas_mod, "AskerResponse")


def test_request_id_pairing_round_trip() -> None:
    """A response echoes the prompt's id verbatim — the contract is a
    string-equality match, no normalization. The gate (Task 8) asserts
    pairing before translating choice into a Decision."""
    request_id = "req-abc-123"
    prompt = AskerPrompt(
        tool="write_file",
        args_preview="path=/tmp/x",
        rule_hint="write_file(/tmp/*)",
        is_destructive=True,
        request_id=request_id,
    )
    response = AskerResponse(choice="yes-always", request_id=prompt.request_id)
    assert response.request_id == prompt.request_id
