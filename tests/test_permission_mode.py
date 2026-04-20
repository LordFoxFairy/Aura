"""Tests for aura.core.permissions.mode — Mode literal alias + DEFAULT_MODE."""

from __future__ import annotations

from typing import get_args

from aura.core.permissions.mode import DEFAULT_MODE, Mode


def test_default_mode_is_default_string() -> None:
    assert DEFAULT_MODE == "default"


def test_mode_literal_accepts_default_and_bypass() -> None:
    # Mode is a Literal alias — the two string values are its runtime args.
    assert set(get_args(Mode)) == {"default", "bypass"}


def test_mode_alias_matches_schemas_permissions_mode_field() -> None:
    # Parallel type-equivalence with PermissionsConfig.mode: same Literal values.
    from aura.schemas.permissions import PermissionsConfig

    schema_field = PermissionsConfig.model_fields["mode"].annotation
    assert set(get_args(schema_field)) == set(get_args(Mode))
