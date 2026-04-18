"""Tests for aura.core.errors.AuraError hierarchy."""

from __future__ import annotations

from aura.config.schema import AuraConfigError
from aura.core.errors import AuraError
from aura.core.hooks.budget import MaxTurnsExceeded
from aura.core.llm import (
    MissingCredentialError,
    MissingProviderDependencyError,
    UnknownModelSpecError,
)


def test_aura_error_is_exception() -> None:
    assert issubclass(AuraError, Exception)


def test_aura_config_error_is_aura_error() -> None:
    assert issubclass(AuraConfigError, AuraError)


def test_max_turns_exceeded_is_aura_error() -> None:
    assert issubclass(MaxTurnsExceeded, AuraError)


def test_llm_errors_are_aura_error_descendants() -> None:
    # Via transitive inheritance through AuraConfigError
    assert issubclass(UnknownModelSpecError, AuraError)
    assert issubclass(MissingProviderDependencyError, AuraError)
    assert issubclass(MissingCredentialError, AuraError)


def test_catching_aura_error_catches_all_subclasses() -> None:
    import pytest

    with pytest.raises(AuraError):
        raise AuraConfigError(source="test", detail="boom")

    with pytest.raises(AuraError):
        raise MaxTurnsExceeded("max_turns=10 reached")

    with pytest.raises(AuraError):
        raise UnknownModelSpecError(source="spec", detail="bogus")


def test_aura_error_does_not_catch_stdlib_exceptions() -> None:
    import pytest

    # Importing here for clarity: AuraError should NOT be caught by unrelated
    # stdlib exception types, and vice versa.
    with pytest.raises(ValueError):
        raise ValueError("not an AuraError")

    # Verify a plain ValueError is not an AuraError
    assert not isinstance(ValueError("x"), AuraError)
