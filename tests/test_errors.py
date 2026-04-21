"""Tests for aura.errors.AuraError hierarchy."""

from __future__ import annotations

from aura.config.schema import AuraConfigError
from aura.core.llm import (
    MissingCredentialError,
    MissingProviderDependencyError,
    UnknownModelSpecError,
)
from aura.errors import AuraError


def test_aura_error_is_exception() -> None:
    assert issubclass(AuraError, Exception)


def test_aura_config_error_is_aura_error() -> None:
    assert issubclass(AuraConfigError, AuraError)


def test_llm_errors_are_aura_error_descendants() -> None:
    assert issubclass(UnknownModelSpecError, AuraError)
    assert issubclass(MissingProviderDependencyError, AuraError)
    assert issubclass(MissingCredentialError, AuraError)


def test_catching_aura_error_catches_all_subclasses() -> None:
    import pytest

    with pytest.raises(AuraError):
        raise AuraConfigError(source="test", detail="boom")

    with pytest.raises(AuraError):
        raise UnknownModelSpecError(source="spec", detail="bogus")


def test_aura_error_does_not_catch_stdlib_exceptions() -> None:
    import pytest

    with pytest.raises(ValueError):
        raise ValueError("not an AuraError")

    assert not isinstance(ValueError("x"), AuraError)
