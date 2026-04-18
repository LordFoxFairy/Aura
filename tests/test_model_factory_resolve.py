"""Tests for ModelFactory.resolve — Task 13."""

from __future__ import annotations

import pytest

from aura.config.schema import AuraConfig, AuraConfigError
from aura.core.llm import ModelFactory, UnknownModelSpecError


def _cfg(providers: list[dict[str, str]], router: dict[str, str]) -> AuraConfig:
    return AuraConfig.model_validate({"providers": providers, "router": router})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_OPENAI_PROVIDER = {"name": "openai", "protocol": "openai"}
_OPENROUTER_PROVIDER = {"name": "openrouter", "protocol": "openai"}
_ANTHROPIC_PROVIDER = {"name": "anthropic-direct", "protocol": "anthropic"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resolve_router_alias_default() -> None:
    cfg = _cfg(
        providers=[_OPENAI_PROVIDER],
        router={"default": "openai:gpt-4o-mini"},
    )
    provider, model_name = ModelFactory.resolve("default", cfg=cfg)
    assert model_name == "gpt-4o-mini"
    assert provider.name == "openai"
    assert provider.protocol == "openai"


def test_resolve_router_alias_custom() -> None:
    cfg = _cfg(
        providers=[_OPENAI_PROVIDER, _OPENROUTER_PROVIDER],
        router={
            "default": "openai:gpt-4o",
            "opus": "openrouter:anthropic/claude-opus-4",
        },
    )
    provider, model_name = ModelFactory.resolve("opus", cfg=cfg)
    assert provider.name == "openrouter"
    assert model_name == "anthropic/claude-opus-4"


def test_resolve_direct_provider_colon_model() -> None:
    cfg = _cfg(
        providers=[_OPENAI_PROVIDER],
        router={"default": "openai:gpt-4o-mini"},
    )
    provider, model_name = ModelFactory.resolve("openai:gpt-4o", cfg=cfg)
    assert provider.name == "openai"
    assert model_name == "gpt-4o"


def test_resolve_preserves_colons_in_model_name() -> None:
    cfg = _cfg(
        providers=[_ANTHROPIC_PROVIDER, _OPENAI_PROVIDER],
        router={"default": "openai:gpt-4o-mini"},
    )
    provider, model_name = ModelFactory.resolve(
        "anthropic-direct:claude-3.5-haiku:20241022", cfg=cfg
    )
    assert provider.name == "anthropic-direct"
    assert model_name == "claude-3.5-haiku:20241022"


def test_resolve_unknown_alias_no_colon_raises() -> None:
    cfg = _cfg(
        providers=[_OPENAI_PROVIDER],
        router={"default": "openai:gpt-4o-mini"},
    )
    with pytest.raises(UnknownModelSpecError) as exc_info:
        ModelFactory.resolve("bogus", cfg=cfg)
    assert "bogus" in str(exc_info.value)


def test_resolve_unknown_provider_direct_raises() -> None:
    cfg = _cfg(
        providers=[_OPENAI_PROVIDER],
        router={"default": "openai:gpt-4o-mini"},
    )
    with pytest.raises(UnknownModelSpecError) as exc_info:
        ModelFactory.resolve("ghost:m", cfg=cfg)
    assert "ghost" in str(exc_info.value)


def test_resolve_error_is_aura_config_error_subclass() -> None:
    assert issubclass(UnknownModelSpecError, AuraConfigError)


def test_resolve_empty_model_name_after_colon() -> None:
    """resolve('openai:') should return (provider, '') — validation is downstream."""
    cfg = _cfg(
        providers=[_OPENAI_PROVIDER],
        router={"default": "openai:gpt-4o-mini"},
    )
    provider, model_name = ModelFactory.resolve("openai:", cfg=cfg)
    assert provider.name == "openai"
    assert model_name == ""
