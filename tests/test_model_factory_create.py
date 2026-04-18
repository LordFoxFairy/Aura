"""Tests for ModelFactory.create — lazy SDK loading + secret resolution."""

from __future__ import annotations

from typing import Any

import pytest

from aura.config.schema import AuraConfigError, ProviderConfig
from aura.core.llm import (
    MissingCredentialError,
    MissingProviderDependencyError,
    ModelFactory,
)

# ---------------------------------------------------------------------------
# Stub classes — record kwargs, don't hit the network
# ---------------------------------------------------------------------------


class _StubOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _StubAnthropic:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _StubOllama:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _stub_kwargs(model: Any) -> dict[str, Any]:
    """Extract recorded kwargs from a stub model instance."""
    return model.kwargs  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_create_openai_happy_path_uses_default_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.core import llm

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_load_openai_class", lambda: _StubOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    provider = ProviderConfig(name="openai", protocol="openai")
    model, protocol = ModelFactory.create(provider, "gpt-4o-mini")

    assert protocol == "openai"
    assert isinstance(model, _StubOpenAI)
    kw = _stub_kwargs(model)
    assert kw["model"] == "gpt-4o-mini"
    assert kw["api_key"] == "sk-test"
    assert "base_url" not in kw


def test_create_openai_with_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.core import llm

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_load_openai_class", lambda: _StubOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    provider = ProviderConfig(
        name="openrouter",
        protocol="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    )
    model, protocol = ModelFactory.create(provider, "mistral-7b")

    assert protocol == "openai"
    kw = _stub_kwargs(model)
    assert kw["base_url"] == "https://openrouter.ai/api/v1"
    assert kw["api_key"] == "or-key"
    assert kw["model"] == "mistral-7b"


def test_create_openai_plaintext_api_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.core import llm

    monkeypatch.setattr(llm, "_load_openai_class", lambda: _StubOpenAI)
    # No env var set — plaintext key on provider should be used
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = ProviderConfig(name="openai", protocol="openai", api_key="plaintext-key")
    model, protocol = ModelFactory.create(provider, "gpt-4o-mini")

    assert _stub_kwargs(model)["api_key"] == "plaintext-key"
    assert protocol == "openai"


def test_create_openai_api_key_env_preferred_over_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.core import llm

    monkeypatch.setattr(llm, "_load_openai_class", lambda: _StubOpenAI)
    monkeypatch.setenv("CUSTOM_KEY", "custom")
    monkeypatch.setenv("OPENAI_API_KEY", "default")

    provider = ProviderConfig(name="openai", protocol="openai", api_key_env="CUSTOM_KEY")
    model, protocol = ModelFactory.create(provider, "gpt-4o-mini")

    assert _stub_kwargs(model)["api_key"] == "custom"


def test_create_anthropic_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.core import llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_load_anthropic_class", lambda: _StubAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

    provider = ProviderConfig(name="anthropic", protocol="anthropic")
    model, protocol = ModelFactory.create(provider, "claude-3-5-sonnet-20241022")

    assert protocol == "anthropic"
    assert isinstance(model, _StubAnthropic)
    kw = _stub_kwargs(model)
    assert kw["model"] == "claude-3-5-sonnet-20241022"
    assert kw["api_key"] == "ant-key"


def test_create_ollama_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.core import llm

    monkeypatch.setattr(llm, "_load_ollama_class", lambda: _StubOllama)

    provider = ProviderConfig(name="ollama", protocol="ollama")
    model, protocol = ModelFactory.create(provider, "llama3")

    assert protocol == "ollama"
    assert isinstance(model, _StubOllama)
    kw = _stub_kwargs(model)
    assert kw["model"] == "llama3"
    assert "api_key" not in kw


def test_create_ollama_with_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.core import llm

    monkeypatch.setattr(llm, "_load_ollama_class", lambda: _StubOllama)

    provider = ProviderConfig(
        name="ollama-remote",
        protocol="ollama",
        base_url="http://remote:11434",
    )
    model, protocol = ModelFactory.create(provider, "llama3")

    kw = _stub_kwargs(model)
    assert kw["base_url"] == "http://remote:11434"
    assert "api_key" not in kw
    assert kw["model"] == "llama3"


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


def test_create_missing_default_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.core import llm

    monkeypatch.setattr(llm, "_load_openai_class", lambda: _StubOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = ProviderConfig(name="openai", protocol="openai")
    with pytest.raises(MissingCredentialError) as exc_info:
        ModelFactory.create(provider, "gpt-4o-mini")

    assert "OPENAI_API_KEY" in exc_info.value.detail


def test_create_missing_api_key_env_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.core import llm

    monkeypatch.setattr(llm, "_load_openai_class", lambda: _StubOpenAI)
    monkeypatch.delenv("CUSTOM_KEY", raising=False)

    provider = ProviderConfig(name="openai", protocol="openai", api_key_env="CUSTOM_KEY")
    with pytest.raises(MissingCredentialError) as exc_info:
        ModelFactory.create(provider, "gpt-4o-mini")

    assert "CUSTOM_KEY" in exc_info.value.detail


def test_create_missing_sdk_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.core import llm

    def _boom() -> type:
        raise MissingProviderDependencyError(
            source="provider sdk",
            detail="langchain_openai not installed. Run: pip install 'aura[openai]'",
        )

    monkeypatch.setattr(llm, "_load_openai_class", _boom)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    provider = ProviderConfig(name="openai", protocol="openai")
    with pytest.raises(MissingProviderDependencyError) as exc_info:
        ModelFactory.create(provider, "gpt-4o-mini")

    assert "pip install" in exc_info.value.detail
    assert "aura[openai]" in exc_info.value.detail


def test_create_errors_are_aura_config_error_subclasses() -> None:
    assert issubclass(MissingProviderDependencyError, AuraConfigError)
    assert issubclass(MissingCredentialError, AuraConfigError)


def test_create_empty_plaintext_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty-string api_key must NOT be forwarded — treat as missing."""
    from aura.core import llm

    monkeypatch.setattr(llm, "_load_openai_class", lambda: _StubOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CUSTOM_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_KEY", "from-env")

    provider = ProviderConfig(name="x", protocol="openai", api_key="", api_key_env="CUSTOM_KEY")
    model, _ = ModelFactory.create(provider, "gpt-4o-mini")

    assert isinstance(model, _StubOpenAI)
    assert model.kwargs["api_key"] == "from-env"  # not ""


def test_create_empty_plaintext_api_key_with_no_fallback_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.core import llm

    monkeypatch.setattr(llm, "_load_openai_class", lambda: _StubOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = ProviderConfig(name="x", protocol="openai", api_key="")
    with pytest.raises(MissingCredentialError) as exc_info:
        ModelFactory.create(provider, "gpt-4o-mini")
    assert "OPENAI_API_KEY" in exc_info.value.detail
