"""ModelFactory — resolves model specs to (ProviderConfig, model_name) and creates LLM instances."""
from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel

from aura.config.schema import AuraConfig, AuraConfigError, ProviderConfig


class UnknownModelSpecError(AuraConfigError):
    """Raised when a model spec can't be resolved to a known provider."""


class MissingProviderDependencyError(AuraConfigError):
    """Raised when the LangChain SDK for a protocol isn't installed."""


class MissingCredentialError(AuraConfigError):
    """Raised when no API key is found for a protocol that requires one."""


# ---------------------------------------------------------------------------
# Lazy import helpers — zero import cost when protocol not used
# ---------------------------------------------------------------------------

def _load_openai_class() -> type[BaseChatModel]:
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        raise MissingProviderDependencyError(
            source="provider sdk",
            detail="langchain_openai not installed. Run: pip install 'aura[openai]'",
        ) from exc
    return ChatOpenAI  # type: ignore[no-any-return]


def _load_anthropic_class() -> type[BaseChatModel]:
    try:
        from langchain_anthropic import ChatAnthropic
    except ModuleNotFoundError as exc:
        raise MissingProviderDependencyError(
            source="provider sdk",
            detail="langchain_anthropic not installed. Run: pip install 'aura[anthropic]'",
        ) from exc
    return ChatAnthropic  # type: ignore[no-any-return]


def _load_ollama_class() -> type[BaseChatModel]:
    try:
        from langchain_ollama import ChatOllama
    except ModuleNotFoundError as exc:
        raise MissingProviderDependencyError(
            source="provider sdk",
            detail="langchain_ollama not installed. Run: pip install 'aura[ollama]'",
        ) from exc
    return ChatOllama  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Secret resolution
# ---------------------------------------------------------------------------

_DEFAULT_KEY_ENV: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
}


def _resolve_api_key(provider: ProviderConfig) -> str | None:
    """Return API key for *provider* or raise MissingCredentialError."""
    if provider.api_key is not None:
        return provider.api_key

    if provider.api_key_env is not None:
        key = os.environ.get(provider.api_key_env)
        if not key:
            raise MissingCredentialError(
                source="env",
                detail=f"${provider.api_key_env} not set for provider {provider.name!r}",
            )
        return key

    default_env = _DEFAULT_KEY_ENV[provider.protocol]
    if default_env is None:
        return None  # ollama doesn't need a key

    key = os.environ.get(default_env)
    if not key:
        raise MissingCredentialError(
            source="env",
            detail=(
                f"${default_env} not set for provider {provider.name!r}"
                f" (protocol {provider.protocol})"
            ),
        )
    return key


# ---------------------------------------------------------------------------
# ModelFactory
# ---------------------------------------------------------------------------

class ModelFactory:
    """Namespace class for model resolution utilities."""

    @staticmethod
    def resolve(spec: str, *, cfg: AuraConfig) -> tuple[ProviderConfig, str]:
        """Resolve *spec* to (ProviderConfig, model_name).

        Resolution rules (spec §4.6):
        1. If *spec* is a router alias, substitute once.
        2. Split on the first ':'. Left = provider name, right = model name.
        3. Unknown alias with no colon → UnknownModelSpecError.
        4. Unknown provider name → UnknownModelSpecError.
        """
        resolved = cfg.router.get(spec, spec)

        provider_name, sep, model_name = resolved.partition(":")

        if not sep:
            # spec was not a router alias AND has no colon
            raise UnknownModelSpecError(
                "model spec",
                f"{spec!r} is not a router alias and not in 'provider:model' form",
            )

        # Linear scan — providers list is short (1-5 in practice)
        for provider in cfg.providers:
            if provider.name == provider_name:
                return (provider, model_name)

        raise UnknownModelSpecError(
            "model spec",
            f"unknown provider {provider_name!r} in spec {spec!r}; "
            f"known providers: {[p.name for p in cfg.providers]}",
        )

    @staticmethod
    def create(provider: ProviderConfig, model_name: str) -> tuple[BaseChatModel, str]:
        """Instantiate a LangChain chat model for *provider* + *model_name*.

        Returns (model, protocol). Protocol is needed by the loop's Ollama branch.
        """
        api_key = _resolve_api_key(provider)

        kwargs: dict[str, Any] = {"model": model_name}
        if provider.base_url is not None:
            kwargs["base_url"] = provider.base_url
        if api_key is not None:
            kwargs["api_key"] = api_key

        if provider.protocol == "openai":
            cls = _load_openai_class()
            model = cls(**kwargs)
        elif provider.protocol == "anthropic":
            cls = _load_anthropic_class()
            model = cls(**kwargs)
        elif provider.protocol == "ollama":
            cls = _load_ollama_class()
            model = cls(**kwargs)
        else:
            # unreachable — pydantic Literal validation in ProviderConfig
            raise UnknownModelSpecError(
                source="provider", detail=f"unknown protocol: {provider.protocol}"
            )

        return model, provider.protocol
