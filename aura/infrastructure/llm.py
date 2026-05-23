"""Model spec resolution + lazy SDK-backed chat model construction."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel

from aura.config.schema import AuraConfig, AuraConfigError, ProviderConfig
from aura.infrastructure.persistence import journal


class UnknownModelSpecError(AuraConfigError):
    """Raised when a model spec can't be resolved to a known provider."""


class MissingProviderDependencyError(AuraConfigError):
    """Raised when the LangChain SDK for a protocol isn't installed."""


class MissingCredentialError(AuraConfigError):
    """Raised when no API key is found for a protocol that requires one."""


# protocol → (SDK module, class name, default API key env var).
_PROTOCOLS: dict[str, tuple[str, str, str | None]] = {
    "openai": ("langchain_openai", "ChatOpenAI", "OPENAI_API_KEY"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic", "ANTHROPIC_API_KEY"),
    "ollama": ("langchain_ollama", "ChatOllama", None),
}


# Family → max context window. Substring match w/ longest-prefix wins.
# Over-stating defaults causes "status bar reads 8% while prompt overflows";
# 128k floor matches claude-code's safe default.
_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-4": 200_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
    "gpt-5": 400_000,
    "gpt-4": 8_192,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o1": 200_000,
    "o3-mini": 200_000,
    "o3": 200_000,
    "gemini-2": 1_000_000,
    "gemini-1.5": 1_000_000,
    "deepseek-chat": 128_000,
    "deepseek-coder": 128_000,
    "deepseek-reasoner": 64_000,
    "deepseek-v3": 128_000,
    "deepseek-v2": 128_000,
    "deepseek": 128_000,
    "glm-4-long": 1_000_000,
    "glm-4-plus": 128_000,
    "glm-4-air": 128_000,
    "glm-4-flash": 128_000,
    "glm-4.5": 128_000,
    "glm-4": 128_000,
    "glm-5": 128_000,
    "glm-z1": 128_000,
    "glm": 128_000,
    "qwen-turbo-1m": 1_000_000,
    "qwen-long": 10_000_000,
    "qwen-max": 32_000,
    "qwen3-coder": 128_000,
    "qwen3": 128_000,
    "qwen2.5-coder": 128_000,
    "qwen2.5": 128_000,
    "qwen-coder": 128_000,
    "qwen-plus": 128_000,
    "qwen-turbo": 128_000,
    "qwen": 128_000,
    "moonshot-v1-128k": 128_000,
    "moonshot-v1-32k": 32_000,
    "moonshot-v1-8k": 8_000,
    "kimi-k2": 128_000,
    "kimi": 128_000,
    "moonshot": 128_000,
}

_DEFAULT_CONTEXT_WINDOW = 128_000


def get_context_window(model_spec: str) -> int:
    """Max context window for ``model_spec`` via longest-prefix substring match."""
    _, _, name = model_spec.rpartition(":")
    name = (name or model_spec).lower()
    best: str | None = None
    for key in _CONTEXT_WINDOWS:
        if key in name and (best is None or len(key) > len(best)):
            best = key
    return _CONTEXT_WINDOWS[best] if best else _DEFAULT_CONTEXT_WINDOW


def _load_class(protocol: str) -> type[BaseChatModel]:
    module_name, class_name, _ = _PROTOCOLS[protocol]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise MissingProviderDependencyError(
            source="provider sdk",
            detail=f"{module_name} not installed. Run: pip install 'aura[{protocol}]'",
        ) from exc
    return getattr(module, class_name)  # type: ignore[no-any-return]  # fake returns Any from __dict__


def _resolve_api_key(provider: ProviderConfig) -> str | None:
    """Return the resolved key or raise; empty-string ``api_key`` is treated as missing."""
    if provider.api_key:
        return provider.api_key

    if provider.api_key_env is not None:
        key = os.environ.get(provider.api_key_env)
        if not key:
            journal.write(
                "credential_missing",
                provider=provider.name,
                env_var=provider.api_key_env,
            )
            raise MissingCredentialError(
                source="env",
                detail=f"${provider.api_key_env} not set for provider {provider.name!r}",
            )
        return key

    default_env = _PROTOCOLS[provider.protocol][2]
    if default_env is None:
        return None  # ollama

    key = os.environ.get(default_env)
    if not key:
        journal.write(
            "credential_missing",
            provider=provider.name,
            env_var=default_env,
        )
        raise MissingCredentialError(
            source="env",
            detail=(
                f"${default_env} not set for provider {provider.name!r}"
                f" (protocol {provider.protocol})"
            ),
        )
    return key


def resolve(spec: str, *, cfg: AuraConfig) -> tuple[ProviderConfig, str]:
    """Resolve *spec* to ``(ProviderConfig, model_name)``.

    Rules: router alias → substitute once; split on first ``:``; left =
    provider name, right = model name.
    """
    resolved = cfg.router.get(spec, spec)

    provider_name, sep, model_name = resolved.partition(":")

    if not sep:
        journal.write("model_resolve_failed", spec=spec, reason="no_colon")
        raise UnknownModelSpecError(
            "model spec",
            f"{spec!r} is not a router alias and not in 'provider:model' form",
        )

    for provider in cfg.providers:
        if provider.name == provider_name:
            journal.write(
                "model_resolved",
                spec=spec,
                provider=provider_name,
                model=model_name,
            )
            return (provider, model_name)

    journal.write(
        "model_resolve_failed",
        spec=spec,
        reason="unknown_provider",
        provider_name=provider_name,
    )
    raise UnknownModelSpecError(
        "model spec",
        f"unknown provider {provider_name!r} in spec {spec!r}; "
        f"known providers: {[p.name for p in cfg.providers]}",
    )


def create(provider: ProviderConfig, model_name: str) -> BaseChatModel:
    api_key = _resolve_api_key(provider)

    kwargs: dict[str, Any] = {**provider.params, "model": model_name}
    if provider.base_url is not None:
        kwargs["base_url"] = provider.base_url
    if api_key is not None:
        kwargs["api_key"] = api_key

    journal.write(
        "model_create_attempt",
        provider=provider.name,
        protocol=provider.protocol,
        model=model_name,
        has_base_url=provider.base_url is not None,
    )

    cls = _load_class(provider.protocol)
    model = cls(**kwargs)

    journal.write(
        "model_created",
        provider=provider.name,
        protocol=provider.protocol,
    )
    return model


def make_model_for_spec(spec: str, cfg: AuraConfig) -> BaseChatModel:
    """One-shot resolve+create. Callers needing pre-flight should use ``resolve``."""
    provider, model_name = resolve(spec, cfg=cfg)
    return create(provider, model_name)


def make_summary_model_factory(
    cfg: AuraConfig,
    main_model: BaseChatModel,
    *,
    summary_spec: str | None = None,
) -> Callable[[], BaseChatModel]:
    """Memoized factory yielding the model for cheap summary turns.

    ``summary_spec=None`` → returns ``main_model`` verbatim. A failed
    spec surfaces on the FIRST invocation (lazy build).
    """
    cached: list[BaseChatModel] = []

    def _factory() -> BaseChatModel:
        if cached:
            return cached[0]
        if summary_spec is None:
            cached.append(main_model)
            return main_model
        provider, model_name = resolve(summary_spec, cfg=cfg)
        m = create(provider, model_name)
        cached.append(m)
        return m

    return _factory
