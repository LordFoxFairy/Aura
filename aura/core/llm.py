"""Model spec resolution + lazy SDK-backed chat model construction."""

from __future__ import annotations

import importlib
import os
from typing import Any

from langchain_core.language_models import BaseChatModel

from aura.config.schema import AuraConfig, AuraConfigError, ProviderConfig
from aura.core.persistence import journal


class UnknownModelSpecError(AuraConfigError):
    """Raised when a model spec can't be resolved to a known provider."""


class MissingProviderDependencyError(AuraConfigError):
    """Raised when the LangChain SDK for a protocol isn't installed."""


class MissingCredentialError(AuraConfigError):
    """Raised when no API key is found for a protocol that requires one."""


# 协议 → (PyPI/SDK 模块名, 导出类名, 默认 API key 环境变量)。
# 加新 provider：这一张表 + ProviderConfig.protocol Literal 两处。
_PROTOCOLS: dict[str, tuple[str, str, str | None]] = {
    "openai": ("langchain_openai", "ChatOpenAI", "OPENAI_API_KEY"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic", "ANTHROPIC_API_KEY"),
    "ollama": ("langchain_ollama", "ChatOllama", None),
}


# Static registry of model family → max context window (tokens). Used by the
# status bar to render `live/window` ratios. We match by substring with
# longest-prefix wins so dated suffixes ("claude-opus-4-20250514") and
# provider prefixes ("anthropic:claude-opus-4") both resolve to the family.
# Unknown models fall back to a 128k modern default rather than raising —
# a wrong ratio is a strictly better UX than a crashed status bar.
_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-4": 200_000,
    # OpenAI — listed longest-first-friendly (longest-prefix match wins at
    # lookup time regardless of insertion order, but the ordering helps
    # humans reading the table).
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
    "gpt-5": 200_000,
    "gpt-4": 8_192,
}
_DEFAULT_CONTEXT_WINDOW = 128_000


def get_context_window(model_spec: str) -> int:
    """Return the max context window in tokens for ``model_spec``.

    Matches by substring with longest-key wins, so ``provider:claude-opus-4-20250514``
    resolves to ``claude-opus-4`` (200k) and ``openai:gpt-4o-mini`` beats
    ``gpt-4`` (which would map to the 8k entry). Falls back to
    ``_DEFAULT_CONTEXT_WINDOW`` when nothing matches — unknown models still
    render a usable ratio on the status bar instead of a zero-division.
    """
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
    return getattr(module, class_name)  # type: ignore[no-any-return]


def _resolve_api_key(provider: ProviderConfig) -> str | None:
    """Return API key for *provider* or raise MissingCredentialError.

    Treats empty-string api_key as missing (fall through to api_key_env /
    default env) — a user with `"api_key": ""` in their config clearly didn't
    mean to hard-code an empty secret.
    """
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
        return None  # ollama doesn't need a key

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
    """Resolve *spec* to (ProviderConfig, model_name).

    Resolution rules:
    1. If *spec* is a router alias, substitute once.
    2. Split on the first ':'. Left = provider name, right = model name.
    3. Unknown alias with no colon → UnknownModelSpecError.
    4. Unknown provider name → UnknownModelSpecError.
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
    """Instantiate a LangChain chat model for *provider* + *model_name*."""
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
