"""Tests for aura.infrastructure.llm.create — lazy SDK loading + secret resolution."""

from __future__ import annotations

import sys
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from aura.config.schema import AuraConfig, AuraConfigError, ProviderConfig
from aura.infrastructure.llm import (
    _DEFAULT_CONTEXT_WINDOW,
    MissingCredentialError,
    MissingProviderDependencyError,
    UnknownModelSpecError,
    create,
    get_context_window,
    make_model_for_spec,
    make_summary_model_factory,
    resolve,
)


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
    val: dict[str, Any] = model.kwargs
    return val


def test_create_openai_happy_path_uses_default_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.infrastructure import llm

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    provider = ProviderConfig(name="openai", protocol="openai")
    model = create(provider, "gpt-4o-mini")

    assert isinstance(model, _StubOpenAI)
    kw = _stub_kwargs(model)
    assert kw["model"] == "gpt-4o-mini"
    assert kw["api_key"] == "sk-test"
    assert "base_url" not in kw


def test_create_openai_with_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.infrastructure import llm

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    provider = ProviderConfig(
        name="openrouter",
        protocol="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    )
    model = create(provider, "mistral-7b")

    kw = _stub_kwargs(model)
    assert kw["base_url"] == "https://openrouter.ai/api/v1"
    assert kw["api_key"] == "or-key"
    assert kw["model"] == "mistral-7b"


def test_create_openai_plaintext_api_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.infrastructure import llm

    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = ProviderConfig(name="openai", protocol="openai", api_key="plaintext-key")
    model = create(provider, "gpt-4o-mini")

    assert _stub_kwargs(model)["api_key"] == "plaintext-key"


def test_create_openai_api_key_env_preferred_over_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.infrastructure import llm

    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOpenAI)
    monkeypatch.setenv("CUSTOM_KEY", "custom")
    monkeypatch.setenv("OPENAI_API_KEY", "default")

    provider = ProviderConfig(name="openai", protocol="openai", api_key_env="CUSTOM_KEY")
    model = create(provider, "gpt-4o-mini")

    assert _stub_kwargs(model)["api_key"] == "custom"


def test_create_anthropic_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.infrastructure import llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

    provider = ProviderConfig(name="anthropic", protocol="anthropic")
    model = create(provider, "claude-3-5-sonnet-20241022")

    assert isinstance(model, _StubAnthropic)
    kw = _stub_kwargs(model)
    assert kw["model"] == "claude-3-5-sonnet-20241022"
    assert kw["api_key"] == "ant-key"


def test_create_ollama_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.infrastructure import llm

    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOllama)

    provider = ProviderConfig(name="ollama", protocol="ollama")
    model = create(provider, "llama3")

    assert isinstance(model, _StubOllama)
    kw = _stub_kwargs(model)
    assert kw["model"] == "llama3"
    assert "api_key" not in kw


def test_create_ollama_with_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.infrastructure import llm

    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOllama)

    provider = ProviderConfig(
        name="ollama-remote",
        protocol="ollama",
        base_url="http://remote:11434",
    )
    model = create(provider, "llama3")

    kw = _stub_kwargs(model)
    assert kw["base_url"] == "http://remote:11434"
    assert "api_key" not in kw
    assert kw["model"] == "llama3"


def test_create_missing_default_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.infrastructure import llm

    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = ProviderConfig(name="openai", protocol="openai")
    with pytest.raises(MissingCredentialError) as exc_info:
        create(provider, "gpt-4o-mini")

    assert "OPENAI_API_KEY" in exc_info.value.detail


def test_create_missing_api_key_env_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.infrastructure import llm

    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOpenAI)
    monkeypatch.delenv("CUSTOM_KEY", raising=False)

    provider = ProviderConfig(name="openai", protocol="openai", api_key_env="CUSTOM_KEY")
    with pytest.raises(MissingCredentialError) as exc_info:
        create(provider, "gpt-4o-mini")

    assert "CUSTOM_KEY" in exc_info.value.detail


def test_create_missing_sdk_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    from aura.infrastructure import llm

    def _boom(_protocol: str) -> type:
        raise MissingProviderDependencyError(
            source="provider sdk",
            detail="langchain_openai not installed. Run: pip install 'aura[openai]'",
        )

    monkeypatch.setattr(llm, "_load_class", _boom)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    provider = ProviderConfig(name="openai", protocol="openai")
    with pytest.raises(MissingProviderDependencyError) as exc_info:
        create(provider, "gpt-4o-mini")

    assert "pip install" in exc_info.value.detail
    assert "aura[openai]" in exc_info.value.detail


def test_create_errors_are_aura_config_error_subclasses() -> None:
    assert issubclass(MissingProviderDependencyError, AuraConfigError)
    assert issubclass(MissingCredentialError, AuraConfigError)


def test_create_empty_plaintext_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty-string api_key must NOT be forwarded — treat as missing."""
    from aura.infrastructure import llm

    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CUSTOM_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_KEY", "from-env")

    provider = ProviderConfig(name="x", protocol="openai", api_key="", api_key_env="CUSTOM_KEY")
    model = create(provider, "gpt-4o-mini")

    assert isinstance(model, _StubOpenAI)
    assert model.kwargs["api_key"] == "from-env"  # not ""


def test_create_empty_plaintext_api_key_with_no_fallback_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.infrastructure import llm

    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = ProviderConfig(name="x", protocol="openai", api_key="")
    with pytest.raises(MissingCredentialError) as exc_info:
        create(provider, "gpt-4o-mini")
    assert "OPENAI_API_KEY" in exc_info.value.detail


def test_create_forwards_provider_params_to_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.infrastructure import llm

    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    provider = ProviderConfig(
        name="openai",
        protocol="openai",
        params={"temperature": 0.3, "max_tokens": 4096, "timeout": 30},
    )
    model = create(provider, "gpt-4o-mini")

    kw = _stub_kwargs(model)
    assert kw["temperature"] == 0.3
    assert kw["max_tokens"] == 4096
    assert kw["timeout"] == 30
    assert kw["model"] == "gpt-4o-mini"
    assert kw["api_key"] == "k"


def test_create_resolved_fields_win_over_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a user puts 'model' in params, the resolved model_name still wins."""
    from aura.infrastructure import llm

    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    provider = ProviderConfig(
        name="openai",
        protocol="openai",
        params={"model": "wrong-name", "base_url": "bad"},
        base_url="https://good.example",
    )
    model = create(provider, "gpt-4o-mini")

    kw = _stub_kwargs(model)
    assert kw["model"] == "gpt-4o-mini"
    assert kw["base_url"] == "https://good.example"


def test_protocols_dict_matches_provider_literal() -> None:
    from typing import get_args

    from aura.config.schema import ProviderConfig
    from aura.infrastructure.llm import _PROTOCOLS

    # Pydantic's Literal annotation is accessible via the model's field info.
    protocol_field = ProviderConfig.model_fields["protocol"]
    literal_args = set(get_args(protocol_field.annotation))

    assert set(_PROTOCOLS.keys()) == literal_args, (
        f"_PROTOCOLS keys {set(_PROTOCOLS.keys())} drifted from "
        f"ProviderConfig.protocol Literal {literal_args}. Adding a provider "
        "requires updating BOTH aura/core/llm.py::_PROTOCOLS AND "
        "aura/config/schema.py::ProviderConfig.protocol."
    )


# --- _load_class: real SDK branches + missing-dependency guard -------------


@pytest.mark.parametrize(
    ("protocol", "expected_class_name"),
    [
        ("openai", "ChatOpenAI"),
        ("anthropic", "ChatAnthropic"),
        ("ollama", "ChatOllama"),
    ],
)
def test_load_class_returns_real_sdk_class(protocol: str, expected_class_name: str) -> None:
    """Each protocol must map to its concrete BaseChatModel subclass, never a stub."""
    from aura.infrastructure.llm import _load_class

    cls = _load_class(protocol)
    assert cls.__name__ == expected_class_name
    assert issubclass(cls, BaseChatModel)


@pytest.mark.parametrize(
    ("protocol", "module_name", "extra"),
    [
        ("openai", "langchain_openai", "openai"),
        ("anthropic", "langchain_anthropic", "anthropic"),
        ("ollama", "langchain_ollama", "ollama"),
    ],
)
def test_load_class_missing_sdk_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch, protocol: str, module_name: str, extra: str
) -> None:
    """A torn-out provider SDK surfaces a pip-install hint, not a raw ImportError."""
    from aura.infrastructure.llm import _load_class

    # Injecting None forces `from <module> import X` to raise ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, module_name, None)

    with pytest.raises(MissingProviderDependencyError) as exc_info:
        _load_class(protocol)

    assert f"{module_name} not installed" in exc_info.value.detail
    assert f"aura[{extra}]" in exc_info.value.detail


# --- get_context_window: longest-prefix substring match --------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("claude-3-5-sonnet-20241022", 200_000),
        ("openai:gpt-4o-mini", 128_000),
        ("gpt-4o", 128_000),
        ("openrouter:deepseek-reasoner", 64_000),
        ("qwen-long", 10_000_000),
    ],
)
def test_get_context_window_known_models(spec: str, expected: int) -> None:
    """Status-bar pressure ratio depends on the right window for each known model."""
    assert get_context_window(spec) == expected


def test_get_context_window_longest_substring_wins() -> None:
    """'gpt-4o' (128k) must beat the shorter substring 'gpt-4' (8k) on overlap."""
    assert get_context_window("gpt-4o") == 128_000
    assert get_context_window("gpt-4") == 8_192


def test_get_context_window_is_case_insensitive() -> None:
    """A model spec is lowercased before lookup so casing never drifts the window."""
    assert get_context_window("GPT-5") == get_context_window("gpt-5") == 400_000


def test_get_context_window_strips_provider_prefix() -> None:
    """Only the portion after the last ':' names the model; the prefix is ignored."""
    assert get_context_window("any-provider:gpt-4") == 8_192


@pytest.mark.parametrize("spec", ["", "totally-unknown-model", "x"])
def test_get_context_window_unknown_falls_back_to_default(spec: str) -> None:
    """An unrecognised or empty spec floors at the 128k default, never KeyErrors."""
    assert get_context_window(spec) == _DEFAULT_CONTEXT_WINDOW
    assert _DEFAULT_CONTEXT_WINDOW == 128_000


# --- resolve: router alias once, then provider:model -----------------------


def _two_provider_cfg() -> AuraConfig:
    return AuraConfig(
        providers=[
            ProviderConfig(name="openai", protocol="openai", api_key_env="OPENAI_API_KEY"),
            ProviderConfig(name="anthropic", protocol="anthropic"),
        ],
        router={"default": "openai:gpt-4o-mini", "fast": "anthropic:claude-3-5-haiku"},
    )


def test_resolve_router_alias() -> None:
    """A router alias expands to the configured provider:model pair."""
    cfg = _two_provider_cfg()
    provider, model = resolve("fast", cfg=cfg)
    assert provider.name == "anthropic"
    assert model == "claude-3-5-haiku"


def test_resolve_direct_provider_model() -> None:
    """A bare 'provider:model' spec resolves without any router entry."""
    cfg = _two_provider_cfg()
    provider, model = resolve("anthropic:claude-3-opus", cfg=cfg)
    assert provider.name == "anthropic"
    assert model == "claude-3-opus"


def test_resolve_model_name_keeps_inner_colons() -> None:
    """Only the first ':' splits provider from model; later colons stay in the name."""
    cfg = _two_provider_cfg()
    provider, model = resolve("openai:org:gpt-4o", cfg=cfg)
    assert provider.name == "openai"
    assert model == "org:gpt-4o"


def test_resolve_alias_is_applied_only_once() -> None:
    """Router lookup is a single hop; an alias whose target is another alias never chains."""
    cfg = AuraConfig(
        providers=[
            ProviderConfig(name="openai", protocol="openai", api_key_env="OPENAI_API_KEY"),
        ],
        router={"default": "openai:gpt-4o-mini", "hop": "openai:gpt-4o"},
    )
    # 'default' -> 'openai:gpt-4o-mini'; the value is not re-resolved as an alias.
    provider, model = resolve("default", cfg=cfg)
    assert provider.name == "openai"
    assert model == "gpt-4o-mini"


def test_resolve_no_colon_raises_unknown_spec() -> None:
    """A bareword that is neither alias nor provider:model is rejected loudly."""
    cfg = _two_provider_cfg()
    with pytest.raises(UnknownModelSpecError) as exc_info:
        resolve("bareword", cfg=cfg)
    assert "bareword" in exc_info.value.detail


def test_resolve_unknown_provider_lists_known() -> None:
    """An unknown provider names the offender and enumerates valid providers."""
    cfg = _two_provider_cfg()
    with pytest.raises(UnknownModelSpecError) as exc_info:
        resolve("ghost:model", cfg=cfg)
    assert "ghost" in exc_info.value.detail
    assert "openai" in exc_info.value.detail
    assert "anthropic" in exc_info.value.detail


def test_resolve_unknown_spec_is_aura_config_error() -> None:
    """UnknownModelSpecError stays in the AuraConfigError hierarchy for uniform handling."""
    assert issubclass(UnknownModelSpecError, AuraConfigError)


# --- make_model_for_spec: one-shot resolve+create --------------------------


def test_make_model_for_spec_resolves_then_creates(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one-shot helper routes the alias and hands the model name to the SDK class."""
    from aura.infrastructure import llm

    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    cfg = _two_provider_cfg()
    model = make_model_for_spec("default", cfg)

    assert isinstance(model, _StubOpenAI)
    assert _stub_kwargs(model)["model"] == "gpt-4o-mini"


def test_make_model_for_spec_propagates_resolve_error() -> None:
    """A bad spec fails at resolve time before any SDK construction is attempted."""
    cfg = _two_provider_cfg()
    with pytest.raises(UnknownModelSpecError):
        make_model_for_spec("ghost:model", cfg)


# --- make_summary_model_factory: lazy + memoized ---------------------------


def _fake_main_model() -> BaseChatModel:
    return GenericFakeChatModel(messages=iter(["main"]))


def test_summary_factory_none_spec_reuses_main_model() -> None:
    """No summary_spec means the summary model IS the main model (no second SDK call)."""
    cfg = _two_provider_cfg()
    main = _fake_main_model()
    factory = make_summary_model_factory(cfg, main, summary_spec=None)
    assert factory() is main


def test_summary_factory_none_spec_is_idempotent() -> None:
    """Repeated factory calls return the identical cached object, never a fresh one."""
    cfg = _two_provider_cfg()
    main = _fake_main_model()
    factory = make_summary_model_factory(cfg, main, summary_spec=None)
    assert factory() is factory() is main


def test_summary_factory_explicit_spec_creates_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit summary_spec builds a distinct model once, then serves it from cache."""
    from aura.infrastructure import llm

    monkeypatch.setattr(llm, "_load_class", lambda _p: _StubOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    cfg = _two_provider_cfg()
    main = _fake_main_model()
    factory = make_summary_model_factory(cfg, main, summary_spec="default")

    first = factory()
    second = factory()
    assert isinstance(first, _StubOpenAI)
    assert first is second  # memoized
    assert first is not main
    assert _stub_kwargs(first)["model"] == "gpt-4o-mini"


def test_summary_factory_bad_spec_surfaces_on_first_call_only() -> None:
    """A bad summary_spec must not raise at construction; it surfaces lazily on first call."""
    cfg = _two_provider_cfg()
    main = _fake_main_model()
    factory = make_summary_model_factory(cfg, main, summary_spec="ghost:model")
    with pytest.raises(UnknownModelSpecError):
        factory()
