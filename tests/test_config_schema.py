"""Tests for aura.config.schema — AuraConfig pydantic v2 schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aura.config.schema import AuraConfig, AuraConfigError

# ---------------------------------------------------------------------------
# 1. Defaults
# ---------------------------------------------------------------------------


def test_defaults() -> None:
    cfg = AuraConfig()
    assert len(cfg.providers) == 1
    p = cfg.providers[0]
    assert p.name == "openai"
    assert p.protocol == "openai"
    assert p.api_key_env == "OPENAI_API_KEY"
    assert p.base_url is None
    assert cfg.router == {"default": "openai:gpt-4o-mini"}
    assert cfg.tools.enabled == [
        "bash", "edit_file", "glob", "grep", "read_file",
        "todo_write", "web_fetch", "write_file",
    ]
    assert cfg.ui.theme == "default"
    assert cfg.storage.path == "~/.aura/sessions.db"


# ---------------------------------------------------------------------------
# 2. Unknown top-level key raises ValidationError
# ---------------------------------------------------------------------------


def test_unknown_top_level_raises() -> None:
    with pytest.raises(ValidationError):
        AuraConfig.model_validate({"bogus": {}})


# ---------------------------------------------------------------------------
# 3. Unknown key on a provider raises ValidationError
# ---------------------------------------------------------------------------


def test_unknown_key_on_provider_raises() -> None:
    with pytest.raises(ValidationError):
        AuraConfig.model_validate(
            {
                "providers": [{"name": "x", "protocol": "openai", "bogus": 1}],
                "router": {"default": "x:m"},
            }
        )


# ---------------------------------------------------------------------------
# 4. Unknown key on tools raises ValidationError
# ---------------------------------------------------------------------------


def test_unknown_key_on_tools_raises() -> None:
    with pytest.raises(ValidationError):
        AuraConfig.model_validate(
            {
                "tools": {"enabled": [], "bogus": 1},
                "providers": [{"name": "x", "protocol": "openai"}],
                "router": {"default": "x:m"},
            }
        )


# ---------------------------------------------------------------------------
# 5. Duplicate provider names raise ValueError
# ---------------------------------------------------------------------------


def test_duplicate_provider_names_raises() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        AuraConfig.model_validate(
            {
                "providers": [
                    {"name": "openai", "protocol": "openai"},
                    {"name": "openai", "protocol": "anthropic"},
                ],
                "router": {"default": "openai:gpt-4o-mini"},
            }
        )


# ---------------------------------------------------------------------------
# 6. router missing 'default' raises
# ---------------------------------------------------------------------------


def test_router_missing_default_raises() -> None:
    with pytest.raises(ValidationError, match="default"):
        AuraConfig.model_validate(
            {
                "router": {},
                "providers": [{"name": "x", "protocol": "openai"}],
            }
        )


# ---------------------------------------------------------------------------
# 7. router references unknown provider raises
# ---------------------------------------------------------------------------


def test_router_unknown_provider_raises() -> None:
    with pytest.raises(ValidationError, match="ghost"):
        AuraConfig.model_validate(
            {
                "router": {"default": "ghost:m"},
                "providers": [{"name": "real", "protocol": "openai"}],
            }
        )


# ---------------------------------------------------------------------------
# 8. router value missing colon raises
# ---------------------------------------------------------------------------


def test_router_value_missing_colon_raises() -> None:
    with pytest.raises(ValidationError):
        AuraConfig.model_validate(
            {
                "router": {"default": "no-colon"},
                "providers": [{"name": "x", "protocol": "openai"}],
            }
        )


# ---------------------------------------------------------------------------
# 9. Invalid protocol raises
# ---------------------------------------------------------------------------


def test_invalid_protocol_raises() -> None:
    with pytest.raises(ValidationError):
        AuraConfig.model_validate(
            {
                "providers": [{"name": "x", "protocol": "bogus"}],
                "router": {"default": "x:m"},
            }
        )


# ---------------------------------------------------------------------------
# 10. resolved_storage_path() expands ~ using $HOME
# ---------------------------------------------------------------------------


def test_resolved_storage_path_expands_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = AuraConfig.model_validate({"storage": {"path": "~/aura.db"}})
    assert cfg.resolved_storage_path() == tmp_path / "aura.db"


# ---------------------------------------------------------------------------
# 11. Full nested dict happy path with 2 providers
# ---------------------------------------------------------------------------


def test_full_nested_dict_happy_path() -> None:
    cfg = AuraConfig.model_validate(
        {
            "providers": [
                {"name": "a", "protocol": "openai", "api_key_env": "A"},
                {"name": "b", "protocol": "anthropic", "api_key_env": "B"},
            ],
            "router": {"default": "a:gpt-4o-mini", "claude": "b:claude-3-5-sonnet-latest"},
            "tools": {"enabled": ["read_file"]},
            "storage": {"path": "/tmp/x.db"},
            "ui": {"theme": "default"},
        }
    )
    assert len(cfg.providers) == 2
    assert cfg.providers[0].name == "a"
    assert cfg.providers[1].name == "b"
    assert cfg.router == {"default": "a:gpt-4o-mini", "claude": "b:claude-3-5-sonnet-latest"}
    assert cfg.tools.enabled == ["read_file"]
    assert cfg.storage.path == "/tmp/x.db"
    assert cfg.ui.theme == "default"


# ---------------------------------------------------------------------------
# 12. Model name with colon preserved in router (split on first colon only)
# ---------------------------------------------------------------------------


def test_model_name_with_colon_preserved_in_router() -> None:
    cfg = AuraConfig.model_validate(
        {
            "providers": [{"name": "p", "protocol": "anthropic"}],
            "router": {"default": "p:claude-3.5-haiku:20241022"},
        }
    )
    assert cfg.router["default"] == "p:claude-3.5-haiku:20241022"


# ---------------------------------------------------------------------------
# 13. AuraConfigError shape
# ---------------------------------------------------------------------------


def test_aura_config_error_shape() -> None:
    err = AuraConfigError(source="user_config", detail="boom")
    assert err.source == "user_config"
    assert err.detail == "boom"
    assert str(err) == "user_config: boom"


# ---------------------------------------------------------------------------
# 14. ProviderConfig.params field
# ---------------------------------------------------------------------------


def test_provider_config_params_default_empty() -> None:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "x", "protocol": "openai"}],
        "router": {"default": "x:gpt-4o-mini"},
    })
    assert cfg.providers[0].params == {}


def test_provider_config_params_accepts_langchain_kwargs() -> None:
    cfg = AuraConfig.model_validate({
        "providers": [{
            "name": "x",
            "protocol": "openai",
            "params": {"temperature": 0.7, "max_tokens": 4096, "timeout": 60},
        }],
        "router": {"default": "x:gpt-4o-mini"},
    })
    assert cfg.providers[0].params == {
        "temperature": 0.7, "max_tokens": 4096, "timeout": 60,
    }


def test_provider_config_unknown_top_level_still_raises() -> None:
    with pytest.raises(ValidationError):
        AuraConfig.model_validate({
            "providers": [{
                "name": "x",
                "protocol": "openai",
                "temperature": 0.7,
            }],
            "router": {"default": "x:gpt-4o-mini"},
        })


# ---------------------------------------------------------------------------
# 15. LogConfig
# ---------------------------------------------------------------------------


def test_log_config_defaults() -> None:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "x", "protocol": "openai"}],
        "router": {"default": "x:m"},
    })
    assert cfg.log.enabled is False
    assert cfg.log.path == "~/.aura/logs/events.jsonl"


def test_log_config_custom_enabled() -> None:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "x", "protocol": "openai"}],
        "router": {"default": "x:m"},
        "log": {"enabled": True, "path": "/tmp/aura.jsonl"},
    })
    assert cfg.log.enabled is True
    assert cfg.log.path == "/tmp/aura.jsonl"


def test_log_config_unknown_key_raises() -> None:
    with pytest.raises(ValidationError):
        AuraConfig.model_validate({
            "providers": [{"name": "x", "protocol": "openai"}],
            "router": {"default": "x:m"},
            "log": {"enabled": True, "bogus": 1},
        })
