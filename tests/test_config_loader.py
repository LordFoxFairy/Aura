"""Tests for aura.config.loader — load_config() with precedence and merge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aura.config.loader import load_config
from aura.config.schema import AuraConfig, AuraConfigError

FIXTURES = Path(__file__).parent / "fixtures" / "config"


def test_load_config_defaults_when_no_files(tmp_path: Path) -> None:
    cfg = load_config(user_config=tmp_path / "u.json", project_config=tmp_path / "p.json")
    assert cfg == AuraConfig()


def test_load_config_user_only_applies(tmp_path: Path) -> None:
    cfg = load_config(
        user_config=FIXTURES / "user_only.json",
        project_config=tmp_path / "nonexistent.json",
    )
    assert len(cfg.providers) == 1
    assert cfg.providers[0].name == "anthropic-direct"
    assert cfg.providers[0].protocol == "anthropic"
    assert cfg.router == {"default": "anthropic-direct:claude-3-5-sonnet-latest"}


def test_load_config_project_wholly_replaces_user_providers(tmp_path: Path) -> None:
    cfg = load_config(
        user_config=FIXTURES / "user_only.json",
        project_config=FIXTURES / "project_override.json",
    )
    assert len(cfg.providers) == 1
    assert cfg.providers[0].name == "openrouter"
    assert cfg.providers[0].protocol == "openai"
    assert cfg.router == {"default": "openrouter:anthropic/claude-opus-4"}


def test_load_config_env_var_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_config = tmp_path / "env_config.json"
    env_config.write_text(
        json.dumps(
            {
                "providers": [
                    {"name": "env-provider", "protocol": "openai", "api_key_env": "ENV_KEY"}
                ],
                "router": {"default": "env-provider:gpt-5"},
            }
        )
    )
    monkeypatch.setenv("AURA_CONFIG", str(env_config))

    cfg = load_config(
        user_config=FIXTURES / "user_only.json",
        project_config=FIXTURES / "project_override.json",
    )
    assert cfg.providers[0].name == "env-provider"
    assert cfg.router == {"default": "env-provider:gpt-5"}


def test_load_config_env_var_missing_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AURA_CONFIG", str(tmp_path / "does_not_exist.json"))

    with pytest.raises(AuraConfigError) as exc_info:
        load_config(user_config=tmp_path / "u.json", project_config=tmp_path / "p.json")

    err = exc_info.value
    assert err.source == "$AURA_CONFIG"
    assert "not found" in err.detail


def test_load_config_invalid_json_raises(tmp_path: Path) -> None:
    invalid_path = FIXTURES / "invalid.json"

    with pytest.raises(AuraConfigError) as exc_info:
        load_config(user_config=invalid_path, project_config=tmp_path / "p.json")

    err = exc_info.value
    assert err.source == str(invalid_path)
    assert err.detail


def test_load_config_non_object_top_level_raises(tmp_path: Path) -> None:
    bad_path = tmp_path / "array.json"
    bad_path.write_text("[1, 2, 3]")

    with pytest.raises(AuraConfigError) as exc_info:
        load_config(user_config=bad_path, project_config=tmp_path / "p.json")

    err = exc_info.value
    assert "object at top level" in err.detail


def test_load_config_validation_error_wraps(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad_schema.json"
    bad_config.write_text(
        json.dumps(
            {
                "router": {"default": "ghost:m"},
                "providers": [{"name": "real", "protocol": "openai"}],
            }
        )
    )

    with pytest.raises(AuraConfigError) as exc_info:
        load_config(user_config=bad_config, project_config=tmp_path / "p.json")

    err = exc_info.value
    assert err.source == "merged config"
    assert err.detail


def test_load_config_top_level_shallow_replace_drops_user_router(tmp_path: Path) -> None:
    user_cfg = tmp_path / "user.json"
    user_cfg.write_text(
        json.dumps(
            {
                "providers": [
                    {"name": "openai", "protocol": "openai", "api_key_env": "OPENAI_API_KEY"}
                ],
                "router": {"default": "openai:gpt-4o-mini", "fast": "openai:gpt-4o-mini"},
            }
        )
    )

    project_cfg = tmp_path / "project.json"
    project_cfg.write_text(
        json.dumps(
            {
                "providers": [
                    {"name": "openai", "protocol": "openai", "api_key_env": "OPENAI_API_KEY"}
                ],
                "router": {"default": "openai:gpt-4o-mini"},
            }
        )
    )

    cfg = load_config(user_config=user_cfg, project_config=project_cfg)

    assert cfg.router == {"default": "openai:gpt-4o-mini"}
    assert "fast" not in cfg.router
