"""Tests for aura.config.loader — load_config() with precedence and merge."""
from __future__ import annotations

from pathlib import Path

import pytest

from aura.config.loader import load_config
from aura.config.schema import AuraConfig, AuraConfigError

FIXTURES = Path(__file__).parent / "fixtures" / "config"


# ---------------------------------------------------------------------------
# Autouse fixture: always clear AURA_CONFIG env var to prevent state leakage
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_aura_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURA_CONFIG", raising=False)


# ---------------------------------------------------------------------------
# 1. Defaults when no files exist
# ---------------------------------------------------------------------------


def test_load_config_defaults_when_no_files(tmp_path: Path) -> None:
    cfg = load_config(user_config=tmp_path / "u.toml", project_config=tmp_path / "p.toml")
    assert cfg == AuraConfig()


# ---------------------------------------------------------------------------
# 2. User-only config applies
# ---------------------------------------------------------------------------


def test_load_config_user_only_applies(tmp_path: Path) -> None:
    cfg = load_config(
        user_config=FIXTURES / "user_only.toml",
        project_config=tmp_path / "nonexistent.toml",
    )
    assert cfg.model.provider == "anthropic"
    assert cfg.model.name == "claude-3-5-sonnet-latest"


# ---------------------------------------------------------------------------
# 3. Project config overrides user section (section-level whole replacement)
# ---------------------------------------------------------------------------


def test_load_config_project_overrides_user_section() -> None:
    cfg = load_config(
        user_config=FIXTURES / "user_only.toml",
        project_config=FIXTURES / "project_override.toml",
    )
    assert cfg.model.provider == "ollama"
    assert cfg.model.name == "llama3"


# ---------------------------------------------------------------------------
# 4. AURA_CONFIG env var wins over project + user
# ---------------------------------------------------------------------------


def test_load_config_env_var_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_config = tmp_path / "env_config.toml"
    env_config.write_text('[model]\nprovider = "openai"\nname = "gpt-5"\n')
    monkeypatch.setenv("AURA_CONFIG", str(env_config))

    cfg = load_config(
        user_config=FIXTURES / "user_only.toml",
        project_config=FIXTURES / "project_override.toml",
    )
    assert cfg.model.provider == "openai"
    assert cfg.model.name == "gpt-5"


# ---------------------------------------------------------------------------
# 5. AURA_CONFIG pointing to missing file raises AuraConfigError
# ---------------------------------------------------------------------------


def test_load_config_env_var_missing_file_raises_AuraConfigError(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AURA_CONFIG", str(tmp_path / "does_not_exist.toml"))

    with pytest.raises(AuraConfigError) as exc_info:
        load_config(user_config=tmp_path / "u.toml", project_config=tmp_path / "p.toml")

    err = exc_info.value
    assert err.source == "$AURA_CONFIG"
    assert "not found" in err.detail


# ---------------------------------------------------------------------------
# 6. Invalid TOML raises AuraConfigError with correct source
# ---------------------------------------------------------------------------


def test_load_config_invalid_toml_raises_AuraConfigError(tmp_path: Path) -> None:
    invalid_path = FIXTURES / "invalid.toml"

    with pytest.raises(AuraConfigError) as exc_info:
        load_config(user_config=invalid_path, project_config=tmp_path / "p.toml")

    err = exc_info.value
    assert err.source == str(invalid_path)
    # detail should contain parse error info
    assert err.detail  # non-empty


# ---------------------------------------------------------------------------
# 7. Pydantic ValidationError wraps as AuraConfigError with source "merged config"
# ---------------------------------------------------------------------------


def test_load_config_validation_error_wraps_as_AuraConfigError(
    tmp_path: Path,
) -> None:
    bad_config = tmp_path / "bad_schema.toml"
    bad_config.write_text("[tools]\nunknown_key = 1\n")

    with pytest.raises(AuraConfigError) as exc_info:
        load_config(user_config=bad_config, project_config=tmp_path / "p.toml")

    err = exc_info.value
    assert err.source == "merged config"


# ---------------------------------------------------------------------------
# 8. Section-level replace: project [model] wholly replaces user's [model]
#    (user extras disappear — NOT key-level merge)
# ---------------------------------------------------------------------------


def test_load_config_section_level_replace_drops_user_extras(tmp_path: Path) -> None:
    user_cfg = tmp_path / "user.toml"
    user_cfg.write_text("[model]\ntemperature = 0.7\n")

    project_cfg = tmp_path / "project.toml"
    project_cfg.write_text('[model]\nprovider = "anthropic"\n')

    cfg = load_config(user_config=user_cfg, project_config=project_cfg)

    assert cfg.model.provider == "anthropic"
    # temperature should NOT be present — project's [model] wholly replaced user's
    assert cfg.model.model_extra is None or "temperature" not in (cfg.model.model_extra or {})
