"""Tests for aura.config.schema — AuraConfig pydantic v2 schema."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aura.config.schema import AuraConfig, AuraConfigError, StorageConfig, ToolsConfig

# ---------------------------------------------------------------------------
# 1. Defaults
# ---------------------------------------------------------------------------


def test_defaults() -> None:
    cfg = AuraConfig()
    assert cfg.model.provider == "openai"
    assert cfg.model.name == "gpt-4o-mini"
    assert cfg.tools.enabled == ["read_file", "write_file", "bash"]
    assert cfg.ui.theme == "default"
    assert cfg.storage.path == "~/.aura/sessions.db"


# ---------------------------------------------------------------------------
# 2. Unknown top-level section raises ValidationError
# ---------------------------------------------------------------------------


def test_unknown_top_level_raises() -> None:
    with pytest.raises(ValidationError):
        AuraConfig.model_validate({"bogus": {}})


# ---------------------------------------------------------------------------
# 3. Extra kwargs on [model] land in model_extra
# ---------------------------------------------------------------------------


def test_model_extra_kwargs_via_nested_dict_validation() -> None:
    cfg = AuraConfig.model_validate({"model": {"temperature": 0.5, "max_tokens": 4096}})
    assert cfg.model.model_extra == {"temperature": 0.5, "max_tokens": 4096}
    assert cfg.model.provider == "openai"
    assert cfg.model.name == "gpt-4o-mini"


def test_full_nested_dict_happy_path() -> None:
    cfg = AuraConfig.model_validate({
        "model": {"provider": "anthropic", "name": "claude-3-5-sonnet-latest", "temperature": 0.2},
        "tools": {"enabled": ["read_file"]},
        "storage": {"path": "/tmp/aura.db"},
        "ui": {"theme": "default"},
    })
    assert cfg.model.provider == "anthropic"
    assert cfg.model.name == "claude-3-5-sonnet-latest"
    assert cfg.model.model_extra == {"temperature": 0.2}
    assert cfg.tools.enabled == ["read_file"]
    assert cfg.storage.path == "/tmp/aura.db"
    assert cfg.ui.theme == "default"


# ---------------------------------------------------------------------------
# 4. Unknown key on [tools] raises ValidationError (extra=forbid)
# ---------------------------------------------------------------------------


def test_tools_extra_key_raises() -> None:
    with pytest.raises(ValidationError):
        ToolsConfig.model_validate({"enabled": [], "unknown_key": 1})


# ---------------------------------------------------------------------------
# 5. resolved_storage_path() expands ~ using $HOME
# ---------------------------------------------------------------------------


def test_resolved_storage_path_expands_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = AuraConfig(storage=StorageConfig(path="~/aura.db"))
    assert cfg.resolved_storage_path() == tmp_path / "aura.db"


# ---------------------------------------------------------------------------
# 6. AuraConfigError shape
# ---------------------------------------------------------------------------


def test_aura_config_error_shape() -> None:
    err = AuraConfigError(source="user_config", detail="boom")
    assert err.source == "user_config"
    assert err.detail == "boom"
    assert str(err) == "user_config: boom"
