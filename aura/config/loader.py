"""load_config() — read TOML configs with precedence and section-level merge."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aura.config.schema import AuraConfig, AuraConfigError


def _read_toml(path: Path, source: str) -> dict[str, Any]:
    """Read a TOML file and return its dict. Wraps parse errors as AuraConfigError."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise AuraConfigError(source=source, detail=str(exc)) from exc


def load_config(
    *,
    user_config: Path | None = None,
    project_config: Path | None = None,
) -> AuraConfig:
    """Load and merge configuration from multiple sources.

    Precedence (highest first):
      1. $AURA_CONFIG env var path
      2. project_config  (default: ./.aura/config.toml)
      3. user_config     (default: ~/.aura/config.toml)
      4. built-in defaults

    Merge is section-level shallow replace: a later source's [model] section
    wholly replaces an earlier source's [model] section.
    """
    # Resolve default discovery paths when not explicitly provided
    if user_config is None:
        user_config = Path.home() / ".aura" / "config.toml"
    if project_config is None:
        project_config = Path.cwd() / ".aura" / "config.toml"

    # Read each optional file (missing optional files are not errors)
    user_dict: dict[str, Any] = {}
    if user_config.exists():
        user_dict = _read_toml(user_config, source=str(user_config))

    project_dict: dict[str, Any] = {}
    if project_config.exists():
        project_dict = _read_toml(project_config, source=str(project_config))

    # $AURA_CONFIG env var: must exist if set
    env_dict: dict[str, Any] = {}
    env_path_str = os.environ.get("AURA_CONFIG", "")
    if env_path_str:
        env_path = Path(env_path_str)
        if not env_path.exists():
            raise AuraConfigError(
                source="$AURA_CONFIG",
                detail=f"config file not found: {env_path}",
            )
        env_dict = _read_toml(env_path, source=str(env_path))

    # Merge: user → project → env (section-level shallow replace)
    merged: dict[str, Any] = {}
    for source_dict in (user_dict, project_dict, env_dict):
        if source_dict:
            merged.update(source_dict)

    # Validate with pydantic; wrap any ValidationError
    try:
        return AuraConfig.model_validate(merged)
    except ValidationError as exc:
        raise AuraConfigError(source="merged config", detail=str(exc)) from exc
