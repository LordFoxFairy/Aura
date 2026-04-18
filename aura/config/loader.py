"""load_config() — read JSON configs with precedence and top-level shallow merge."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aura.config.schema import AuraConfig, AuraConfigError
from aura.core import journal


def _read_json(path: Path, source: str) -> dict[str, Any]:
    """Read a JSON file and return its dict. Wraps parse errors as AuraConfigError."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise AuraConfigError(source=source, detail=str(exc)) from exc
    if not isinstance(data, dict):
        type_name = type(data).__name__
        raise AuraConfigError(
            source=source, detail=f"expected object at top level, got {type_name}"
        )
    return data


def load_config(
    *,
    user_config: Path | None = None,
    project_config: Path | None = None,
) -> AuraConfig:
    """Load and merge configuration from multiple sources.

    Precedence (highest first):
      1. $AURA_CONFIG env var path
      2. project_config  (default: ./.aura/config.json)
      3. user_config     (default: ~/.aura/config.json)
      4. built-in defaults

    Merge is top-level shallow replace: a later source's top-level key wholly
    replaces an earlier source's key (no deep merge).
    """
    if user_config is None:
        user_config = Path.home() / ".aura" / "config.json"
    if project_config is None:
        project_config = Path.cwd() / ".aura" / "config.json"

    env_path_str = os.environ.get("AURA_CONFIG", "")

    journal.write(
        "config_load_begin",
        user_config=str(user_config),
        project_config=str(project_config),
        env_config=env_path_str or None,
    )

    user_dict: dict[str, Any] = {}
    if user_config.exists():
        user_dict = _read_json(user_config, source=str(user_config))

    project_dict: dict[str, Any] = {}
    if project_config.exists():
        project_dict = _read_json(project_config, source=str(project_config))

    env_dict: dict[str, Any] = {}
    if env_path_str:
        env_path = Path(env_path_str)
        if not env_path.exists():
            journal.write(
                "config_load_failed",
                reason="env_config_missing",
                path=str(env_path),
            )
            raise AuraConfigError(
                source="$AURA_CONFIG",
                detail=f"config file not found: {env_path}",
            )
        env_dict = _read_json(env_path, source=str(env_path))

    merged: dict[str, Any] = {}
    for src in (user_dict, project_dict, env_dict):
        merged.update(src)

    try:
        cfg = AuraConfig.model_validate(merged)
    except ValidationError as exc:
        journal.write("config_load_failed", reason="validation", detail=str(exc))
        raise AuraConfigError(source="merged config", detail=str(exc)) from exc

    journal.write(
        "config_load_end",
        providers=[p.name for p in cfg.providers],
        router_keys=list(cfg.router.keys()),
    )
    return cfg
