"""Tests for aura.cli.__main__ entry point."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path


def test_version_flag_fast_path() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "aura" in result.stdout.lower()


def test_help_flag() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "aura" in result.stdout.lower()


def test_plaintext_api_key_emits_warning(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "providers": [{
            "name": "test",
            "protocol": "openai",
            "api_key": "sk-plaintext-secret",
        }],
        "router": {"default": "test:gpt-4o-mini"},
    }))

    from rich.console import Console

    from aura.cli.__main__ import _warn_plaintext_api_keys
    from aura.config.loader import load_config

    cfg = load_config(user_config=config_path, project_config=tmp_path / "absent.json")
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)

    _warn_plaintext_api_keys(cfg, console)

    out = buf.getvalue()
    assert "Warning" in out
    assert "'test'" in out
    assert "plaintext" in out


def _ns(**kw: object) -> object:
    import argparse

    return argparse.Namespace(**kw)


def test_resolve_mode_defaults_to_default() -> None:
    from aura.cli.__main__ import _resolve_mode
    from aura.config.schema import AuraConfig

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    args = _ns(bypass_permissions=False)
    assert _resolve_mode(args, cfg) == "default"  # type: ignore[arg-type]


def test_resolve_mode_reads_config_permissions_mode() -> None:
    from aura.cli.__main__ import _resolve_mode
    from aura.config.schema import AuraConfig

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "permissions": {"mode": "bypass"},
    })
    args = _ns(bypass_permissions=False)
    assert _resolve_mode(args, cfg) == "bypass"  # type: ignore[arg-type]


def test_resolve_mode_cli_flag_wins_over_config_default() -> None:
    from aura.cli.__main__ import _resolve_mode
    from aura.config.schema import AuraConfig

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "permissions": {"mode": "default"},
    })
    args = _ns(bypass_permissions=True)
    assert _resolve_mode(args, cfg) == "bypass"  # type: ignore[arg-type]
