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


def test_plaintext_api_key_emits_warning_only_in_verbose(tmp_path: Path) -> None:
    # Non-verbose runs SHOULD stay silent — the warning fires every startup
    # otherwise and operators tune it out (which defeats the point of the
    # warning in the first place). Verbose still prints for operators
    # actively debugging or auditing.
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

    # Default (no --verbose) — silent.
    buf_quiet = io.StringIO()
    _warn_plaintext_api_keys(
        cfg, Console(file=buf_quiet, force_terminal=False, width=200),
    )
    assert "Warning" not in buf_quiet.getvalue()
    assert "plaintext" not in buf_quiet.getvalue()

    # --verbose — prints.
    buf_verbose = io.StringIO()
    _warn_plaintext_api_keys(
        cfg, Console(file=buf_verbose, force_terminal=False, width=200),
        verbose=True,
    )
    out = buf_verbose.getvalue()
    assert "Warning" in out
    assert "'test'" in out
    assert "plaintext" in out


def test_plaintext_api_key_writes_journal_event(tmp_path: Path) -> None:
    # Troubleshooting contract: the console warning is ``--verbose`` only,
    # so the journal entry MUST fire unconditionally — that's what makes
    # the audit trail reliable. An operator grepping events.jsonl for
    # "why did this provider expose a key" must find a machine-readable
    # record regardless of whether anyone was watching the console.
    import pytest
    from rich.console import Console

    from aura.cli.__main__ import _warn_plaintext_api_keys
    from aura.config.loader import load_config
    from aura.core.persistence import journal

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "providers": [
            {"name": "alpha", "protocol": "openai", "api_key": "sk-1"},
            {"name": "beta", "protocol": "openai", "api_key_env": "BETA_KEY"},
            {"name": "gamma", "protocol": "openai", "api_key": "sk-2"},
        ],
        "router": {"default": "alpha:gpt-4o-mini"},
    }))
    cfg = load_config(user_config=config_path, project_config=tmp_path / "absent.json")

    events: list[tuple[str, dict[str, object]]] = []

    def _capture(event: str, /, **fields: object) -> None:
        events.append((event, fields))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(journal, "write", _capture)
        _warn_plaintext_api_keys(
            cfg, Console(file=io.StringIO(), force_terminal=False, width=200),
        )

    plaintext_events = [e for e in events if e[0] == "plaintext_api_key_warning"]
    assert len(plaintext_events) == 2  # alpha + gamma, beta uses api_key_env
    names = {e[1]["provider"] for e in plaintext_events}
    assert names == {"alpha", "gamma"}


def _ns(**kw: object) -> object:
    import argparse

    return argparse.Namespace(**kw)


def test_resolve_mode_defaults_to_default() -> None:
    # Post-2026-04-21: _resolve_mode takes a PermissionsConfig, not AuraConfig.
    # No flag + default PermissionsConfig (mode="default") → "default".
    from aura.cli.__main__ import _resolve_mode
    from aura.schemas.permissions import PermissionsConfig

    args = _ns(bypass_permissions=False)
    assert _resolve_mode(args, PermissionsConfig()) == "default"  # type: ignore[arg-type]


def test_resolve_mode_reads_permissions_config_mode() -> None:
    # PermissionsConfig comes from settings.json (via store.load), not from
    # AuraConfig. Mode set there should be honored when the flag is off.
    from aura.cli.__main__ import _resolve_mode
    from aura.schemas.permissions import PermissionsConfig

    perm_cfg = PermissionsConfig(mode="bypass")
    args = _ns(bypass_permissions=False)
    assert _resolve_mode(args, perm_cfg) == "bypass"  # type: ignore[arg-type]


def test_resolve_mode_cli_flag_wins_over_settings_default() -> None:
    from aura.cli.__main__ import _resolve_mode
    from aura.schemas.permissions import PermissionsConfig

    perm_cfg = PermissionsConfig(mode="default")
    args = _ns(bypass_permissions=True)
    assert _resolve_mode(args, perm_cfg) == "bypass"  # type: ignore[arg-type]


def test_resolve_mode_cli_flag_wins_even_over_settings_bypass() -> None:
    # Trivial but worth locking: flag True always wins regardless of the
    # settings value. (A user could explicitly set bypass in both places;
    # ordering must be predictable.)
    from aura.cli.__main__ import _resolve_mode
    from aura.schemas.permissions import PermissionsConfig

    perm_cfg = PermissionsConfig(mode="bypass")
    args = _ns(bypass_permissions=True)
    assert _resolve_mode(args, perm_cfg) == "bypass"  # type: ignore[arg-type]
