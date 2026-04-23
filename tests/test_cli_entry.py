"""Tests for aura.cli.__main__ entry point."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest


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


# ---------------------------------------------------------------------------
# disable_bypass — audit Finding B: org-level kill switch for
# --bypass-permissions. Two entry points need enforcement:
#   1. CLI flag path (this file's subprocess tests).
#   2. Programmatic Agent/set_mode path (tests/test_agent.py).
# ---------------------------------------------------------------------------
def test_bypass_refused_message_is_stable() -> None:
    # The error message is the single piece of text the operator sees
    # when their --bypass-permissions attempt gets refused. Lock its
    # shape so docs / support runbooks can reference it.
    from aura.cli.__main__ import _bypass_refused_message

    msg = _bypass_refused_message()
    assert "--bypass-permissions is disabled" in msg
    assert "disable_bypass=true" in msg


def test_bypass_refused_end_to_end_via_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # Finding B acceptance: --bypass-permissions + disable_bypass=true
    # must exit 2 (config error) and print the refused message to
    # stderr. We exercise main() in-process (subprocess-based tests
    # misroute to a stale parent venv in editable-install setups)
    # with HOME + cwd scoped to tmp_path and the LLM factory
    # monkeypatched out so startup never touches real providers.
    from aura.cli.__main__ import main

    # Minimal user config so load_config succeeds.
    user_aura_dir = tmp_path / ".aura"
    user_aura_dir.mkdir()
    (user_aura_dir / "config.json").write_text(json.dumps({
        "providers": [{
            "name": "p1",
            "protocol": "openai",
            "api_key_env": "FAKE_API_KEY",
        }],
        "router": {"default": "p1:fake-model"},
    }))
    # Project settings.json sets the kill switch.
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / ".aura").mkdir()
    (project_dir / ".aura" / "settings.json").write_text(json.dumps({
        "permissions": {"disable_bypass": True},
    }))

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FAKE_API_KEY", "dummy")
    monkeypatch.setattr(sys, "argv", ["aura", "--bypass-permissions"])

    rc = main()
    captured = capsys.readouterr()
    assert rc == 2, (
        f"expected exit 2, got {rc}; "
        f"stdout={captured.out!r} stderr={captured.err!r}"
    )
    assert "--bypass-permissions is disabled" in captured.err
    assert "disable_bypass=true" in captured.err


def test_disable_bypass_false_allows_bypass_flag() -> None:
    # Regression lock: when disable_bypass is false (the default), the
    # --bypass-permissions flag still yields mode="bypass" via
    # _resolve_mode. We exercise the resolver directly because the full
    # main() path requires an LLM client; the kill-switch check lives
    # AFTER _resolve_mode in main() and is covered by the subprocess
    # test above.
    from aura.cli.__main__ import _resolve_mode
    from aura.schemas.permissions import PermissionsConfig

    perm_cfg = PermissionsConfig(disable_bypass=False)
    args = _ns(bypass_permissions=True)
    assert _resolve_mode(args, perm_cfg) == "bypass"  # type: ignore[arg-type]
    assert perm_cfg.disable_bypass is False
