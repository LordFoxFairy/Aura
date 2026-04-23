"""Tests for the user-configurable StatusLine hook.

Covers:
- the hook runner (aura.cli.statusline_hook) — success, non-zero exit,
  timeout, spawn failure, empty output paths;
- the envelope schema (stable v1 keys);
- render_bottom_toolbar_with_hook — hook output reaches the bar
  (ANSI-wrapped) vs. silent fallback on failure;
- the StatusLineConfig pydantic model — defaults, timeout clamp, is_active;
- the settings.json round-trip — unset section is fine, disabled section
  is honored, both project and local files contribute.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from aura.cli.status_bar import (
    render_bottom_toolbar_html,
    render_bottom_toolbar_with_hook,
)
from aura.cli.statusline_hook import (
    STATUSLINE_ENVELOPE_VERSION,
    build_envelope,
    run_statusline_command,
)
from aura.core.permissions import store
from aura.schemas.permissions import PermissionsConfig, StatusLineConfig

# ---------------------------------------------------------------------------
# run_statusline_command — happy path + every documented failure mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_returns_stdout_on_success() -> None:
    out = await run_statusline_command(
        command="printf 'hello-bar'",
        timeout_seconds=1.0,
        envelope={"version": "1"},
    )
    assert out == "hello-bar"


@pytest.mark.asyncio
async def test_runner_strips_trailing_newline_only() -> None:
    # Internal whitespace and ANSI escapes are sacred; only the trailing
    # \n from ``echo`` should be stripped.
    out = await run_statusline_command(
        command="printf '  spaced \\x1b[31mred\\x1b[0m\\n'",
        timeout_seconds=1.0,
        envelope={"version": "1"},
    )
    assert out is not None
    assert out.startswith("  spaced ")
    assert "\x1b[31mred\x1b[0m" in out
    assert not out.endswith("\n")


@pytest.mark.asyncio
async def test_runner_receives_envelope_on_stdin(tmp_path: Path) -> None:
    # Use a python -c script to faithfully echo the stdin back to a file,
    # then assert the JSON envelope arrived intact.
    outfile = tmp_path / "env.json"
    cmd = (
        f"{sys.executable} -c "
        f"\"import sys; open('{outfile}','w').write(sys.stdin.read()); print('ok')\""
    )
    envelope = {"version": "1", "model": "x", "mode": "plan"}
    out = await run_statusline_command(
        command=cmd, timeout_seconds=5.0, envelope=envelope,
    )
    assert out == "ok"
    assert json.loads(outfile.read_text()) == envelope


@pytest.mark.asyncio
async def test_runner_nonzero_exit_returns_none() -> None:
    out = await run_statusline_command(
        command="exit 7",
        timeout_seconds=1.0,
        envelope={"version": "1"},
    )
    assert out is None


@pytest.mark.asyncio
async def test_runner_empty_stdout_returns_none() -> None:
    out = await run_statusline_command(
        command="true",
        timeout_seconds=1.0,
        envelope={"version": "1"},
    )
    assert out is None


@pytest.mark.asyncio
async def test_runner_timeout_kills_child_and_returns_none() -> None:
    # Sleep past the tight timeout; runner must kill the child and fall
    # back to None promptly (not hang the test suite).
    started = asyncio.get_event_loop().time()
    out = await run_statusline_command(
        command="sleep 5",
        timeout_seconds=0.1,
        envelope={"version": "1"},
    )
    elapsed = asyncio.get_event_loop().time() - started
    assert out is None
    # Tight upper bound so a silent regression to "wait for the child"
    # is caught — 2s is very generous for a 100ms budget.
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_runner_command_not_found_returns_none() -> None:
    # ``create_subprocess_shell`` routes through /bin/sh, so a missing
    # binary surfaces as a non-zero exit (127), not a spawn OSError.
    # Either way, the runner returns None — the bar falls back cleanly.
    out = await run_statusline_command(
        command="this_binary_definitely_does_not_exist_aura_12345",
        timeout_seconds=1.0,
        envelope={"version": "1"},
    )
    assert out is None


# ---------------------------------------------------------------------------
# build_envelope — v1 schema stability
# ---------------------------------------------------------------------------


def test_envelope_has_all_documented_keys() -> None:
    env = build_envelope(
        model="deepseek:glm-5",
        context_window_size=512_000,
        input_tokens=8_200,
        cache_read_tokens=0,
        pinned_estimate_tokens=1_500,
        mode="default",
        cwd="/some/cwd",
        last_turn_seconds=3.4,
    )
    assert env["version"] == STATUSLINE_ENVELOPE_VERSION == "1"
    assert env["model"] == "deepseek:glm-5"
    assert env["context_window"] == {
        "size": 512_000, "used": 8_200, "pct": 2,
    }
    assert env["tokens"] == {
        "input": 8_200, "cache_read": 0, "pinned_estimate": 1_500,
    }
    assert env["mode"] == "default"
    assert env["cwd"] == "/some/cwd"
    assert env["last_turn_seconds"] == 3.4
    # Exactly these top-level keys — no accidental leakage.
    assert set(env) == {
        "version", "model", "context_window", "tokens",
        "mode", "cwd", "last_turn_seconds",
    }


def test_envelope_is_json_serializable() -> None:
    env = build_envelope(
        model=None,
        context_window_size=0,
        input_tokens=0,
        cache_read_tokens=0,
        pinned_estimate_tokens=0,
        mode="default",
        cwd="/",
        last_turn_seconds=0.0,
    )
    # Must round-trip — the runner json.dumps() it onto the child's stdin.
    assert json.loads(json.dumps(env)) == env


# ---------------------------------------------------------------------------
# render_bottom_toolbar_with_hook — integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_output_replaces_default_toolbar(tmp_path: Path) -> None:
    # Hook returns "XXX"; the bar MUST reflect "XXX" (wrapped as ANSI),
    # not the default humanized-tokens render.
    from prompt_toolkit.formatted_text import ANSI, to_formatted_text

    result = await render_bottom_toolbar_with_hook(
        model="m",
        input_tokens=100,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
        hook_command="printf 'XXX-BAR'",
        hook_timeout_s=2.0,
    )
    assert isinstance(result, ANSI)
    # Pull the rendered plain text out of the pt ANSI wrapper.
    pieces = to_formatted_text(result)
    text = "".join(seg for _, seg, *_ in pieces)
    assert "XXX-BAR" in text


@pytest.mark.asyncio
async def test_hook_nonzero_exit_falls_back_to_default(tmp_path: Path) -> None:
    result = await render_bottom_toolbar_with_hook(
        model="m",
        input_tokens=100,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
        hook_command="exit 1",
        hook_timeout_s=2.0,
    )
    expected = render_bottom_toolbar_html(
        model="m",
        input_tokens=100,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
    )
    assert str(result) == str(expected)


@pytest.mark.asyncio
async def test_hook_timeout_falls_back_to_default(tmp_path: Path) -> None:
    result = await render_bottom_toolbar_with_hook(
        model="m",
        input_tokens=100,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
        hook_command="sleep 5",
        hook_timeout_s=0.1,
    )
    expected = render_bottom_toolbar_html(
        model="m",
        input_tokens=100,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
    )
    assert str(result) == str(expected)


@pytest.mark.asyncio
async def test_hook_unset_returns_default_no_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If hook_command is None, we must NOT spawn a subprocess — assert
    # by patching the runner to blow up when called.
    async def _boom(**_: Any) -> str:
        raise AssertionError("runner should not be invoked when no hook set")
    monkeypatch.setattr(
        "aura.cli.statusline_hook.run_statusline_command", _boom,
    )
    result = await render_bottom_toolbar_with_hook(
        model="m",
        input_tokens=100,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
        hook_command=None,
    )
    expected = render_bottom_toolbar_html(
        model="m",
        input_tokens=100,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
    )
    assert str(result) == str(expected)


@pytest.mark.asyncio
async def test_hook_empty_output_falls_back(tmp_path: Path) -> None:
    # Empty stdout — runner returns None, bar falls back.
    result = await render_bottom_toolbar_with_hook(
        model="m",
        input_tokens=100,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
        hook_command="true",
        hook_timeout_s=2.0,
    )
    expected = render_bottom_toolbar_html(
        model="m",
        input_tokens=100,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
    )
    assert str(result) == str(expected)


# ---------------------------------------------------------------------------
# StatusLineConfig — schema
# ---------------------------------------------------------------------------


def test_statusline_config_defaults_are_inert() -> None:
    cfg = StatusLineConfig()
    assert cfg.command == ""
    assert cfg.timeout_ms == 500
    assert cfg.enabled is True
    # No command ⇒ inactive even though enabled is True.
    assert cfg.is_active is False


def test_statusline_config_is_active_requires_command() -> None:
    # Whitespace-only command is NOT a valid command — treat as inert so
    # typos don't silently break the bar.
    assert StatusLineConfig(command="   ").is_active is False
    assert StatusLineConfig(command="echo hi").is_active is True


def test_statusline_config_is_active_respects_enabled_flag() -> None:
    assert StatusLineConfig(command="echo hi", enabled=False).is_active is False
    assert StatusLineConfig(command="echo hi", enabled=True).is_active is True


def test_statusline_config_timeout_clamp() -> None:
    # Below 50ms clamps up, above 5000 clamps down, in-range preserved.
    assert StatusLineConfig(timeout_ms=10).timeout_ms == 50
    assert StatusLineConfig(timeout_ms=99_999).timeout_ms == 5000
    assert StatusLineConfig(timeout_ms=750).timeout_ms == 750


def test_statusline_config_rejects_unknown_fields() -> None:
    # Silent typos in security-adjacent config are unacceptable.
    with pytest.raises(Exception):  # noqa: B017
        StatusLineConfig.model_validate({"commnd": "echo hi"})


# ---------------------------------------------------------------------------
# Settings.json round-trip
# ---------------------------------------------------------------------------


def test_permissions_config_without_statusline_loads_cleanly(
    tmp_path: Path,
) -> None:
    # Backwards compatibility: existing settings.json files have no
    # "statusline" key; load() must not blow up.
    (tmp_path / ".aura").mkdir()
    (tmp_path / ".aura" / "settings.json").write_text(
        json.dumps({"permissions": {"allow": ["bash"]}})
    )
    cfg = store.load(tmp_path)
    assert cfg.statusline is None
    assert cfg.allow == ["bash"]


def test_permissions_config_with_statusline_enabled_loads(tmp_path: Path) -> None:
    (tmp_path / ".aura").mkdir()
    (tmp_path / ".aura" / "settings.json").write_text(
        json.dumps({
            "permissions": {
                "allow": [],
                "statusline": {
                    "command": "echo hi",
                    "timeout_ms": 250,
                    "enabled": True,
                },
            },
        })
    )
    cfg = store.load(tmp_path)
    assert isinstance(cfg.statusline, StatusLineConfig)
    assert cfg.statusline.command == "echo hi"
    assert cfg.statusline.timeout_ms == 250
    assert cfg.statusline.is_active is True


def test_permissions_config_with_statusline_disabled_is_inactive(
    tmp_path: Path,
) -> None:
    (tmp_path / ".aura").mkdir()
    (tmp_path / ".aura" / "settings.json").write_text(
        json.dumps({
            "permissions": {
                "statusline": {
                    "command": "echo hi",
                    "enabled": False,
                },
            },
        })
    )
    cfg = store.load(tmp_path)
    assert cfg.statusline is not None
    assert cfg.statusline.is_active is False


def test_permissions_config_statusline_local_overrides_project(
    tmp_path: Path,
) -> None:
    # settings.local.json is the per-machine override surface; its
    # statusline wins entirely when both files set one.
    (tmp_path / ".aura").mkdir()
    (tmp_path / ".aura" / "settings.json").write_text(
        json.dumps({
            "permissions": {
                "statusline": {"command": "project-bar"},
            },
        })
    )
    (tmp_path / ".aura" / "settings.local.json").write_text(
        json.dumps({
            "permissions": {
                "statusline": {"command": "local-bar"},
            },
        })
    )
    cfg = store.load(tmp_path)
    assert cfg.statusline is not None
    assert cfg.statusline.command == "local-bar"


def test_permissions_config_statusline_project_used_when_local_absent(
    tmp_path: Path,
) -> None:
    (tmp_path / ".aura").mkdir()
    (tmp_path / ".aura" / "settings.json").write_text(
        json.dumps({
            "permissions": {
                "statusline": {"command": "project-bar"},
            },
        })
    )
    cfg = store.load(tmp_path)
    assert cfg.statusline is not None
    assert cfg.statusline.command == "project-bar"


def test_permissions_config_rejects_bad_statusline_field(
    tmp_path: Path,
) -> None:
    from aura.config.schema import AuraConfigError

    (tmp_path / ".aura").mkdir()
    (tmp_path / ".aura" / "settings.json").write_text(
        json.dumps({
            "permissions": {
                "statusline": {"xyz": "nope"},
            },
        })
    )
    with pytest.raises(AuraConfigError):
        store.load(tmp_path)


# ---------------------------------------------------------------------------
# PermissionsConfig surface — top-level
# ---------------------------------------------------------------------------


def test_permissions_config_statusline_is_optional() -> None:
    # Absent is fine; it's the common case today.
    cfg = PermissionsConfig()
    assert cfg.statusline is None
