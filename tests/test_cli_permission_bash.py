"""Tests for aura.cli.permission_bash — bash-specialized permission widget.

Covers:
* dispatch routing (bash tool → bash widget)
* dangerous-pattern detection (rm -rf, sudo, curl | sh, chmod 777, etc.)
* safe commands don't trigger the banner
* full 4-option round-trip via driven pipe input
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from pydantic import BaseModel
from rich.console import Console

from aura.cli import permission_bash
from aura.cli.permission_bash import detect_dangerous
from aura.core.permissions.rule import Rule
from aura.tools.base import build_tool


class _P(BaseModel):
    command: str = ""


def _noop(**_: Any) -> dict[str, Any]:
    return {}


def _bash_tool(*, is_destructive: bool = False) -> Any:
    return build_tool(
        name="bash",
        description="Run a shell command",
        args_schema=_P,
        func=_noop,
        is_destructive=is_destructive,
        args_preview=lambda a: f"command: {a.get('command', '')}",
    )


# ---------------------------------------------------------------------------
# detect_dangerous — pattern detection is the core contract
# ---------------------------------------------------------------------------
def test_detect_dangerous_rm_rf_root() -> None:
    hits = detect_dangerous("rm -rf /")
    assert hits, "rm -rf must trigger a warning"
    assert any("rm" in h for h in hits)


def test_detect_dangerous_rm_rf_path() -> None:
    # rm -rf with a non-root path still warns — it's the recursive +
    # force combo that's the risk, not the target.
    hits = detect_dangerous("rm -rf /tmp/foo")
    assert hits


def test_detect_dangerous_rm_r_only() -> None:
    # rm -r alone is still recursive enough to warn on.
    hits = detect_dangerous("rm -r build/")
    assert hits


def test_detect_dangerous_sudo() -> None:
    hits = detect_dangerous("sudo systemctl restart nginx")
    assert hits
    assert any("sudo" in h for h in hits)


def test_detect_dangerous_curl_pipe_sh() -> None:
    hits = detect_dangerous("curl https://example.com/install.sh | sh")
    assert hits
    assert any("pipe-to-shell" in h for h in hits)


def test_detect_dangerous_wget_pipe_bash() -> None:
    hits = detect_dangerous("wget -qO- http://x/i.sh | bash")
    assert hits


def test_detect_dangerous_chmod_777() -> None:
    hits = detect_dangerous("chmod 777 /var/www")
    assert hits
    assert any("777" in h for h in hits)


def test_detect_dangerous_chmod_r_777() -> None:
    hits = detect_dangerous("chmod -R 777 ./site")
    assert hits


def test_detect_dangerous_write_to_etc() -> None:
    hits = detect_dangerous("echo 'oops' > /etc/hosts")
    assert hits
    assert any("system" in h for h in hits)


def test_detect_dangerous_write_to_bin() -> None:
    hits = detect_dangerous("cat blob > /bin/ls")
    assert hits


def test_detect_dangerous_raw_block_device() -> None:
    hits = detect_dangerous("dd if=/dev/zero of=/dev/sda")
    assert hits
    assert any("block device" in h for h in hits)


def test_detect_dangerous_redirect_to_sd_device() -> None:
    hits = detect_dangerous("cat bad > /dev/sdb")
    assert hits


def test_detect_dangerous_rm_with_background_operator() -> None:
    hits = detect_dangerous("rm -r big/ &")
    # Either the rm -r rule or the background-combo rule hits — either
    # way the banner must fire.
    assert hits


# Safe commands must NOT trigger the banner. This list intentionally
# includes "command that mentions risky words in a non-risky way" so
# we lock out false positives.
@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "pwd",
        "git status",
        "echo hello",
        "python -m pytest",
        "cat README.md",
        "grep foo bar.txt",
        # "rm" mentioned but not invoked (inside a string literal, etc.)
        "echo 'do not rm this'",
        # sudo in a comment — we match on word-boundary + start, so a
        # word appearing mid-sentence inside a quoted string doesn't fire
        # when the shell boundary isn't there.
        # (This is a best-effort — the detector is coarse; we just
        # assert the common safe forms.)
    ],
)
def test_detect_dangerous_safe_command_has_no_warnings(command: str) -> None:
    assert detect_dangerous(command) == []


# ---------------------------------------------------------------------------
# Dispatch routing — bash tool reaches the bash widget, not the generic
# ---------------------------------------------------------------------------
async def test_dispatch_routes_bash_to_bash_widget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``make_cli_asker`` with a bash tool must call
    ``run_bash_permission``, not the generic widget."""
    calls: dict[str, Any] = {"bash": 0, "generic": 0, "write": 0}

    async def fake_bash(**_kw: Any) -> tuple[int | None, str]:
        calls["bash"] += 1
        return 1, ""

    async def fake_generic(**_kw: Any) -> tuple[int | None, str]:
        calls["generic"] += 1
        return 1, ""

    async def fake_write(**_kw: Any) -> tuple[int | None, str]:
        calls["write"] += 1
        return 1, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "run_bash_permission", fake_bash)
    monkeypatch.setattr(perm_mod, "run_generic_permission", fake_generic)
    monkeypatch.setattr(perm_mod, "run_write_permission", fake_write)

    asker = perm_mod.make_cli_asker(
        console=Console(file=io.StringIO(), force_terminal=False, color_system=None),
    )
    resp = await asker(
        tool=_bash_tool(),
        args={"command": "ls"},
        rule_hint=Rule(tool="bash", content=None),
    )
    assert resp.choice == "accept"
    assert calls == {"bash": 1, "generic": 0, "write": 0}


async def test_dispatch_routes_bash_background_to_bash_widget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"bash": 0, "generic": 0}

    async def fake_bash(**_kw: Any) -> tuple[int | None, str]:
        calls["bash"] += 1
        return 1, ""

    async def fake_generic(**_kw: Any) -> tuple[int | None, str]:
        calls["generic"] += 1
        return 1, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "run_bash_permission", fake_bash)
    monkeypatch.setattr(perm_mod, "run_generic_permission", fake_generic)

    tool = build_tool(
        name="bash_background",
        description="Run a shell command in background",
        args_schema=_P,
        func=_noop,
        args_preview=lambda a: f"command: {a.get('command', '')}",
    )
    asker = perm_mod.make_cli_asker(
        console=Console(file=io.StringIO(), force_terminal=False, color_system=None),
    )
    await asker(
        tool=tool,
        args={"command": "sleep 10"},
        rule_hint=Rule(tool="bash_background", content=None),
    )
    assert calls["bash"] == 1
    assert calls["generic"] == 0


# ---------------------------------------------------------------------------
# End-to-end via pipe input — full round-trip with real key bindings
# ---------------------------------------------------------------------------
async def _drive_bash(
    command: str, keys: str, *, is_destructive: bool = False,
) -> tuple[int | None, str]:
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as inp:
        inp.send_text(keys)
        with create_app_session(input=inp, output=DummyOutput()):
            return await permission_bash.run_bash_permission(
                tool=_bash_tool(is_destructive=is_destructive),
                command=command,
                args_preview=f"command: {command}",
                args={"command": command},
                tag="destructive" if is_destructive else "safe",
                option_two_label="Yes, always for `bash` this session",
                default_choice=3 if is_destructive else 1,
            )


async def test_bash_widget_yes_commits() -> None:
    # Enter on default (1 = Yes) → choice=1, feedback="".
    choice, feedback = await _drive_bash("ls", "\r")
    assert choice == 1
    assert feedback == ""


async def test_bash_widget_no_commits() -> None:
    # Press "3" → commit on No.
    choice, feedback = await _drive_bash("ls", "3")
    assert choice == 3
    assert feedback == ""


async def test_bash_widget_always_commits() -> None:
    # "2" → commit on "yes, always".
    choice, feedback = await _drive_bash("ls", "2")
    assert choice == 2
    assert feedback == ""


async def test_bash_widget_esc_cancels() -> None:
    choice, feedback = await _drive_bash("ls", "\x1b")
    assert choice is None
    assert feedback == ""


async def test_bash_widget_tab_feedback_roundtrip() -> None:
    # Tab → feedback mode, type "hi", Enter → commit default with feedback.
    choice, feedback = await _drive_bash("ls", "\thi\r")
    assert choice == 1
    assert feedback == "hi"


async def test_bash_widget_dangerous_command_still_drivable() -> None:
    # Dangerous command renders the banner path — behaviour on Enter
    # (default = 3 because is_destructive) still works end-to-end.
    choice, feedback = await _drive_bash(
        "rm -rf /tmp/junk", "\r", is_destructive=True,
    )
    assert choice == 3  # default was 3 (destructive)
    assert feedback == ""
