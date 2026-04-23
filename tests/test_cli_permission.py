"""Tests for aura.cli.permission — inline numbered prompt.

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §8.1–§8.5 (with the
deliberate divergence from radiolist dialog to inline prompt — see the
module docstring for why).

The asker has ONE job: present the choice, capture the answer. It does
NOT decide, persist, or emit domain events — those are the hook's and
store's jobs. These tests pin that boundary.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from rich.console import Console

from aura.cli.permission import (
    _parse_choice,
    _render_prompt_block,
    _tag,
    make_cli_asker,
    print_bypass_banner,
)
from aura.core import journal as journal_module
from aura.core.hooks.permission import AskerResponse
from aura.core.permissions.matchers import exact_match_on
from aura.core.permissions.rule import Rule
from aura.tools.base import build_tool


class _P(BaseModel):
    command: str = ""
    msg: str = ""


def _noop(**_: Any) -> dict[str, Any]:
    return {}


def _bash_like(
    *,
    is_destructive: bool = False,
    is_read_only: bool = False,
    with_matcher: bool = True,
    with_preview: bool = True,
) -> BaseTool:
    return build_tool(
        name="bash",
        description="bash-like tool",
        args_schema=_P,
        func=_noop,
        is_destructive=is_destructive,
        is_read_only=is_read_only,
        rule_matcher=exact_match_on("command") if with_matcher else None,
        args_preview=(
            (lambda args: f"command: {args.get('command', '')}")
            if with_preview
            else None
        ),
    )


_HINT = Rule(tool="bash", content=None)  # hint arg is unused now; kept for signature


def _capture_console() -> tuple[Console, io.StringIO]:
    """Return a Console that writes to an in-memory buffer.

    ``force_terminal=False`` strips ANSI codes so substring assertions
    work against plain text; ``color_system=None`` disables all styling
    so rich markup doesn't leak through.
    """
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    return console, buf


def _stub_read(responses: list[str]) -> Any:
    """Return an async fn that pops from ``responses`` on each call.

    Install via ``monkeypatch.setattr('aura.cli.permission._read_choice', fn)``.
    Tests supply a list of raw strings (what the user types on each
    reprompt) — the asker re-reads until a valid choice is given.
    """
    iterator = iter(responses)

    async def _fake(prompt: str = "  ❯ ") -> str:  # noqa: ARG001
        try:
            return next(iterator)
        except StopIteration as exc:
            raise EOFError("test responses exhausted") from exc

    return _fake


def _stub_read_raises(exc: BaseException) -> Any:
    async def _fake(prompt: str = "  ❯ ") -> str:  # noqa: ARG001
        raise exc

    return _fake


@pytest.fixture
def _journal_capture(tmp_path: Path) -> Any:
    log = tmp_path / "journal.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        yield log
    finally:
        journal_module.reset()


def _events(log: Path) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# _parse_choice — pure parser
# ---------------------------------------------------------------------------
def test_parse_empty_returns_default() -> None:
    assert _parse_choice("", default=1) == 1
    assert _parse_choice("   ", default=3) == 3


@pytest.mark.parametrize("token", ["1", "y", "yes", "Y", "YES", "  1  "])
def test_parse_token_one(token: str) -> None:
    assert _parse_choice(token, default=3) == 1


@pytest.mark.parametrize("token", ["2", "a", "always", "ALWAYS"])
def test_parse_token_two(token: str) -> None:
    assert _parse_choice(token, default=1) == 2


@pytest.mark.parametrize("token", ["3", "n", "no", "No"])
def test_parse_token_three(token: str) -> None:
    assert _parse_choice(token, default=1) == 3


@pytest.mark.parametrize("token", ["4", "maybe", "ok", "abc123"])
def test_parse_invalid_returns_none(token: str) -> None:
    # None = "reprompt" signal. Don't silently default on garbage — that
    # would make typos destructive (e.g. typo on a destructive prompt
    # could fall through to accept).
    assert _parse_choice(token, default=1) is None


# ---------------------------------------------------------------------------
# _render_prompt_block — output shape
# ---------------------------------------------------------------------------
def test_render_block_includes_tool_name_and_tag() -> None:
    console, buf = _capture_console()
    tool = _bash_like(is_destructive=True)
    _render_prompt_block(
        console,
        tool=tool,
        preview="command: rm -rf /",
        tag=_tag(tool),
        option_two_label="Yes, and always allow `bash(rm -rf /)` in this project",
        default_choice=3,
    )
    out = buf.getvalue()
    assert "bash" in out
    assert "destructive" in out
    assert "rm -rf /" in out
    assert "1. Yes" in out
    assert "2. Yes, and always allow" in out
    assert "3. No" in out


def test_render_block_marks_default_option_with_cursor() -> None:
    console, buf = _capture_console()
    tool = _bash_like(is_destructive=True)
    _render_prompt_block(
        console, tool=tool, preview="x", tag="destructive",
        option_two_label="always", default_choice=3,
    )
    out = buf.getvalue()
    # ❯ appears exactly once (next to the default). Using a loose check
    # because rich may emit other glyphs; just verify the default's line
    # differs from the others.
    lines = [ln for ln in out.splitlines() if "1." in ln or "2." in ln or "3." in ln]
    assert len(lines) == 3
    assert lines[2].lstrip().startswith("❯")  # 3. No is the default


def test_render_block_omits_preview_when_equal_to_tool_name() -> None:
    console, buf = _capture_console()
    tool = _bash_like(with_preview=False)  # preview falls back to "bash"
    _render_prompt_block(
        console, tool=tool, preview="bash", tag="safe",
        option_two_label="...", default_choice=1,
    )
    out = buf.getvalue()
    # "bash" is on the header line; there should be NO separate dim
    # preview line that only contains "bash" again.
    occurrences = out.count("bash")
    assert occurrences == 1


def test_render_block_shows_enter_default_hint() -> None:
    console, buf = _capture_console()
    tool = _bash_like()
    _render_prompt_block(
        console, tool=tool, preview="x", tag="safe",
        option_two_label="...", default_choice=1,
    )
    out = buf.getvalue()
    assert "Enter = default" in out
    assert "Ctrl+C" in out


def test_render_block_no_box_border_chars() -> None:
    # Explicit regression guard: the whole point of this rewrite is to
    # remove the boxed dialog. If a future refactor accidentally brings
    # back Panel / box glyphs, this breaks.
    console, buf = _capture_console()
    tool = _bash_like()
    _render_prompt_block(
        console, tool=tool, preview="x", tag="safe",
        option_two_label="...", default_choice=1,
    )
    out = buf.getvalue()
    for glyph in ("╭", "╮", "╰", "╯", "│", "─", "┌", "┐", "└", "┘"):
        assert glyph not in out, f"box-drawing glyph {glyph!r} leaked into inline prompt"


# ---------------------------------------------------------------------------
# make_cli_asker — full flow
# ---------------------------------------------------------------------------
async def test_accept_returns_accept(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    monkeypatch.setattr("aura.cli.permission._read_choice", _stub_read(["1"]))
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(), args={"command": "npm test"}, rule_hint=_HINT,
    )
    assert isinstance(resp, AskerResponse)
    assert resp.choice == "accept"
    assert resp.rule is None
    events = {e["event"] for e in _events(_journal_capture)}
    assert "permission_asked" in events
    assert "permission_answered" in events


async def test_deny_returns_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aura.cli.permission._read_choice", _stub_read(["3"]))
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(), args={"command": "rm -rf /"}, rule_hint=_HINT,
    )
    assert resp.choice == "deny"


async def test_empty_line_accepts_default_for_safe_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Safe tool default is "1. Yes". Pressing Enter with no input must
    # resolve to accept, NOT deny — matches claude-code convention.
    monkeypatch.setattr("aura.cli.permission._read_choice", _stub_read([""]))
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(is_destructive=False),
        args={"command": "ls"},
        rule_hint=_HINT,
    )
    assert resp.choice == "accept"


async def test_empty_line_denies_default_for_destructive_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Destructive tool default is "3. No". Bare Enter must resolve to
    # deny so an unattended Enter can't destroy anything.
    monkeypatch.setattr("aura.cli.permission._read_choice", _stub_read([""]))
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(is_destructive=True),
        args={"command": "rm"},
        rule_hint=_HINT,
    )
    assert resp.choice == "deny"


async def test_always_with_precise_rule_is_project_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aura.cli.permission._read_choice", _stub_read(["2"]))
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(with_matcher=True),
        args={"command": "npm test"},
        rule_hint=_HINT,
    )
    assert resp.choice == "always"
    assert resp.scope == "project"
    assert resp.rule == Rule(tool="bash", content="npm test")


async def test_always_fallback_to_session_when_no_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aura.cli.permission._read_choice", _stub_read(["2"]))
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    tool = _bash_like(with_matcher=False)
    resp = await asker(tool=tool, args={"command": "anything"}, rule_hint=_HINT)
    assert resp.choice == "always"
    assert resp.scope == "session"
    assert resp.rule == Rule(tool="bash", content=None)


async def test_invalid_then_valid_reprompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First two tokens are garbage, third is "1". The asker must keep
    # asking, then accept on the good token. Validates the reprompt
    # loop without silently defaulting on typos.
    monkeypatch.setattr(
        "aura.cli.permission._read_choice",
        _stub_read(["4", "abc", "1"]),
    )
    console, buf = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(), args={"command": "ls"}, rule_hint=_HINT,
    )
    assert resp.choice == "accept"
    # Hint was printed at least once.
    assert "Please enter 1, 2, or 3" in buf.getvalue()


async def test_ctrl_c_is_deny_not_propagate(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    # The permission prompt treats Ctrl+C as "no to THIS tool call", not
    # "kill the agent turn" — matches claude-code UX. (Upstream turn
    # cancellation still works because pt's prompt_async surfaces
    # KeyboardInterrupt only inside the prompt, not outside it.)
    monkeypatch.setattr(
        "aura.cli.permission._read_choice", _stub_read_raises(KeyboardInterrupt()),
    )
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(), args={"command": "x"}, rule_hint=_HINT,
    )
    assert resp.choice == "deny"
    answered = [
        e for e in _events(_journal_capture) if e["event"] == "permission_answered"
    ]
    assert len(answered) == 1
    assert answered[0]["choice"] == "deny"


async def test_eof_is_deny(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    # Piped stdin hit end → fail-closed deny.
    monkeypatch.setattr(
        "aura.cli.permission._read_choice", _stub_read_raises(EOFError()),
    )
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(), args={"command": "x"}, rule_hint=_HINT,
    )
    assert resp.choice == "deny"


async def test_read_unavailable_raises_writes_journal(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    # Any other exception from _read_choice → journal as "unavailable"
    # and deny. Mirrors the old no-tty path.
    monkeypatch.setattr(
        "aura.cli.permission._read_choice",
        _stub_read_raises(RuntimeError("no tty")),
    )
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(), args={"command": "x"}, rule_hint=_HINT,
    )
    assert resp.choice == "deny"
    events = {e["event"] for e in _events(_journal_capture)}
    assert "permission_prompt_unavailable" in events


async def test_destructive_tool_shows_red_tag_in_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aura.cli.permission._read_choice", _stub_read(["3"]))
    console, buf = _capture_console()
    asker = make_cli_asker(console=console)
    await asker(
        tool=_bash_like(is_destructive=True),
        args={"command": "rm"},
        rule_hint=_HINT,
    )
    out = buf.getvalue()
    assert "destructive" in out


async def test_read_only_tool_shows_tag_in_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aura.cli.permission._read_choice", _stub_read(["1"]))
    console, buf = _capture_console()
    asker = make_cli_asker(console=console)
    await asker(
        tool=_bash_like(is_read_only=True),
        args={"command": "cat x"},
        rule_hint=_HINT,
    )
    out = buf.getvalue()
    assert "read-only" in out


async def test_long_preview_is_truncated_with_ellipsis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aura.cli.permission._read_choice", _stub_read(["3"]))
    console, buf = _capture_console()
    big_cmd = "x" * 500
    tool = build_tool(
        name="bash",
        description="bash",
        args_schema=_P,
        func=_noop,
        args_preview=lambda args: f"command: {args.get('command', '')}",
    )
    asker = make_cli_asker(console=console)
    await asker(tool=tool, args={"command": big_cmd}, rule_hint=_HINT)
    out = buf.getvalue()
    assert "…" in out
    # Every line is bounded — even with wrapping, the preview one-line
    # block can't exceed the cap.
    for line in out.splitlines():
        if "command:" in line:
            assert len(line) <= _PREVIEW_MAX_CHARS_WITH_INDENT


async def test_option_two_label_includes_project_rule_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aura.cli.permission._read_choice", _stub_read(["1"]))
    console, buf = _capture_console()
    asker = make_cli_asker(console=console)
    await asker(
        tool=_bash_like(with_matcher=True),
        args={"command": "npm test"},
        rule_hint=_HINT,
    )
    out = buf.getvalue()
    assert "bash(npm test)" in out
    assert "in this project" in out


async def test_option_two_label_falls_back_to_session_without_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aura.cli.permission._read_choice", _stub_read(["1"]))
    console, buf = _capture_console()
    asker = make_cli_asker(console=console)
    tool = _bash_like(with_matcher=False)
    await asker(tool=tool, args={"command": "x"}, rule_hint=_HINT)
    out = buf.getvalue()
    # No precise rule → tool-wide / session language.
    assert "for this session" in out
    assert "bash" in out


# ---------------------------------------------------------------------------
# print_bypass_banner — unchanged contract
# ---------------------------------------------------------------------------
def test_print_bypass_banner_writes_warning() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=120)
    print_bypass_banner(console)
    out = buffer.getvalue()
    assert "PERMISSION CHECKS DISABLED" in out


# Indent-adjusted version of _PREVIEW_MAX_CHARS — rich prints the preview
# with a two-space indent, so the visual line is up to the cap plus 2
# spaces of leading indent. Kept as a local constant so the assertion
# above reads explicitly.
_PREVIEW_MAX_CHARS_WITH_INDENT = 202
