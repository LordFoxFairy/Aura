"""Tests for aura.cli.permission — inline interactive list widget.

The heart of the v0.7.5 rewrite is the ``_pick_choice_interactive``
prompt_toolkit Application. Testing pt Applications in-process is
brittle (they want a real terminal); so we test the flow by stubbing
``_pick_choice_interactive`` with a fake async function that returns a
preset choice. The unit-level tests below pin: journal events, rule
scope derivation, tag classification, audit-line rendering, and the
asker's mapping from picker-return-value → AskerResponse.

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §8.1–§8.5 (with the
deliberate divergence from radiolist dialog to inline interactive
widget — see the module docstring for why).
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
    _compose_option_two,
    _preview,
    _render_decision_audit_line,
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
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    return console, buf


def _stub_picker(
    return_value: int | None, feedback: str = "",
) -> Any:
    """Async stub for ``_pick_choice_interactive``. Returns ``(choice,
    feedback)`` on every call; records the call kwargs for assertion.

    Default ``feedback=""`` keeps backwards compat with every existing
    test that only cares about ``choice``.
    """
    captured: dict[str, Any] = {}

    async def _picker(**kwargs: Any) -> tuple[int | None, str]:
        captured.update(kwargs)
        return return_value, feedback

    return _picker, captured


def _stub_picker_raises(exc: BaseException) -> Any:
    async def _picker(**_: Any) -> tuple[int | None, str]:
        raise exc

    return _picker


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
# _tag — classification
# ---------------------------------------------------------------------------
def test_tag_destructive_wins_over_read_only() -> None:
    tool = _bash_like(is_destructive=True, is_read_only=True)
    assert _tag(tool) == "destructive"


def test_tag_read_only_when_flagged() -> None:
    assert _tag(_bash_like(is_read_only=True)) == "read-only"


def test_tag_safe_when_no_flags() -> None:
    assert _tag(_bash_like()) == "safe"


# ---------------------------------------------------------------------------
# _preview — args → one-line string
# ---------------------------------------------------------------------------
def test_preview_uses_tool_args_preview_fn_and_strips_prefix() -> None:
    # Under the new widget design the header already names the tool,
    # so the "command: " prefix that ``bash`` emits from its args_preview
    # is redundant — the asker strips it so the scrollback line reads
    # just ``ls`` (not ``command: ls``).
    tool = _bash_like()
    assert _preview(tool, {"command": "ls"}) == "ls"


def test_preview_falls_back_to_tool_name_when_no_fn() -> None:
    tool = _bash_like(with_preview=False)
    assert _preview(tool, {"command": "ls"}) == "bash"


def test_preview_truncates_at_cap() -> None:
    tool = _bash_like()
    long_cmd = "x" * 500
    out = _preview(tool, {"command": long_cmd})
    assert len(out) <= 200
    assert out.endswith("…")


def test_preview_swallows_exceptions_from_user_fn() -> None:
    # An args_preview fn that raises must not break the prompt — the
    # asker falls back to the bare tool name.
    def _bad_preview(_args: dict[str, Any]) -> str:
        raise RuntimeError("oops")

    tool = build_tool(
        name="bash",
        description="b",
        args_schema=_P,
        func=_noop,
        args_preview=_bad_preview,
    )
    assert _preview(tool, {"command": "x"}) == "bash"


# ---------------------------------------------------------------------------
# _compose_option_two — rule + scope derivation
# ---------------------------------------------------------------------------
def test_option_two_project_scope_with_matcher() -> None:
    tool = _bash_like(with_matcher=True)
    label, rule, scope = _compose_option_two(tool, {"command": "npm test"})
    assert scope == "project"
    assert rule == Rule(tool="bash", content="npm test")
    assert "in this project" in label
    assert "bash(npm test)" in label
    # v0.7.5 wording change: "don't ask again for" matches claude-code.
    assert "don't ask again" in label


def test_option_two_session_fallback_without_matcher() -> None:
    tool = _bash_like(with_matcher=False)
    label, rule, scope = _compose_option_two(tool, {"command": "x"})
    assert scope == "session"
    assert rule == Rule(tool="bash", content=None)
    assert "this session" in label
    assert "don't ask again" in label


# ---------------------------------------------------------------------------
# _render_decision_audit_line — post-widget scrollback trace
# ---------------------------------------------------------------------------
def test_audit_line_accept(tmp_path: Path) -> None:
    console, buf = _capture_console()
    _render_decision_audit_line(
        console,
        tool=_bash_like(),
        tag="safe",
        preview="command: ls",
        choice=1,
    )
    out = buf.getvalue()
    assert "bash" in out
    assert "command: ls" in out
    assert "yes" in out


def test_audit_line_always() -> None:
    console, buf = _capture_console()
    _render_decision_audit_line(
        console,
        tool=_bash_like(),
        tag="safe",
        preview="command: ls",
        choice=2,
    )
    assert "yes (always)" in buf.getvalue()


def test_audit_line_deny_destructive() -> None:
    console, buf = _capture_console()
    _render_decision_audit_line(
        console,
        tool=_bash_like(is_destructive=True),
        tag="destructive",
        preview="command: rm",
        choice=3,
    )
    out = buf.getvalue()
    assert "no" in out
    assert "rm" in out


def test_audit_line_cancelled_shows_explicit_state() -> None:
    # Ctrl+C / Esc returns None from the picker — the audit trail
    # must NOT silently look like "no"; it should say "cancelled" so
    # the transcript distinguishes intent from interruption.
    console, buf = _capture_console()
    _render_decision_audit_line(
        console,
        tool=_bash_like(),
        tag="safe",
        preview="command: x",
        choice=None,
    )
    assert "cancelled" in buf.getvalue()


# ---------------------------------------------------------------------------
# make_cli_asker — full flow, picker stubbed
# ---------------------------------------------------------------------------
async def test_accept_returns_accept(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    picker, _ = _stub_picker(1)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
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
    picker, _ = _stub_picker(3)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(), args={"command": "rm -rf /"}, rule_hint=_HINT,
    )
    assert resp.choice == "deny"


async def test_cancelled_picker_returns_deny(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    # Picker returns None (user Ctrl+C / Esc). That must resolve to
    # deny in the AskerResponse AND the journal.
    picker, _ = _stub_picker(None)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
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


async def test_always_with_precise_rule_is_project_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    picker, _ = _stub_picker(2)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
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
    picker, _ = _stub_picker(2)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    tool = _bash_like(with_matcher=False)
    resp = await asker(tool=tool, args={"command": "anything"}, rule_hint=_HINT)
    assert resp.choice == "always"
    assert resp.scope == "session"
    assert resp.rule == Rule(tool="bash", content=None)


async def test_picker_default_is_3_for_destructive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The asker MUST pass default_choice=3 to the picker for destructive
    # tools — so ↑/↓ cursor starts on "No" and a blind Enter can't
    # destroy anything.
    picker, captured = _stub_picker(3)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    await asker(
        tool=_bash_like(is_destructive=True),
        args={"command": "rm"},
        rule_hint=_HINT,
    )
    assert captured["default_choice"] == 3


async def test_picker_default_is_1_for_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    picker, captured = _stub_picker(1)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    await asker(
        tool=_bash_like(is_destructive=False),
        args={"command": "ls"},
        rule_hint=_HINT,
    )
    assert captured["default_choice"] == 1


async def test_picker_receives_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    picker, captured = _stub_picker(3)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    await asker(
        tool=_bash_like(is_destructive=True),
        args={"command": "rm"},
        rule_hint=_HINT,
    )
    assert captured["tag"] == "destructive"


async def test_picker_receives_option_two_label_with_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    picker, captured = _stub_picker(1)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    await asker(
        tool=_bash_like(with_matcher=True),
        args={"command": "npm test"},
        rule_hint=_HINT,
    )
    assert "bash(npm test)" in captured["option_two_label"]
    assert "in this project" in captured["option_two_label"]


async def test_picker_unavailable_writes_journal(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    # Picker raises (e.g. pt can't take over the TTY) → fail-closed
    # deny + journal the unavailability.
    monkeypatch.setattr(
        "aura.cli.permission._pick_choice_interactive",
        _stub_picker_raises(RuntimeError("no tty")),
    )
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(), args={"command": "x"}, rule_hint=_HINT,
    )
    assert resp.choice == "deny"
    events = {e["event"] for e in _events(_journal_capture)}
    assert "permission_prompt_unavailable" in events


async def test_keyboard_interrupt_is_deny(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    # Outer KeyboardInterrupt escapes pt's c-c binding → still deny.
    monkeypatch.setattr(
        "aura.cli.permission._pick_choice_interactive",
        _stub_picker_raises(KeyboardInterrupt()),
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


async def test_audit_line_prints_after_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The scrollback MUST show a one-line audit entry after the widget
    # closes (it erased itself) — otherwise operators scanning the
    # conversation can't tell what happened.
    picker, _ = _stub_picker(1)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, buf = _capture_console()
    asker = make_cli_asker(console=console)
    await asker(
        tool=_bash_like(),
        args={"command": "ls"},
        rule_hint=_HINT,
    )
    out = buf.getvalue()
    assert "bash" in out
    # ``command: `` prefix is stripped through the full asker path
    # (``_preview`` is called on ``{"command": "ls"}`` which returns
    # ``"ls"``), so the audit line reads cleanly.
    assert "ls" in out
    assert "yes" in out


# ---------------------------------------------------------------------------
# Tab-to-amend — feedback threading through the asker
# ---------------------------------------------------------------------------
async def test_feedback_flows_through_accept(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    # Picker returns (1, "needs --verbose"); the asker must pass the
    # note through to AskerResponse.feedback AND record it in the
    # permission_answered journal event.
    picker, _ = _stub_picker(1, feedback="needs --verbose")
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(), args={"command": "npm test"}, rule_hint=_HINT,
    )
    assert resp.choice == "accept"
    assert resp.feedback == "needs --verbose"
    answered = next(
        e for e in _events(_journal_capture) if e["event"] == "permission_answered"
    )
    assert answered["feedback"] == "needs --verbose"


async def test_feedback_flows_through_deny(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    picker, _ = _stub_picker(3, feedback="wrong dir")
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(), args={"command": "rm -rf /"}, rule_hint=_HINT,
    )
    assert resp.choice == "deny"
    assert resp.feedback == "wrong dir"


async def test_feedback_flows_through_always(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    picker, _ = _stub_picker(2, feedback="trusted suite")
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(with_matcher=True),
        args={"command": "npm test"},
        rule_hint=_HINT,
    )
    assert resp.choice == "always"
    assert resp.feedback == "trusted suite"
    assert resp.rule == Rule(tool="bash", content="npm test")


async def test_empty_feedback_is_backwards_compatible(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    # When the user never pressed Tab, feedback is "" — the journal
    # event should NOT carry a spurious feedback field.
    picker, _ = _stub_picker(1)  # feedback defaults to ""
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    resp = await asker(
        tool=_bash_like(), args={"command": "ls"}, rule_hint=_HINT,
    )
    assert resp.feedback == ""
    answered = next(
        e for e in _events(_journal_capture) if e["event"] == "permission_answered"
    )
    assert "feedback" not in answered


def test_audit_line_includes_feedback_when_present() -> None:
    console, buf = _capture_console()
    _render_decision_audit_line(
        console,
        tool=_bash_like(),
        tag="safe",
        preview="ls",
        choice=1,
        feedback="double-check the path",
    )
    out = buf.getvalue()
    assert "yes" in out
    assert "double-check the path" in out


def test_audit_line_omits_feedback_when_empty() -> None:
    # Default empty feedback must not render an empty quoted suffix.
    console, buf = _capture_console()
    _render_decision_audit_line(
        console,
        tool=_bash_like(),
        tag="safe",
        preview="ls",
        choice=1,
    )
    # No dangling `"` pair from a zero-width feedback string.
    assert '""' not in buf.getvalue()


# ---------------------------------------------------------------------------
# _pick_choice_interactive — Tab-to-amend key handling (driven via
# synthesized key-press input so we exercise real KeyBindings rather
# than stubbing the picker).
# ---------------------------------------------------------------------------
async def _drive_picker(
    keys: str, *, is_destructive: bool = False,
) -> tuple[int | None, str]:
    """Run the real ``_pick_choice_interactive`` against a synthetic
    stdin buffer. Uses pt's ``create_pipe_input`` so we can feed raw
    key bytes without a real TTY."""
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from aura.cli import permission as cli_permission_mod

    with create_pipe_input() as inp:
        inp.send_text(keys)
        # pt Application picks up get_app_session's input/output when
        # constructed, so temporarily install a session with our pipe.
        from prompt_toolkit.application import create_app_session

        with create_app_session(input=inp, output=DummyOutput()):
            return await cli_permission_mod._pick_choice_interactive(
                tool=_bash_like(is_destructive=is_destructive),
                preview="ls",
                tag="destructive" if is_destructive else "safe",
                option_two_label="Yes, and don't ask again",
                default_choice=3 if is_destructive else 1,
            )


async def test_picker_tab_then_type_then_enter_commits_with_feedback() -> None:
    # Cursor starts on option 1 (default_choice=1 for safe). Tab →
    # feedback mode; type "hi"; Enter → commit option 1 with feedback.
    choice, feedback = await _drive_picker("\thi\r")
    assert choice == 1
    assert feedback == "hi"


async def test_picker_tab_then_esc_returns_to_option_mode() -> None:
    # Tab → feedback mode; Esc → back to option mode (buffer cleared);
    # Enter → commit default option with empty feedback.
    choice, feedback = await _drive_picker("\thi\x1b\r")
    assert choice == 1
    assert feedback == ""


async def test_picker_no_tab_returns_empty_feedback() -> None:
    # Plain Enter on the default: backwards-compat path.
    choice, feedback = await _drive_picker("\r")
    assert choice == 1
    assert feedback == ""


# ---------------------------------------------------------------------------
# print_bypass_banner — unchanged contract
# ---------------------------------------------------------------------------
def test_print_bypass_banner_writes_warning() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=120)
    print_bypass_banner(console)
    out = buffer.getvalue()
    assert "PERMISSION CHECKS DISABLED" in out
