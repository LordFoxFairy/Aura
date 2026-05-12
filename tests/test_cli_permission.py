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
    _build_explanation,
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
# _build_explanation — Ctrl+E static explanation block
# ---------------------------------------------------------------------------
def _fragment_text(frags: Any) -> str:
    """Flatten a FormattedText list into a single string for content
    assertions (style tuples → just their text component)."""
    return "".join(text for _style, text in frags)


def test_build_explanation_has_required_sections() -> None:
    tool = _bash_like()
    frags = _build_explanation(tool, {"command": "ls"}, "safe")
    text = _fragment_text(frags)
    assert "Arguments:" in text
    assert "Risk:" in text
    assert "What happens" in text
    # Framing chrome so it reads as a collapsible panel.
    assert "┌ Explanation" in text
    assert "└" in text


def test_build_explanation_renders_args_key_value() -> None:
    tool = _bash_like()
    frags = _build_explanation(tool, {"command": "npm test"}, "safe")
    text = _fragment_text(frags)
    assert "command: npm test" in text


def test_build_explanation_truncates_long_arg_values() -> None:
    tool = _bash_like()
    huge = "x" * 500
    frags = _build_explanation(tool, {"command": huge}, "safe")
    text = _fragment_text(frags)
    # No line in the block should carry the full 500-char value.
    assert huge not in text
    assert "…" in text


def test_build_explanation_destructive_risk_line() -> None:
    tool = _bash_like(is_destructive=True)
    frags = _build_explanation(tool, {"command": "rm -rf /"}, "destructive")
    text = _fragment_text(frags)
    assert "modify or delete" in text


def test_build_explanation_read_only_risk_line() -> None:
    tool = _bash_like(is_read_only=True)
    frags = _build_explanation(tool, {"command": "cat x"}, "read-only")
    text = _fragment_text(frags)
    assert "no side effects" in text


def test_build_explanation_safe_risk_line() -> None:
    tool = _bash_like()
    frags = _build_explanation(tool, {"command": "mkdir a"}, "safe")
    text = _fragment_text(frags)
    assert "Low risk" in text


def test_build_explanation_uses_tool_verb_for_known_tool() -> None:
    # ``bash`` is in _TOOL_VERB → the "What happens" line uses "Run
    # shell command" rather than the generic fallback.
    tool = _bash_like()
    frags = _build_explanation(tool, {"command": "ls"}, "safe")
    text = _fragment_text(frags)
    assert "Run shell command" in text


def test_build_explanation_handles_empty_args() -> None:
    # A tool invoked with no args still renders an Arguments section,
    # marked ``(none)`` — not an empty / missing block.
    tool = _bash_like()
    frags = _build_explanation(tool, {}, "safe")
    text = _fragment_text(frags)
    assert "Arguments:" in text
    assert "(none)" in text


# ---------------------------------------------------------------------------
# _pick_choice_interactive — Ctrl+E toggle via driven keystrokes
# ---------------------------------------------------------------------------
async def test_picker_ctrl_e_does_not_crash_and_commits() -> None:
    # Ctrl+E toggles the panel and Enter still commits the default
    # option. We can't introspect the widget's internal state from
    # outside the Application, but the commit-after-toggle path
    # exercises the keybinding end-to-end — if the binding were
    # unregistered or raised, Enter wouldn't reach the c-m handler
    # cleanly.
    choice, feedback = await _drive_picker("\x05\r")  # ctrl+e then Enter
    assert choice == 1
    assert feedback == ""


async def test_picker_ctrl_e_toggle_twice_still_commits() -> None:
    # Two Ctrl+E presses (show, then hide) and Enter commits default —
    # the state must flip both ways without wedging the widget.
    choice, feedback = await _drive_picker("\x05\x05\r")
    assert choice == 1
    assert feedback == ""


async def test_picker_ctrl_e_ignored_in_feedback_mode() -> None:
    # Tab → feedback mode; Ctrl+E is non-printable so the ``<any>``
    # feedback-mode binding filters it out, AND the option-mode filter
    # on the c-e binding prevents a toggle. Result: typed "ab", Ctrl+E
    # is dropped, feedback is "ab", Enter commits default.
    choice, feedback = await _drive_picker("\ta\x05b\r")
    assert choice == 1
    assert feedback == "ab"


# ---------------------------------------------------------------------------
# print_bypass_banner — unchanged contract
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dispatch routing — v0.7.6 specialized widgets. Generic tools (grep,
# read_file, etc.) still hit the generic widget; bash / bash_background
# go to the bash widget; write_file / edit_file go to the write widget.
# These tests pin the dispatch contract by stubbing each specialized
# entrypoint and asserting the right one fired.
# ---------------------------------------------------------------------------
async def test_dispatch_generic_for_non_specialized_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"bash": 0, "write": 0, "generic": 0}

    async def fake_bash(**_kw: Any) -> tuple[int | None, str]:
        calls["bash"] += 1
        return 1, ""

    async def fake_write(**_kw: Any) -> tuple[int | None, str]:
        calls["write"] += 1
        return 1, ""

    async def fake_generic(**_kw: Any) -> tuple[int | None, str]:
        calls["generic"] += 1
        return 1, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "run_bash_permission", fake_bash)
    monkeypatch.setattr(perm_mod, "run_write_permission", fake_write)
    monkeypatch.setattr(perm_mod, "run_generic_permission", fake_generic)

    grep_tool = build_tool(
        name="grep",
        description="Search file contents",
        args_schema=_P,
        func=_noop,
        is_read_only=True,
    )
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    await asker(tool=grep_tool, args={"msg": "x"}, rule_hint=_HINT)
    assert calls == {"bash": 0, "write": 0, "generic": 1}


async def test_dispatch_bash_for_bash_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"bash": 0, "write": 0, "generic": 0}

    async def fake_bash(**_kw: Any) -> tuple[int | None, str]:
        calls["bash"] += 1
        return 1, ""

    async def fake_write(**_kw: Any) -> tuple[int | None, str]:
        calls["write"] += 1
        return 1, ""

    async def fake_generic(**_kw: Any) -> tuple[int | None, str]:
        calls["generic"] += 1
        return 1, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "run_bash_permission", fake_bash)
    monkeypatch.setattr(perm_mod, "run_write_permission", fake_write)
    monkeypatch.setattr(perm_mod, "run_generic_permission", fake_generic)

    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    await asker(tool=_bash_like(), args={"command": "ls"}, rule_hint=_HINT)
    assert calls == {"bash": 1, "write": 0, "generic": 0}


async def test_dispatch_write_for_edit_file_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"bash": 0, "write": 0, "generic": 0}

    async def fake_bash(**_kw: Any) -> tuple[int | None, str]:
        calls["bash"] += 1
        return 1, ""

    async def fake_write(**_kw: Any) -> tuple[int | None, str]:
        calls["write"] += 1
        return 1, ""

    async def fake_generic(**_kw: Any) -> tuple[int | None, str]:
        calls["generic"] += 1
        return 1, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "run_bash_permission", fake_bash)
    monkeypatch.setattr(perm_mod, "run_write_permission", fake_write)
    monkeypatch.setattr(perm_mod, "run_generic_permission", fake_generic)

    edit_tool = build_tool(
        name="edit_file",
        description="Edit a file",
        args_schema=_P,
        func=_noop,
        is_destructive=True,
    )
    console, _ = _capture_console()
    asker = make_cli_asker(console=console)
    await asker(
        tool=edit_tool,
        args={"path": "foo.py", "command": "unused"},
        rule_hint=_HINT,
    )
    assert calls == {"bash": 0, "write": 1, "generic": 0}


def test_print_bypass_banner_writes_warning() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=120)
    print_bypass_banner(console)
    out = buffer.getvalue()
    assert "PERMISSION CHECKS DISABLED" in out


# ---------------------------------------------------------------------------
# Timeout — audit Finding A: unattended / stale sessions must not hang the
# turn forever. A non-response resolves to the existing deny shape and the
# journal records a ``permission_prompt_timeout`` event so the audit trail
# distinguishes "user said no" from "user never answered".
# ---------------------------------------------------------------------------
async def test_timeout_resolves_to_deny(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(
        "aura.cli.permission._pick_choice_interactive",
        _stub_picker_raises(TimeoutError()),
    )
    console, _ = _capture_console()
    asker = make_cli_asker(console=console, timeout=0.1)
    resp = await asker(
        tool=_bash_like(), args={"command": "npm test"}, rule_hint=_HINT,
    )
    assert isinstance(resp, AskerResponse)
    assert resp.choice == "deny"


async def test_timeout_writes_journal_entry(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:

    monkeypatch.setattr(
        "aura.cli.permission._pick_choice_interactive",
        _stub_picker_raises(TimeoutError()),
    )
    console, _ = _capture_console()
    asker = make_cli_asker(console=console, timeout=0.1)
    await asker(tool=_bash_like(), args={"command": "npm test"}, rule_hint=_HINT)
    events = _events(_journal_capture)
    timeout_events = [e for e in events if e["event"] == "permission_prompt_timeout"]
    assert len(timeout_events) == 1
    assert timeout_events[0]["tool"] == "bash"
    assert timeout_events[0]["timeout_sec"] == 0.1
    # A permission_answered=deny with reason=timeout must also fire so
    # downstream audit consumers see a single "decision" event stream
    # rather than having to special-case the timeout surface.
    answered = [e for e in events if e["event"] == "permission_answered"]
    assert len(answered) == 1
    assert answered[0]["choice"] == "deny"
    assert answered[0]["reason"] == "timeout"


async def test_timeout_none_preserves_legacy_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # timeout=None means "wait forever" — the picker runs normally and
    # its preset answer propagates through. No wait_for wrapping fires.
    picker, captured = _stub_picker(1)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console, timeout=None)
    resp = await asker(
        tool=_bash_like(), args={"command": "ls"}, rule_hint=_HINT,
    )
    assert resp.choice == "accept"
    assert captured.get("timeout") is None


async def test_timeout_plumbed_through_to_picker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    picker, captured = _stub_picker(1)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console, timeout=5.0)
    await asker(tool=_bash_like(), args={"command": "ls"}, rule_hint=_HINT)
    assert captured["timeout"] == 5.0


async def test_fast_response_not_affected_by_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # User responds quickly; timeout=5.0 is plenty. The picker returns
    # its preset value and no TimeoutError fires — the asker resolves
    # normally.
    picker, _ = _stub_picker(1)
    monkeypatch.setattr("aura.cli.permission._pick_choice_interactive", picker)
    console, _ = _capture_console()
    asker = make_cli_asker(console=console, timeout=5.0)
    resp = await asker(
        tool=_bash_like(), args={"command": "ls"}, rule_hint=_HINT,
    )
    assert resp.choice == "accept"


async def test_real_timeout_via_wait_for() -> None:
    # End-to-end: _pick_choice_interactive (via the dispatched widget)
    # wraps pt.Application.run_async with asyncio.wait_for(timeout=...).
    # Stub run_async with a coroutine that sleeps longer than the
    # timeout; confirm TimeoutError bubbles.
    #
    # Application now lives in ``aura.cli.permission_generic`` (where
    # the shared pt driver lives); patch the symbol there. Use a
    # non-specialized tool so the generic widget path runs.
    import asyncio

    from aura.cli import permission as permission_module
    from aura.cli import permission_generic

    class _FakeApp:
        is_running = True

        async def run_async(self) -> None:
            await asyncio.sleep(10.0)

        def exit(self) -> None:
            self.is_running = False

    fake_app = _FakeApp()

    class _FakeApplicationCls:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def __new__(cls, *_a: Any, **_kw: Any) -> Any:
            return fake_app

    import pytest as _pytest

    generic_tool = build_tool(
        name="grep",
        description="search",
        args_schema=_P,
        func=_noop,
        is_read_only=True,
    )
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(permission_generic, "Application", _FakeApplicationCls)
        with _pytest.raises(asyncio.TimeoutError):
            await permission_module._pick_choice_interactive(
                tool=generic_tool,
                preview="ls",
                tag="safe",
                option_two_label="always",
                default_choice=1,
                args={"command": "ls"},
                timeout=0.05,
            )
    # pt Application.exit() was called to unwind its render loop.
    assert fake_app.is_running is False


# ---------------------------------------------------------------------------
# Phase 5 Task 4 — v2 asker (AskerPrompt → new AskerResponse).
#
# The v2 entry point consumes :class:`AskerPrompt` (display strings +
# request_id) and returns the four-state new :class:`AskerResponse`.
# Same widget driver as legacy; UI behavior matches.
# ---------------------------------------------------------------------------
from aura.cli.permission import (  # noqa: E402
    legacy_asker_from_v2,
    make_cli_asker_v2,
)
from aura.schemas.permissions import AskerPrompt  # noqa: E402
from aura.schemas.permissions import AskerResponse as AskerResponseV2  # noqa: E402


def _make_v2_prompt(
    *,
    tool: str = "bash",
    args_preview: str = "ls",
    rule_hint: str = "",
    is_destructive: bool = False,
    request_id: str = "req-1",
) -> AskerPrompt:
    return AskerPrompt(
        tool=tool,
        args_preview=args_preview,
        rule_hint=rule_hint,
        is_destructive=is_destructive,
        request_id=request_id,
    )


async def test_v2_yes_returns_yes(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    # User picks option 1 → new "yes". The v2 asker calls
    # ``_run_widget`` directly (no per-tool dispatch) so we patch that.
    async def fake_run_widget(**_kw: Any) -> tuple[int | None, str]:
        return 1, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "_run_widget", fake_run_widget)
    asker = make_cli_asker_v2()
    resp = await asker(_make_v2_prompt())
    assert isinstance(resp, AskerResponseV2)
    assert resp.choice == "yes"
    assert resp.request_id == "req-1"
    answered = [
        e for e in _events(_journal_capture) if e["event"] == "permission_answered"
    ]
    assert answered and answered[0]["choice"] == "yes"
    assert answered[0]["request_id"] == "req-1"


async def test_v2_yes_always_returns_yes_always(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_widget(**_kw: Any) -> tuple[int | None, str]:
        return 2, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "_run_widget", fake_run_widget)
    asker = make_cli_asker_v2()
    resp = await asker(_make_v2_prompt(rule_hint="bash(npm test)"))
    assert resp.choice == "yes-always"


async def test_v2_no_returns_no(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_widget(**_kw: Any) -> tuple[int | None, str]:
        return 3, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "_run_widget", fake_run_widget)
    asker = make_cli_asker_v2()
    resp = await asker(_make_v2_prompt(is_destructive=True))
    assert resp.choice == "no"


async def test_v2_cancel_resolves_to_no(monkeypatch: pytest.MonkeyPatch) -> None:
    # Picker returns None (Esc / Ctrl+C) → fail-safe deny ("no").
    async def fake_run_widget(**_kw: Any) -> tuple[int | None, str]:
        return None, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "_run_widget", fake_run_widget)
    asker = make_cli_asker_v2()
    resp = await asker(_make_v2_prompt())
    assert resp.choice == "no"


async def test_v2_timeout_resolves_to_no(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    async def fake_run_widget(**_kw: Any) -> tuple[int | None, str]:
        raise TimeoutError

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "_run_widget", fake_run_widget)
    asker = make_cli_asker_v2(timeout=0.1)
    resp = await asker(_make_v2_prompt())
    assert resp.choice == "no"
    events = {e["event"] for e in _events(_journal_capture)}
    assert "permission_prompt_timeout" in events


async def test_v2_default_choice_is_3_for_destructive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_widget(**kw: Any) -> tuple[int | None, str]:
        captured.update(kw)
        return 3, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "_run_widget", fake_run_widget)
    asker = make_cli_asker_v2()
    await asker(_make_v2_prompt(is_destructive=True))
    assert captured["default_choice"] == 3


async def test_v2_default_choice_is_1_for_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_widget(**kw: Any) -> tuple[int | None, str]:
        captured.update(kw)
        return 1, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "_run_widget", fake_run_widget)
    asker = make_cli_asker_v2()
    await asker(_make_v2_prompt(is_destructive=False))
    assert captured["default_choice"] == 1


async def test_v2_request_id_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_widget(**_kw: Any) -> tuple[int | None, str]:
        return 1, ""

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "_run_widget", fake_run_widget)
    asker = make_cli_asker_v2()
    resp = await asker(_make_v2_prompt(request_id="abc-123"))
    assert resp.request_id == "abc-123"


async def test_v2_widget_failure_resolves_to_no(
    monkeypatch: pytest.MonkeyPatch, _journal_capture: Path,
) -> None:
    async def fake_run_widget(**_kw: Any) -> tuple[int | None, str]:
        raise RuntimeError("no tty")

    from aura.cli import permission as perm_mod

    monkeypatch.setattr(perm_mod, "_run_widget", fake_run_widget)
    asker = make_cli_asker_v2()
    resp = await asker(_make_v2_prompt())
    assert resp.choice == "no"
    events = {e["event"] for e in _events(_journal_capture)}
    assert "permission_prompt_unavailable" in events


# ---------------------------------------------------------------------------
# legacy_asker_from_v2 — adapter that exposes a v2 asker as the legacy
# PermissionAsker Protocol so ``make_permission_hook`` keeps working.
# ---------------------------------------------------------------------------
async def test_legacy_adapter_yes_maps_to_accept() -> None:
    async def fake_v2(_prompt: AskerPrompt) -> AskerResponseV2:
        return AskerResponseV2(choice="yes", request_id=_prompt.request_id)

    legacy = legacy_asker_from_v2(fake_v2)
    resp = await legacy(
        tool=_bash_like(), args={"command": "ls"}, rule_hint=_HINT,
    )
    assert isinstance(resp, AskerResponse)
    assert resp.choice == "accept"
    assert resp.rule is None


async def test_legacy_adapter_yes_always_maps_to_always_with_rule() -> None:
    async def fake_v2(prompt: AskerPrompt) -> AskerResponseV2:
        return AskerResponseV2(choice="yes-always", request_id=prompt.request_id)

    legacy = legacy_asker_from_v2(fake_v2)
    resp = await legacy(
        tool=_bash_like(with_matcher=True),
        args={"command": "npm test"},
        rule_hint=_HINT,
    )
    assert resp.choice == "always"
    assert resp.scope == "project"
    assert resp.rule == Rule(tool="bash", content="npm test")


async def test_legacy_adapter_no_maps_to_deny() -> None:
    async def fake_v2(prompt: AskerPrompt) -> AskerResponseV2:
        return AskerResponseV2(choice="no", request_id=prompt.request_id)

    legacy = legacy_asker_from_v2(fake_v2)
    resp = await legacy(
        tool=_bash_like(), args={"command": "rm"}, rule_hint=_HINT,
    )
    assert resp.choice == "deny"


async def test_legacy_adapter_no_always_maps_to_deny() -> None:
    # ``no-always`` has no legacy equivalent (legacy AskerResponse can't
    # represent "deny rule"); the adapter degrades to a one-shot deny.
    # Task 8's PermissionGate is responsible for installing the deny
    # rule out-of-band when the new shape becomes the wire format.
    async def fake_v2(prompt: AskerPrompt) -> AskerResponseV2:
        return AskerResponseV2(choice="no-always", request_id=prompt.request_id)

    legacy = legacy_asker_from_v2(fake_v2)
    resp = await legacy(
        tool=_bash_like(), args={"command": "rm"}, rule_hint=_HINT,
    )
    assert resp.choice == "deny"


async def test_legacy_adapter_builds_prompt_with_unique_request_id() -> None:
    # Each adapter invocation generates a fresh request_id (uuid4) since
    # legacy callers don't carry one — IPC correlation only matters for
    # the desktop asker.
    seen_ids: list[str] = []

    async def fake_v2(prompt: AskerPrompt) -> AskerResponseV2:
        seen_ids.append(prompt.request_id)
        return AskerResponseV2(choice="yes", request_id=prompt.request_id)

    legacy = legacy_asker_from_v2(fake_v2)
    await legacy(tool=_bash_like(), args={"command": "ls"}, rule_hint=_HINT)
    await legacy(tool=_bash_like(), args={"command": "ls"}, rule_hint=_HINT)
    assert len(seen_ids) == 2
    assert seen_ids[0] != seen_ids[1]
    assert all(s for s in seen_ids)  # non-empty


async def test_legacy_adapter_passes_destructive_flag_through() -> None:
    captured: list[AskerPrompt] = []

    async def fake_v2(prompt: AskerPrompt) -> AskerResponseV2:
        captured.append(prompt)
        return AskerResponseV2(choice="no", request_id=prompt.request_id)

    legacy = legacy_asker_from_v2(fake_v2)
    await legacy(
        tool=_bash_like(is_destructive=True),
        args={"command": "rm"},
        rule_hint=_HINT,
    )
    assert captured[0].is_destructive is True


async def test_legacy_adapter_passes_rule_hint_string() -> None:
    captured: list[AskerPrompt] = []

    async def fake_v2(prompt: AskerPrompt) -> AskerResponseV2:
        captured.append(prompt)
        return AskerResponseV2(choice="yes", request_id=prompt.request_id)

    legacy = legacy_asker_from_v2(fake_v2)
    await legacy(
        tool=_bash_like(with_matcher=True),
        args={"command": "npm test"},
        rule_hint=_HINT,
    )
    # Rule hint string mirrors the option-two rule's wire format.
    assert captured[0].rule_hint == "bash(npm test)"
