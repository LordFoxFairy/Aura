"""Tests for aura.cli.repl.run_repl_async."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console

from aura.cli.repl import run_repl_async
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _agent(tmp_path: Path, turns: list[FakeTurn] | None = None) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=turns or []),
        storage=SessionStorage(tmp_path / "db"),
    )


def _capture_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=200, highlight=False), buf


class _ScriptedInput:
    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    async def __call__(self, prompt: str) -> str:
        if not self._lines:
            raise EOFError()
        return self._lines.pop(0)


async def test_exit_command_returns(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    console, buf = _capture_console()

    await run_repl_async(
        agent, input_fn=_ScriptedInput(["/exit"]), console=console,
    )
    await agent.aclose()


async def test_help_then_exit(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    console, buf = _capture_console()

    await run_repl_async(
        agent, input_fn=_ScriptedInput(["/help", "/exit"]), console=console,
    )
    assert "/exit" in buf.getvalue()
    await agent.aclose()


async def test_non_slash_line_forwards_to_agent(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        turns=[FakeTurn(message=AIMessage(content="hello back"))],
    )
    console, buf = _capture_console()

    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["hello agent", "/exit"]),
        console=console,
    )
    assert "hello back" in buf.getvalue()
    await agent.aclose()


async def test_eof_exits_cleanly(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    console, buf = _capture_console()

    await run_repl_async(
        agent, input_fn=_ScriptedInput([]), console=console,
    )
    await agent.aclose()


async def test_empty_input_does_not_reach_agent(tmp_path: Path) -> None:
    # Providers 400 on empty user turns — REPL must never round-trip
    # them to the model. Whitespace-only lines are reprompted silently.
    # Scripted inputs: empty string, whitespace, then an exit.
    agent = _agent(tmp_path, turns=[FakeTurn(message=AIMessage(content="should not fire"))])
    console, buf = _capture_console()

    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["", "   ", "\t\n", "/exit"]),
        console=console,
    )
    # Agent's single queued turn was never consumed — the empty inputs
    # skipped the astream path entirely.
    assert "should not fire" not in buf.getvalue()
    await agent.aclose()


async def test_verbose_prints_turn_summary(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        turns=[FakeTurn(message=AIMessage(content="hi"))],
    )
    console, buf = _capture_console()

    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["hello", "/exit"]),
        console=console,
        verbose=True,
    )

    out = buf.getvalue()
    assert "turn 1" in out
    assert "tokens" in out
    await agent.aclose()


async def test_non_verbose_does_not_print_summary(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        turns=[FakeTurn(message=AIMessage(content="hi"))],
    )
    console, buf = _capture_console()

    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["hello", "/exit"]),
        console=console,
        verbose=False,
    )

    out = buf.getvalue()
    assert "turn 1" not in out
    await agent.aclose()


async def test_bypass_mode_prompt_carries_bypass_marker(tmp_path: Path) -> None:
    # Instead of testing that the banner scrolls off (out of REPL scope),
    # prove the bypass prompt string contains the visible marker so users
    # see it every line.
    agent = _agent(tmp_path)
    console, _buf = _capture_console()

    # Custom input_fn that captures the prompt string seen by the REPL.
    seen_prompts: list[str] = []

    async def _capture_prompt(prompt: str) -> str:
        seen_prompts.append(prompt)
        raise EOFError  # exit immediately after first prompt

    await run_repl_async(
        agent, input_fn=_capture_prompt, console=console, bypass=True,
    )
    assert seen_prompts
    assert "bypass" in seen_prompts[0]
    await agent.aclose()


async def test_non_bypass_mode_uses_plain_prompt(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    console, _buf = _capture_console()
    seen_prompts: list[str] = []

    async def _capture_prompt(prompt: str) -> str:
        seen_prompts.append(prompt)
        raise EOFError

    await run_repl_async(agent, input_fn=_capture_prompt, console=console)
    assert seen_prompts
    assert "bypass" not in seen_prompts[0]
    await agent.aclose()


async def test_welcome_banner_shows_core_info_compactly(tmp_path: Path) -> None:
    # Single compact cyan Panel (v0.8.0 shape restored after operator
    # feedback). Branding line + keybinding hints + model + cwd + tip, all
    # inside ONE Panel. ``expand=False`` so the panel is content-sized,
    # not terminal-sized. Cyan border drawn with box-drawing glyphs.
    from aura import __version__
    from aura.cli.repl import _STARTUP_TIPS

    agent = _agent(tmp_path)
    console, buf = _capture_console()

    await run_repl_async(
        agent, input_fn=_ScriptedInput(["/exit"]), console=console,
    )
    out = buf.getvalue()

    # Branding line — mark + wordmark + version all in one place.
    assert "✱ Aura" in out
    assert f"v{__version__}" in out

    # Info lines inside the same Panel.
    assert "model:" in out
    assert "cwd:" in out
    assert "tip:" in out

    # Keybinding hints still on the branding line.
    assert "/help" in out
    assert "Ctrl+D" in out
    assert "exit" in out

    # Cyan Panel border uses Unicode box-drawing glyphs — pick any of the
    # four corner characters; if the Panel vanishes all four disappear.
    assert any(glyph in out for glyph in ("╭", "╮", "╰", "╯"))

    # Tip line matches one of the curated options (exact substring).
    assert any(tip in out for tip in _STARTUP_TIPS)

    await agent.aclose()


def test_welcome_banner_spinner_frames_are_non_empty_and_include_settle_glyph(
    tmp_path: Path,
) -> None:
    # U2: the animated welcome banner rotates through the same glyph
    # family the in-turn ThinkingSpinner uses, then settles on ``✱``.
    # Guard the constants so the animation is never silently de-armed
    # (e.g. an empty tuple would make the Live loop a no-op).
    from aura.cli.repl import (
        _BANNER_ANIMATION_SECONDS,
        _BANNER_SETTLE_GLYPH,
        _BANNER_SPINNER_FRAMES,
    )
    assert _BANNER_SPINNER_FRAMES
    assert _BANNER_SETTLE_GLYPH == "✱"
    # Frame set must overlap with the ThinkingSpinner's glyphs (visual
    # continuity between startup and in-flight). At least one common
    # character is required — catches accidental typo drift.
    from aura.cli.spinner import _GLYPHS as _THINKING_GLYPHS
    assert set(_BANNER_SPINNER_FRAMES) & set(_THINKING_GLYPHS)
    # Animation duration must be user-perceptible (>=0.4s) but not
    # obstructive (<=3s).
    assert 0.4 <= _BANNER_ANIMATION_SECONDS <= 3.0


def test_welcome_banner_static_path_renders_settled_glyph_for_non_tty(
    tmp_path: Path,
) -> None:
    # StringIO-backed Console reports ``is_terminal=False``; the welcome
    # helper MUST take the short-circuit path and print the settled
    # banner WITHOUT running a Live animation. Dogfood check: the
    # captured output carries the settled ``✱`` glyph AND none of the
    # intermediate spinner frames (the animation never ran).
    from aura.cli.repl import _BANNER_SPINNER_FRAMES, _print_welcome

    agent = _agent(tmp_path)
    console, buf = _capture_console()
    assert not console.is_terminal  # precondition for the short-circuit
    _print_welcome(agent, console)
    out = buf.getvalue()
    assert "✱ Aura" in out
    # None of the transient animation frames should leak into scrollback
    # (each frame would have rendered ``<frame> Aura``). The settle ``✱``
    # is in the list too — exclude it from the check.
    for frame in _BANNER_SPINNER_FRAMES:
        if frame == "✱":
            continue
        assert f"{frame} Aura" not in out, (
            f"intermediate frame {frame!r} leaked into non-TTY banner"
        )
    agent.close()


def test_welcome_banner_animated_path_runs_in_real_tty(
    tmp_path: Path,
) -> None:
    # U2 dogfood: drive a REAL pty + Aura subprocess and confirm the
    # welcome banner scrolls with an animated leading glyph ending in
    # ``✱``. This is the ground truth that a StringIO test can't give.
    # Uses Python's ``pty`` module — no external deps.
    import os
    import pty
    import select
    import sys
    import time as _time

    # Spawn a tiny driver that imports the helper and prints the banner
    # with a REAL Rich Console attached to the pty (is_terminal=True).
    # The driver exits when the banner finishes so the test doesn't hang.
    driver = (
        "from pathlib import Path\n"
        "from rich.console import Console\n"
        "from aura.cli.repl import _print_welcome\n"
        "from aura.core.agent import Agent\n"
        "from aura.config.schema import AuraConfig\n"
        "from aura.core.persistence.storage import SessionStorage\n"
        "from tests.conftest import FakeChatModel\n"
        "cfg = AuraConfig.model_validate({\n"
        "    'providers': [{'name': 'openai', 'protocol': 'openai'}],\n"
        "    'router': {'default': 'openai:gpt-4o-mini'},\n"
        "    'tools': {'enabled': []},\n"
        "})\n"
        "import tempfile\n"
        "d = Path(tempfile.mkdtemp())\n"
        "agent = Agent(config=cfg, model=FakeChatModel(turns=[]),\n"
        "              storage=SessionStorage(d / 'db'))\n"
        "_print_welcome(agent, Console(force_terminal=True))\n"
        "agent.close()\n"
    )
    pid, fd = pty.fork()
    if pid == 0:
        # Child process — exec Python with the driver script.
        os.execvp(
            sys.executable, [sys.executable, "-c", driver],
        )
    captured = bytearray()
    deadline = _time.monotonic() + 10.0
    try:
        while _time.monotonic() < deadline:
            rlist, _, _ = select.select([fd], [], [], 0.5)
            if rlist:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                captured.extend(chunk)
            # Non-blocking wait — exit the read loop as soon as the
            # child is done AND the pipe has drained.
            done_pid, _status = os.waitpid(pid, os.WNOHANG)
            if done_pid == pid:
                # Drain any remaining bytes.
                while True:
                    rlist, _, _ = select.select([fd], [], [], 0.1)
                    if not rlist:
                        break
                    try:
                        chunk = os.read(fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    captured.extend(chunk)
                break
    finally:
        import contextlib
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError, ChildProcessError):
            os.waitpid(pid, 0)

    text = captured.decode("utf-8", errors="replace")
    # The final settle frame MUST be ``✱ Aura`` — proves the animation
    # landed on the stable glyph.
    assert "✱ Aura" in text, f"settle glyph missing from pty output: {text!r}"
    # And at least one INTERMEDIATE frame leaked — proves the animation
    # actually ran (StringIO path would lack these entirely).
    intermediate_seen = any(
        f"{f} Aura" in text for f in ("✻", "✶", "✳", "✢")
    )
    assert intermediate_seen, (
        "no intermediate banner spinner frame observed — "
        "animation never ran in pty path: "
        f"{text!r}"
    )


async def test_welcome_banner_renders_even_with_odd_version(
    tmp_path: Path,
) -> None:
    # Banner must not crash when ``__version__`` drifts (e.g. a dev build
    # sets it to "0.0.0+dev"). Poke the module dict (not attribute access)
    # so mypy doesn't flag ``__version__`` as non-exported and ruff doesn't
    # flag a constant setattr.
    from aura.cli import repl as repl_mod

    agent = _agent(tmp_path)
    console, buf = _capture_console()

    mod_ns = vars(repl_mod)
    original = mod_ns["__version__"]
    mod_ns["__version__"] = "9.9.9+dev"
    try:
        await run_repl_async(
            agent, input_fn=_ScriptedInput(["/exit"]), console=console,
        )
    finally:
        mod_ns["__version__"] = original

    out = buf.getvalue()
    assert "v9.9.9+dev" in out
    assert "✱ Aura" in out
    await agent.aclose()


def test_prompt_session_no_toolbar_when_agent_not_passed(tmp_path: Path) -> None:
    # Bare-bones construction (tests only exercising history/completion)
    # still works: without an agent, there's nothing to render in the bar,
    # so it's elided.
    from aura.cli.repl import _build_prompt_session
    from aura.core.commands import CommandRegistry

    session = _build_prompt_session(CommandRegistry())
    assert session.bottom_toolbar is None


def test_prompt_session_bottom_toolbar_shows_context_when_agent_passed(
    tmp_path: Path,
) -> None:
    # Real REPL wiring: pass an Agent, get a live toolbar callable. The
    # toolbar reads _token_stats fresh on each render; exercise it with
    # seeded stats and assert the rendered HTML carries model + bar +
    # pinned cached + cwd.
    from aura.cli.repl import _build_prompt_session
    from aura.core.agent import Agent
    from aura.core.commands import CommandRegistry
    from tests.conftest import FakeChatModel
    from tests.test_agent import _minimal_config, _storage

    agent = Agent(
        config=_minimal_config(enabled=[]),
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
    )
    agent.state.custom["_token_stats"] = {
        "last_input_tokens": 5400,
        "last_cache_read_tokens": 34_000,
    }

    session = _build_prompt_session(CommandRegistry(), agent=agent)
    assert callable(session.bottom_toolbar)
    html = session.bottom_toolbar()
    # HTML carries the pieces as children of ansigray / ansi-color tags.
    text = str(html)
    assert "5.4k/" in text            # live tokens over window
    assert "34.0k cached" in text or "34k cached" in text
    agent.close()


async def test_alt_enter_inserts_newline_in_prompt_buffer(
    tmp_path: Path,
) -> None:
    # Real-pt round-trip: drive a PromptSession built with the shared
    # KeyBindings via create_pipe_input. Send Alt+Enter (ESC + CR) then
    # Enter (CR). The Alt+Enter binding must insert a literal newline;
    # only the final CR submits. Result should carry "\n" between the
    # two halves of the typed text.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from aura.cli.repl import _build_mode_key_bindings

    agent = _agent(tmp_path)
    console, _buf = _capture_console()
    kb = _build_mode_key_bindings(agent, console)
    with create_pipe_input() as inp:
        # "line1" + ESC + CR (alt+enter) + "line2" + CR (enter/submit).
        inp.send_text("line1\x1b\rline2\r")
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        result = await session.prompt_async("> ")
    assert result == "line1\nline2"
    await agent.aclose()


async def test_ctrl_j_inserts_newline_in_prompt_buffer(
    tmp_path: Path,
) -> None:
    # Ctrl+J (0x0a, "\n") is the universal fallback for terminals that
    # remap Shift+Enter. Same contract as Alt+Enter: insert a literal
    # newline, don't submit.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from aura.cli.repl import _build_mode_key_bindings

    agent = _agent(tmp_path)
    console, _buf = _capture_console()
    kb = _build_mode_key_bindings(agent, console)
    with create_pipe_input() as inp:
        inp.send_text("a\nb\r")
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        result = await session.prompt_async("> ")
    assert result == "a\nb"
    await agent.aclose()


async def test_multiline_slash_command_uses_first_line_only(
    tmp_path: Path,
) -> None:
    # If a user pastes a multi-line block whose first line is a slash
    # command (e.g. ``/exit`` followed by stray paste lines), the REPL
    # must still dispatch the command cleanly. Policy: first line is
    # the command, remaining lines are ignored.
    agent = _agent(tmp_path)
    console, _buf = _capture_console()

    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["/exit\nleftover paste content"]),
        console=console,
    )
    # Reached the exit path without hitting EOFError (i.e. /exit handled).
    await agent.aclose()


async def test_multiline_non_slash_input_reaches_agent_intact(
    tmp_path: Path,
) -> None:
    # Multi-line natural-language / pasted-code prompts must flow through
    # to the agent with newlines preserved. Subclass FakeChatModel to
    # snoop the HumanMessage seen by ``_agenerate``.
    from langchain_core.messages import BaseMessage, HumanMessage

    captured: list[str] = []

    class _CaptureModel(FakeChatModel):
        async def _agenerate(  # type: ignore[override]
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: object | None = None,
            **kwargs: object,
        ) -> object:
            for msg in messages:
                if isinstance(msg, HumanMessage) and isinstance(
                    msg.content, str,
                ):
                    captured.append(msg.content)
            return await super()._agenerate(
                messages, stop, run_manager,  # type: ignore[arg-type]
                **kwargs,
            )

    agent = Agent(
        config=AuraConfig.model_validate({
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }),
        model=_CaptureModel(
            turns=[FakeTurn(message=AIMessage(content="ok"))],  # type: ignore[call-arg]
        ),
        storage=SessionStorage(tmp_path / "db"),
    )
    console, _buf = _capture_console()
    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["first line\nsecond line", "/exit"]),
        console=console,
    )
    assert any("first line\nsecond line" in c for c in captured)
    await agent.aclose()


async def test_single_line_slash_command_unchanged(tmp_path: Path) -> None:
    # Regression guard: the multi-line dispatch split must not break
    # the overwhelmingly common single-line slash-command path.
    agent = _agent(tmp_path)
    console, buf = _capture_console()
    await run_repl_async(
        agent, input_fn=_ScriptedInput(["/help", "/exit"]), console=console,
    )
    assert "/exit" in buf.getvalue()
    await agent.aclose()


def test_post_turn_status_is_slim_done_marker(tmp_path: Path) -> None:
    # Bug fix — the post-turn checkpoint used to duplicate the whole
    # bottom_toolbar (model / ctx bar / pinned / mode / cwd), which
    # then collided with pt's live bottom_toolbar above the prompt.
    # Shape is now: a single dim line with "done" + elapsed seconds.
    # Nothing from the live toolbar should show up inline.
    from aura.cli.repl import _print_post_turn_status

    agent = _agent(tmp_path)
    # Seed stats that WOULD have shown up in the old render — they must
    # NOT appear in the new slim line.
    agent.state.custom["_token_stats"] = {
        "last_input_tokens": 9900,
        "last_cache_read_tokens": 2600,
    }
    console, buf = _capture_console()

    _print_post_turn_status(agent, console, last_turn_seconds=21.4)

    out = buf.getvalue()
    # New shape: "done" marker + elapsed seconds.
    assert "done" in out
    assert "21.4s" in out
    # Old content that now lives ONLY in bottom_toolbar must not appear.
    assert "model:" not in out
    assert "pinned" not in out
    assert "cached" not in out
    assert "cwd" not in out
    # Single-line output — no duplicated status bar (one trailing newline
    # from rich is expected; no embedded newlines inside the content).
    assert out.count("\n") == 1
    agent.close()


def test_post_turn_status_elides_duration_when_zero(tmp_path: Path) -> None:
    # Defensive: zero elapsed means "we don't have a measurement yet".
    # Skip the duration tail instead of printing "0.0s" noise.
    from aura.cli.repl import _print_post_turn_status

    agent = _agent(tmp_path)
    console, buf = _capture_console()

    _print_post_turn_status(agent, console, last_turn_seconds=0.0)

    out = buf.getvalue()
    assert "done" in out
    assert "s" not in out.replace("done", "")  # no seconds suffix
    agent.close()


def test_post_turn_status_uses_integer_seconds_at_or_above_60s(
    tmp_path: Path,
) -> None:
    # Consistency with the live bottom_toolbar: sub-minute shows decimal,
    # ≥60s drops the decimal (visual noise at that scale).
    from aura.cli.repl import _print_post_turn_status

    agent = _agent(tmp_path)
    console, buf = _capture_console()

    _print_post_turn_status(agent, console, last_turn_seconds=125.7)

    out = buf.getvalue()
    assert "125s" in out
    assert "125.7" not in out
    agent.close()


async def test_shift_tab_cycles_mode_silently_no_scrollback_spam(
    tmp_path: Path,
) -> None:
    # Regression — rapid shift+tab presses used to spam the scrollback
    # with dozens of "mode: X (press shift+tab to cycle …)" lines
    # (operator report + tmp/img_2.png). Fix: bindings only flip state
    # + call ``event.app.invalidate()`` so the bottom_toolbar redraws
    # with the new mode. Zero scrollback output.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from aura.cli.repl import _build_mode_key_bindings

    agent = _agent(tmp_path)
    console, buf = _capture_console()
    kb = _build_mode_key_bindings(agent, console)
    assert agent.mode == "default"

    with create_pipe_input() as inp:
        # Shift+Tab at xterm-compatible terminals emits ESC [ Z.
        inp.send_text("\x1b[Zq\r")
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        await session.prompt_async("> ")

    # State changed: default → accept_edits.
    assert agent.mode == "accept_edits"
    # Nothing printed — bottom_toolbar is the sole feedback surface.
    out = buf.getvalue()
    assert "mode:" not in out
    assert "shift+tab to cycle" not in out
    await agent.aclose()


async def test_ctrl_c_with_text_clears_buffer_and_does_not_exit(
    tmp_path: Path,
) -> None:
    # U1: claude-code parity (src/hooks/useExitOnCtrlCD.ts + PromptInput's
    # onBufferReset). When the buffer has text, Ctrl+C discards it and
    # leaves the session alive — never an exit on first press with content.
    # Typed "abc" then Ctrl+C; the session stays open until we send "\r".
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from aura.cli.repl import _build_mode_key_bindings

    agent = _agent(tmp_path)
    console, _buf = _capture_console()
    kb = _build_mode_key_bindings(agent, console)
    with create_pipe_input() as inp:
        # "abc" + Ctrl+C (clears) + "q" + CR (submits the "q").
        inp.send_text("abc\x03q\r")
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        result = await session.prompt_async("> ")
    # Ctrl+C cleared the "abc" — result only carries post-Ctrl-C typing.
    assert result == "q"
    await agent.aclose()


async def test_ctrl_c_empty_buffer_single_press_does_not_exit(
    tmp_path: Path,
) -> None:
    # U1: empty-buffer first Ctrl+C must NOT exit — it arms a "press
    # again to exit" state. Mirror of useExitOnCtrlCD's pending state.
    # Send Ctrl+C then "hi\r"; prompt should return "hi", not raise
    # KeyboardInterrupt. (Exit on double-press is tested separately.)
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from aura.cli.repl import _build_mode_key_bindings

    agent = _agent(tmp_path)
    console, _buf = _capture_console()
    kb = _build_mode_key_bindings(agent, console)
    with create_pipe_input() as inp:
        # Bare Ctrl+C, then "hi" + CR (wait long enough the window closes
        # is NOT needed — the next typing keystroke happens immediately
        # and doesn't re-trigger the double-press path).
        inp.send_text("\x03hi\r")
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        result = await session.prompt_async("> ")
    assert result == "hi"
    await agent.aclose()


async def test_ctrl_c_double_press_empty_buffer_raises_keyboard_interrupt(
    tmp_path: Path,
) -> None:
    # U1: TWO bare Ctrl+C within the window raises KeyboardInterrupt,
    # which the outer REPL loop treats as the exit signal. Parity with
    # claude-code's useDoublePress.
    import pytest
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from aura.cli.repl import _build_mode_key_bindings

    agent = _agent(tmp_path)
    console, _buf = _capture_console()
    kb = _build_mode_key_bindings(agent, console)
    with create_pipe_input() as inp:
        inp.send_text("\x03\x03")  # Ctrl+C, Ctrl+C (back-to-back).
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        with pytest.raises(KeyboardInterrupt):
            await session.prompt_async("> ")
    await agent.aclose()


async def test_ctrl_c_second_press_outside_window_does_not_exit(
    tmp_path: Path,
) -> None:
    # U1: if the second Ctrl+C arrives AFTER the 800ms window, it
    # restarts the arm — does not exit. This guards against a
    # forgotten-but-never-cleared state.
    import time as _time

    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from aura.cli.repl import _build_mode_key_bindings, _CtrlCState

    agent = _agent(tmp_path)
    console, _buf = _capture_console()
    state = _CtrlCState()
    kb = _build_mode_key_bindings(agent, console, state)
    # Prime: simulate a Ctrl+C that happened LONG ago (outside window).
    state.last_press_at = _time.monotonic() - 5.0
    assert not state.hint_active(_time.monotonic())

    with create_pipe_input() as inp:
        # Single Ctrl+C — should arm, not exit (previous press stale).
        inp.send_text("\x03ok\r")
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        result = await session.prompt_async("> ")
    assert result == "ok"
    await agent.aclose()


async def test_ctrl_c_text_present_does_not_arm_double_press(
    tmp_path: Path,
) -> None:
    # U1: Ctrl+C on a non-empty buffer clears the text but MUST NOT arm
    # the exit window — otherwise "Ctrl+C clear input, then Ctrl+C
    # clear-again-accidentally-discovers-empty-buffer" would exit.
    # Contract: only BARE Ctrl+C (empty buffer) advances the exit state
    # machine.
    import time as _time

    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from aura.cli.repl import _build_mode_key_bindings, _CtrlCState

    agent = _agent(tmp_path)
    console, _buf = _capture_console()
    state = _CtrlCState()
    kb = _build_mode_key_bindings(agent, console, state)

    with create_pipe_input() as inp:
        inp.send_text("xyz\x03q\r")  # type, clear via Ctrl+C, type "q", submit.
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        result = await session.prompt_async("> ")
    assert result == "q"
    # Buffer-clearing Ctrl+C does NOT advance the double-press state —
    # exit window remains inactive.
    assert state.last_press_at == 0.0
    assert not state.hint_active(_time.monotonic())
    await agent.aclose()


async def test_escape_resets_mode_silently_no_scrollback_spam(
    tmp_path: Path,
) -> None:
    # Same silent-feedback contract as shift+tab: escape must flip
    # mode and invalidate the toolbar, without printing to stdout.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from aura.cli.repl import _build_mode_key_bindings

    agent = _agent(tmp_path)
    agent.set_mode("accept_edits")
    console, buf = _capture_console()
    kb = _build_mode_key_bindings(agent, console)

    with create_pipe_input() as inp:
        inp.send_text("\x1b")
        inp.send_text("q\r")
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        await session.prompt_async("> ")

    assert agent.mode == "default"
    out = buf.getvalue()
    assert "mode:" not in out
    await agent.aclose()


async def test_mention_preprocessor_injects_attachment_into_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The REPL's ``_run_turn`` calls ``extract_and_resolve_attachments``
    # before handing the prompt to ``astream``. When the agent has a live
    # MCP manager that knows the mentioned URI, the rendered
    # ``<mcp-resource>`` envelope must reach the LLM's ``_agenerate``.
    from langchain_core.messages import BaseMessage, HumanMessage

    from aura.core import agent as agent_module

    class _FakeMgr:
        def __init__(self, configs: object) -> None:
            # Minimal MCPManager-shaped object — only the surface used by
            # the preprocessor + aconnect.
            self._resources: dict[tuple[str, str], object] = {
                ("srv", "mem://doc.md"): object(),
            }

        def resources_catalogue(
            self,
        ) -> list[tuple[str, str, str, str, str | None]]:
            return [("srv", "mem://doc.md", "doc.md", "", None)]

        async def read_resource(self, uri: str) -> dict[str, object]:
            return {
                "uri": uri,
                "server": "srv",
                "contents": [
                    {"type": "text", "text": "INJECTED DOC", "uri": uri},
                ],
            }

        async def start_all(self) -> tuple[list[object], list[object]]:
            return [], []

        async def stop_all(self) -> None:
            return None

    monkeypatch.setattr(agent_module, "MCPManager", _FakeMgr)

    captured: list[list[BaseMessage]] = []

    class _Capture(FakeChatModel):
        async def _agenerate(  # type: ignore[override]
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: object | None = None,
            **kwargs: object,
        ) -> object:
            captured.append(list(messages))
            return await super()._agenerate(
                messages, stop, run_manager, **kwargs,  # type: ignore[arg-type]
            )

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "mcp_servers": [
            {
                "name": "srv",
                "transport": "stdio",
                "command": "echo",
                "args": ["noop"],
            },
        ],
    })
    agent = Agent(
        config=cfg,
        model=_Capture(
            turns=[FakeTurn(message=AIMessage(content="ok"))],  # type: ignore[call-arg]
        ),
        storage=SessionStorage(tmp_path / "db"),
    )
    await agent.aconnect()
    console, buf = _capture_console()
    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["pull @srv:mem://doc.md", "/exit"]),
        console=console,
    )
    # B3: live MCP manager inside async loop → must use aclose().
    await agent.aclose()

    assert captured, "model was never invoked"
    # Envelope injected BEFORE the user's HumanMessage.
    sent = captured[0]
    envelopes = [
        m for m in sent
        if isinstance(m, HumanMessage)
        and isinstance(m.content, str)
        and "<mcp-resource" in m.content
    ]
    assert len(envelopes) == 1
    assert "INJECTED DOC" in str(envelopes[0].content)
    # REPL surfaces a dim confirmation line for each attached resource.
    assert "attached @srv:mem://doc.md" in buf.getvalue()


async def test_prompt_without_mentions_skips_attachment_path(
    tmp_path: Path,
) -> None:
    # A plain prompt MUST NOT trigger any MCP calls or inject envelopes,
    # even when the agent has no manager wired (common path).
    from langchain_core.messages import BaseMessage, HumanMessage

    captured: list[list[BaseMessage]] = []

    class _Capture(FakeChatModel):
        async def _agenerate(  # type: ignore[override]
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: object | None = None,
            **kwargs: object,
        ) -> object:
            captured.append(list(messages))
            return await super()._agenerate(
                messages, stop, run_manager, **kwargs,  # type: ignore[arg-type]
            )

    agent = Agent(
        config=AuraConfig.model_validate({
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }),
        model=_Capture(
            turns=[FakeTurn(message=AIMessage(content="ok"))],  # type: ignore[call-arg]
        ),
        storage=SessionStorage(tmp_path / "db"),
    )
    console, _buf = _capture_console()
    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["just a plain prompt", "/exit"]),
        console=console,
    )
    await agent.aclose()

    assert captured
    sent = captured[0]
    assert not any(
        isinstance(m, HumanMessage)
        and isinstance(m.content, str)
        and "<mcp-resource" in m.content
        for m in sent
    )


async def test_turn_exception_does_not_kill_repl(tmp_path: Path) -> None:
    # Real resilience: if Agent.astream raises (network error, client bug,
    # provider 500), the REPL must print an error and keep looping, NOT
    # crash the interactive session with a traceback.
    class _ExplodingModel:
        async def ainvoke(self, *a: object, **kw: object) -> object:
            raise RuntimeError("network went sideways")
        def bind_tools(self, tools: list[object]) -> _ExplodingModel:
            return self

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    agent = Agent(
        config=cfg,
        model=_ExplodingModel(),  # type: ignore[arg-type]
        storage=SessionStorage(tmp_path / "db"),
    )
    console, buf = _capture_console()

    # Two lines: the first triggers the exploding model; the second /exit.
    # If the REPL is resilient, both get processed and we exit cleanly.
    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["hello", "/exit"]),
        console=console,
    )
    out = buf.getvalue()
    assert "turn failed" in out
    assert "network went sideways" in out
    await agent.aclose()
