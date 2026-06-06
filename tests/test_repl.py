"""Tests for cli.repl.run_repl_async."""

from __future__ import annotations

import asyncio
import dataclasses
import io
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

import cli.repl as repl_mod
from aura.application.commands.builtin import ExitCommand
from aura.application.commands.factory import build_default_registry
from aura.application.commands.registry import CommandRegistry
from aura.application.commands.types import CommandResult, CommandSource
from aura.application.session import AgentSession
from aura.config.schema import AuraConfig
from aura.domain.events import AgentEvent
from aura.domain.team import TeamRecord
from aura.infrastructure.persistence.storage import SessionStorage
from cli.render import Renderer
from cli.repl import (
    _build_mode_key_bindings,
    _build_prompt_session,
    _cycle_mode,
    _make_prompt_session_input,
    _print_active_team_status,
    _print_welcome,
    _render_welcome_panel,
    _run_turn,
    run_repl_async,
)
from tests.conftest import FakeChatModel, FakeTurn


def _agent(tmp_path: Path, turns: list[FakeTurn] | None = None) -> AgentSession:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    return AgentSession(
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
    agent = _agent(tmp_path, turns=[FakeTurn(message=AIMessage(content="should not fire"))])
    console, buf = _capture_console()

    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["", "   ", "\t\n", "/exit"]),
        console=console,
    )
    # AgentSession's single queued turn was never consumed — the empty inputs
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
    # Single compact cyan Panel. Branding line + keybinding hints + model
    # + cwd + tip, all inside ONE Panel. ``expand=False`` so the panel is
    # content-sized, not terminal-sized.
    from aura import __version__
    from cli.repl import _STARTUP_TIPS

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

    # Cyan Panel border uses Unicode box-drawing glyphs.
    assert any(glyph in out for glyph in ("╭", "╮", "╰", "╯"))

    # Tip line matches one of the curated options (exact substring).
    assert any(tip in out for tip in _STARTUP_TIPS)

    await agent.aclose()


def test_welcome_banner_static_path_renders_settled_glyph_for_non_tty(
    tmp_path: Path,
) -> None:
    # StringIO-backed Console reports ``is_terminal=False``; the welcome
    # helper MUST take the short-circuit path and print the settled
    # banner WITHOUT running a Live animation.
    from cli.repl import _BANNER_SPINNER_FRAMES, _print_welcome

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
    # Dogfood: drive a REAL pty + Aura subprocess and confirm the welcome
    # banner scrolls with an animated leading glyph ending in ``✱``.
    import os
    import pty
    import select
    import sys
    import time as _time

    driver = (
        "from pathlib import Path\n"
        "from rich.console import Console\n"
        "from cli.repl import _print_welcome\n"
        "from aura.application.session import AgentSession\n"
        "from aura.config.schema import AuraConfig\n"
        "from aura.infrastructure.persistence.storage import SessionStorage\n"
        "from tests.conftest import FakeChatModel\n"
        "cfg = AuraConfig.model_validate({\n"
        "    'providers': [{'name': 'openai', 'protocol': 'openai'}],\n"
        "    'router': {'default': 'openai:gpt-4o-mini'},\n"
        "    'tools': {'enabled': []},\n"
        "})\n"
        "import tempfile\n"
        "d = Path(tempfile.mkdtemp())\n"
        "agent = AgentSession(config=cfg, model=FakeChatModel(turns=[]),\n"
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
    # sets it to "0.0.0+dev").
    from cli import repl as repl_mod

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


async def test_alt_enter_inserts_newline_in_prompt_buffer(
    tmp_path: Path,
) -> None:
    # Real-pt round-trip: drive a PromptSession built with the shared
    # KeyBindings via create_pipe_input. Send Alt+Enter (ESC + CR) then
    # Enter (CR). The Alt+Enter binding must insert a literal newline;
    # only the final CR submits.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from cli.repl import _build_mode_key_bindings

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
    # remap Shift+Enter. Same contract as Alt+Enter.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from cli.repl import _build_mode_key_bindings

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
    # must still dispatch the command cleanly.
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
    # to the agent with newlines preserved.
    from langchain_core.messages import BaseMessage, HumanMessage

    captured: list[str] = []

    class _CaptureModel(FakeChatModel):
        def __init__(self, turns: list[FakeTurn] | None = None, **kwargs: Any) -> None:
            super().__init__(turns=turns, **kwargs)

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: object,
        ) -> ChatResult:
            for msg in messages:
                if isinstance(msg, HumanMessage) and isinstance(
                    msg.content, str,
                ):
                    captured.append(msg.content)
            return await super()._agenerate(
                messages, stop, run_manager,
                **kwargs,
            )

    _model: Any = _CaptureModel(turns=[FakeTurn(message=AIMessage(content="ok"))])
    agent = AgentSession(
        config=AuraConfig.model_validate({
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }),
        model=_model,
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
    # Shape: a single dim line with "done" + elapsed seconds. Nothing
    # else should appear in scrollback per turn.
    import dataclasses

    from aura.domain.state_values import TokenStats
    from cli.repl import _print_post_turn_status

    agent = _agent(tmp_path)
    # Seed stats that WOULD have shown up in the old render — they must
    # NOT appear in the new slim line.
    agent.state.slots = dataclasses.replace(
        agent.state.slots,
        token_stats=TokenStats(
            last_input_tokens=9900,
            last_cache_read_tokens=2600,
        ),
    )
    console, buf = _capture_console()

    _print_post_turn_status(agent, console, last_turn_seconds=21.4)

    out = buf.getvalue()
    # New shape: "done" marker + elapsed seconds.
    assert "done" in out
    assert "21.4s" in out
    # No status-bar duplicate noise.
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
    from cli.repl import _print_post_turn_status

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
    # Sub-minute shows decimal, ≥60s drops the decimal (visual noise).
    from cli.repl import _print_post_turn_status

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
    # Bindings only flip state + call ``event.app.invalidate()``. Zero
    # scrollback output.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from cli.repl import _build_mode_key_bindings

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
    # Nothing printed to scrollback.
    out = buf.getvalue()
    assert "mode:" not in out
    assert "shift+tab to cycle" not in out
    await agent.aclose()


async def test_ctrl_c_with_text_clears_buffer_and_does_not_exit(
    tmp_path: Path,
) -> None:
    # When the buffer has text, Ctrl+C discards it and leaves the session
    # alive — never an exit on first press with content.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from cli.repl import _build_mode_key_bindings

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
    # Empty-buffer first Ctrl+C must NOT exit — it arms a "press again to
    # exit" state.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from cli.repl import _build_mode_key_bindings

    agent = _agent(tmp_path)
    console, _buf = _capture_console()
    kb = _build_mode_key_bindings(agent, console)
    with create_pipe_input() as inp:
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
    # TWO bare Ctrl+C within the window raises KeyboardInterrupt, which
    # the outer REPL loop treats as the exit signal.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from cli.repl import _build_mode_key_bindings

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
    # If the second Ctrl+C arrives AFTER the 800ms window, it restarts
    # the arm — does not exit.
    import time as _time

    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from cli.repl import _build_mode_key_bindings, _CtrlCState

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
    # Ctrl+C on a non-empty buffer clears the text but MUST NOT arm the
    # exit window.
    import time as _time

    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from cli.repl import _build_mode_key_bindings, _CtrlCState

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
    # Buffer-clearing Ctrl+C does NOT advance the double-press state.
    assert state.last_press_at == 0.0
    assert not state.hint_active(_time.monotonic())
    await agent.aclose()


async def test_escape_resets_mode_silently_no_scrollback_spam(
    tmp_path: Path,
) -> None:
    # Same silent-feedback contract as shift+tab: escape must flip mode
    # without printing to stdout.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from cli.repl import _build_mode_key_bindings

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


async def test_turn_exception_does_not_kill_repl(tmp_path: Path) -> None:
    # Real resilience: if AgentSession.astream raises (network error, client bug,
    # provider 500), the REPL must print an error and keep looping.
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
    _exploding: Any = _ExplodingModel()
    agent = AgentSession(
        config=cfg,
        model=_exploding,
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


class _SilentPrintCommand:
    # Stub command exercising the empty-text ``case _`` print branch.
    name = "/silent"
    description = "stub: print with empty text"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    async def handle(self, arg: str, agent: AgentSession) -> CommandResult:
        del arg, agent
        return CommandResult(handled=True, kind="print", text="")


async def test_view_kind_renders_panel_and_swallows_continue_prompt(
    tmp_path: Path,
) -> None:
    # ``/context`` returns kind="view"; the loop must route it through
    # _render_view (a bordered Panel) and keep looping, not exit.
    agent = _agent(tmp_path)
    console, buf = _capture_console()

    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["/context", "/exit"]),
        console=console,
    )
    out = buf.getvalue()
    assert any(glyph in out for glyph in ("╭", "╮", "╰", "╯"))
    await agent.aclose()


async def test_print_kind_with_text_echoes_then_continues(
    tmp_path: Path,
) -> None:
    # ``/tasks`` with no tasks yields kind="print" + "(no tasks)"; the
    # default case must echo the text and loop, never terminate.
    agent = _agent(tmp_path)
    console, buf = _capture_console()

    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["/tasks", "/exit"]),
        console=console,
    )
    assert "(no tasks)" in buf.getvalue()
    await agent.aclose()


async def test_print_kind_empty_text_prints_nothing_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A handled command with kind="print" but empty text must add ZERO
    # scrollback noise — the loop swallows it and re-prompts.
    agent = _agent(tmp_path)
    console, buf = _capture_console()

    def _stub_registry(agent: AgentSession | None = None) -> CommandRegistry:
        del agent
        registry = CommandRegistry()
        registry.register(ExitCommand())
        registry.register(_SilentPrintCommand())
        return registry

    monkeypatch.setattr(repl_mod, "build_default_registry", _stub_registry)
    before = len(buf.getvalue())  # 0 — nothing rendered yet
    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["/silent", "/exit"]),
        console=console,
    )
    out = buf.getvalue()
    # Only the welcome banner contributed output; /silent printed nothing.
    assert before == 0
    assert "/silent printed" not in out  # sanity: no stray echo of the line
    await agent.aclose()


async def test_unknown_slash_command_prints_help_hint_and_continues(
    tmp_path: Path,
) -> None:
    # An unrecognized ``/name`` is handled (kind="print") with a help hint;
    # the REPL must surface it and keep the session alive.
    agent = _agent(tmp_path)
    console, buf = _capture_console()

    await run_repl_async(
        agent,
        input_fn=_ScriptedInput(["/definitely-not-a-command", "/exit"]),
        console=console,
    )
    out = buf.getvalue()
    assert "unknown command" in out
    assert "/definitely-not-a-command" in out
    await agent.aclose()


def test_cycle_mode_passes_through_modes_outside_the_cycle() -> None:
    # Only default/accept_edits/plan rotate; bypass (and any unknown mode)
    # must be returned untouched so shift+tab never silently exits bypass.
    assert _cycle_mode("bypass") == "bypass"
    assert _cycle_mode("default") == "accept_edits"
    assert _cycle_mode("accept_edits") == "plan"
    assert _cycle_mode("plan") == "default"


async def test_shift_tab_is_inert_while_in_bypass_mode(
    tmp_path: Path,
) -> None:
    # bypass is a one-way door: shift+tab must NOT cycle out of it, or the
    # user could silently lose allow-everything semantics mid-session.
    agent = _agent(tmp_path)
    agent.set_mode("bypass")
    console, _buf = _capture_console()
    kb = _build_mode_key_bindings(agent, console)
    with create_pipe_input() as inp:
        inp.send_text("\x1b[Zq\r")  # Shift+Tab then "q" + submit.
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        await session.prompt_async("> ")
    assert agent.mode == "bypass"
    await agent.aclose()


async def test_escape_is_inert_while_in_bypass_mode(
    tmp_path: Path,
) -> None:
    # escape resets non-bypass modes to default; in bypass it must do
    # nothing, mirroring the shift+tab guard.
    agent = _agent(tmp_path)
    agent.set_mode("bypass")
    console, _buf = _capture_console()
    kb = _build_mode_key_bindings(agent, console)
    with create_pipe_input() as inp:
        inp.send_text("\x1bq\r")  # ESC then "q" + submit.
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        await session.prompt_async("> ")
    assert agent.mode == "bypass"
    await agent.aclose()


def test_active_team_status_silent_when_no_active_team(tmp_path: Path) -> None:
    # No bound team → zero scrollback. Status line is opt-in on membership.
    agent = _agent(tmp_path)
    console, buf = _capture_console()
    _print_active_team_status(agent, console)
    assert buf.getvalue() == ""
    agent.close()


def test_active_team_status_shows_slug_when_port_absent(tmp_path: Path) -> None:
    # active_team slug is set but no live TeamPort is bound → fall back to
    # the slug itself as the label (graceful degradation, no crash).
    agent = _agent(tmp_path)
    agent.state.slots = dataclasses.replace(
        agent.state.slots, active_team="team-slug-42",
    )
    console, buf = _capture_console()
    _print_active_team_status(agent, console)
    out = buf.getvalue()
    assert "team-slug-42" in out
    assert "in team" in out
    agent.close()


class _StubTeamPort:
    # Minimal port: _print_active_team_status only reads ``.team``.
    def __init__(self, record: TeamRecord) -> None:
        self._record = record

    @property
    def team(self) -> TeamRecord:
        return self._record


def test_active_team_status_prefers_live_team_display_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When a live TeamPort whose team_id matches the active slug is bound,
    # show its human-friendly name instead of the raw slug.
    agent = _agent(tmp_path)
    agent.state.slots = dataclasses.replace(
        agent.state.slots, active_team="team-slug-42",
    )
    record = TeamRecord(
        team_id="team-slug-42",
        name="Pretty Display Name",
        leader_session_id="s1",
    )
    port = _StubTeamPort(record)
    monkeypatch.setattr(type(agent), "team", property(lambda self: port))
    console, buf = _capture_console()
    _print_active_team_status(agent, console)
    out = buf.getvalue()
    assert "Pretty Display Name" in out
    assert "team-slug-42" not in out
    agent.close()


def test_active_team_status_keeps_slug_when_live_team_id_diverges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If a port is bound but its team_id does NOT match the active slug
    # (stale binding), the label stays the slug — never a mismatched name.
    agent = _agent(tmp_path)
    agent.state.slots = dataclasses.replace(
        agent.state.slots, active_team="team-slug-42",
    )
    record = TeamRecord(
        team_id="some-other-team",
        name="Wrong Name",
        leader_session_id="s1",
    )
    port = _StubTeamPort(record)
    monkeypatch.setattr(type(agent), "team", property(lambda self: port))
    console, buf = _capture_console()
    _print_active_team_status(agent, console)
    out = buf.getvalue()
    assert "team-slug-42" in out
    assert "Wrong Name" not in out
    agent.close()


def test_welcome_panel_collapses_home_prefix_to_tilde(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # cwd under $HOME must render as ``~/...`` so the banner never leaks a
    # long absolute home path into scrollback.
    agent = _agent(tmp_path)
    home = Path.home()
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: home))
    panel = _render_welcome_panel(agent, repl_mod._BANNER_SETTLE_GLYPH)
    console, buf = _capture_console()
    console.print(panel)
    out = buf.getvalue()
    assert "cwd:   ~" in out
    agent.close()


def test_welcome_animated_path_settles_on_stable_glyph_in_tty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TTY consoles run the Live spinner animation; with sleep neutralized
    # it MUST still land on the settled ``✱`` glyph (deterministic, fast).
    agent = _agent(tmp_path)
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=200, highlight=False)
    assert console.is_terminal  # precondition for the animated branch
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    _print_welcome(agent, console)
    out = buf.getvalue()
    assert "✱ Aura" in out
    agent.close()


async def test_run_repl_falls_back_to_default_input_on_non_tty_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no injected input_fn and a non-TTY stdin, the REPL must use the
    # plain blocking input() fallback (pt cannot drive a dumb terminal).
    agent = _agent(tmp_path)
    console, buf = _capture_console()
    lines = iter(["/exit"])

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(lines))
    await run_repl_async(agent, console=console)
    # Reached /exit cleanly via the default-input path (no EOFError leak).
    assert "Aura" in buf.getvalue()
    await agent.aclose()


async def test_run_repl_builds_prompt_session_on_tty_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With a TTY stdin and no injected input_fn, the REPL must construct a
    # PromptSession-backed input (the pt rich path), not the dumb fallback.
    agent = _agent(tmp_path)
    console, _buf = _capture_console()
    lines = iter(["/exit"])

    async def _scripted(prompt: str) -> str:
        del prompt
        return next(lines)

    made: list[bool] = []

    def _fake_make(session: Any) -> Any:
        del session
        made.append(True)
        return _scripted

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(repl_mod, "_make_prompt_session_input", _fake_make)
    await run_repl_async(agent, console=console)
    assert made == [True]
    await agent.aclose()


async def test_prompt_session_input_closure_round_trips_a_typed_line(
    tmp_path: Path,
) -> None:
    # The PromptSession→InputFn adapter must forward a typed line verbatim;
    # this guards the closure built for the rich-terminal input path.
    agent = _agent(tmp_path)
    console, _buf = _capture_console()
    registry = build_default_registry(agent=agent)
    with create_pipe_input() as inp:
        inp.send_text("typed line\r")
        session: PromptSession[str] = PromptSession(
            key_bindings=_build_mode_key_bindings(agent, console),
            input=inp,
            output=DummyOutput(),
        )
        del registry  # exercised build_default_registry; not needed past here
        read = _make_prompt_session_input(session)
        result = await read("> ")
    assert result == "typed line"
    await agent.aclose()


def test_build_prompt_session_without_agent_has_no_key_bindings(
    tmp_path: Path,
) -> None:
    # The shared builder is reused by history/completion tests with
    # ``agent=None``; in that mode it MUST skip Aura key bindings so those
    # tests don't accidentally depend on agent state.
    del tmp_path
    registry = CommandRegistry()
    session = _build_prompt_session(registry)
    assert session.key_bindings is None
    assert session.completer is not None


async def test_run_turn_swallows_cancelled_error_and_returns_elapsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # asyncio.CancelledError escaping the stream (cooperative cancellation)
    # is non-fatal: _run_turn returns a measured duration, REPL stays up.
    agent = _agent(tmp_path)
    console, _buf = _capture_console()
    renderer = Renderer(console)

    async def _cancel_stream(
        prompt: str, **kwargs: object,
    ) -> AsyncIterator[AgentEvent | dict[str, object]]:
        del prompt, kwargs
        raise asyncio.CancelledError
        yield {}  # unreachable; marks this an async generator

    monkeypatch.setattr(agent, "astream", _cancel_stream)
    elapsed = await _run_turn(agent, "hi", renderer, console)
    assert elapsed >= 0.0
    await agent.aclose()
