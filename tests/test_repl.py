"""Tests for aura.cli.repl.run_repl_async."""

from __future__ import annotations

import io
from pathlib import Path

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
    agent.close()


async def test_help_then_exit(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    console, buf = _capture_console()

    await run_repl_async(
        agent, input_fn=_ScriptedInput(["/help", "/exit"]), console=console,
    )
    assert "/exit" in buf.getvalue()
    agent.close()


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
    agent.close()


async def test_eof_exits_cleanly(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    console, buf = _capture_console()

    await run_repl_async(
        agent, input_fn=_ScriptedInput([]), console=console,
    )
    agent.close()


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
    agent.close()


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
    agent.close()


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
    agent.close()


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
    agent.close()


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
    agent.close()


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

    agent.close()


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
    agent.close()


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
    agent.close()


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
    agent.close()


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
    agent.close()


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
    agent.close()


async def test_single_line_slash_command_unchanged(tmp_path: Path) -> None:
    # Regression guard: the multi-line dispatch split must not break
    # the overwhelmingly common single-line slash-command path.
    agent = _agent(tmp_path)
    console, buf = _capture_console()
    await run_repl_async(
        agent, input_fn=_ScriptedInput(["/help", "/exit"]), console=console,
    )
    assert "/exit" in buf.getvalue()
    agent.close()


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
    agent.close()
