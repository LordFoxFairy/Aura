"""Async REPL — the main loop that drives the agent from user input."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from aura import __version__
from aura.cli.commands import build_default_registry, dispatch
from aura.cli.render import Renderer
from aura.cli.spinner import ThinkingSpinner
from aura.core.agent import Agent
from aura.core.persistence import journal

InputFn = Callable[[str], Awaitable[str]]


async def _default_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def run_repl_async(
    agent: Agent,
    *,
    input_fn: InputFn | None = None,
    console: Console | None = None,
    verbose: bool = False,
    bypass: bool = False,
) -> None:
    journal.write("repl_started")
    _input = input_fn if input_fn is not None else _default_input
    _console = console if console is not None else Console()
    renderer = Renderer(_console)
    registry = build_default_registry(agent=agent)

    _print_welcome(agent, _console)

    # In bypass mode the startup banner scrolls off after a few turns;
    # encode bypass into the prompt string so every line reminds the user
    # they're in "allow-everything" mode. ANSI red around the marker —
    # works in any terminal that supports colour; plain terminals just see
    # the escapes as text (ugly but not dangerous).
    prompt_str = "\x1b[31maura[!bypass]>\x1b[0m " if bypass else "aura> "

    while True:
        try:
            line = await _input(prompt_str)
        except (EOFError, KeyboardInterrupt):
            journal.write("repl_exit", reason="eof_or_ctrlc")
            _console.print()
            return

        # Empty / whitespace-only input: reprompt silently. Sending an empty
        # HumanMessage to the model always 400s (providers reject empty user
        # turns), so it's a pure UX nuisance to round-trip it.
        if not line.strip():
            continue

        journal.write("user_input", line=line[:500])

        result = await dispatch(line, agent, registry)
        if result.handled:
            journal.write(
                "slash_command",
                line=line[:200],
                kind=result.kind,
            )
            if result.kind == "exit":
                journal.write("repl_exit", reason="slash_exit")
                return
            if result.text:
                _console.print(result.text)
            continue

        try:
            await _run_turn(agent, line, renderer, _console)
        except Exception as exc:  # noqa: BLE001 — REPL resilience
            # Don't catch BaseException: KeyboardInterrupt/SystemExit/
            # CancelledError must still propagate up to main() so the
            # whole process can exit cleanly. But a network hiccup, an
            # LLM client bug, or a provider 500 should NOT tear down the
            # user's interactive session — surface the error, journal it,
            # loop back to the prompt.
            journal.write(
                "turn_failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
            _console.print(
                f"[red]turn failed: {type(exc).__name__}: {exc}[/red]"
            )

        if verbose:
            _print_verbose_summary(agent, _console)


def _print_welcome(agent: Agent, console: Console) -> None:
    body = Text()
    body.append("✱ Welcome to Aura", style="bold")
    body.append(f" v{__version__}\n\n", style="dim")
    body.append("/help", style="cyan")
    body.append(" for help\n\n", style="dim")
    body.append("cwd: ", style="dim")
    body.append(f"{Path.cwd()}\n", style="")
    body.append("model: ", style="dim")
    body.append(agent.current_model, style="")
    console.print(Panel(body, border_style="cyan", padding=(0, 2)))


def _print_verbose_summary(agent: Agent, console: Console) -> None:
    state = agent.state
    console.print(
        f"[dim]\\[turn {state.turn_count} \u00b7 "
        f"{state.total_tokens_used:,} tokens \u00b7 "
        f"{agent.current_model}][/dim]"
    )


async def _run_turn(
    agent: Agent, prompt: str, renderer: Renderer, console: Console,
) -> None:
    spinner = ThinkingSpinner(console)
    spinner.start()
    spinner_stopped = False

    async def _stop_spinner() -> None:
        nonlocal spinner_stopped
        if not spinner_stopped:
            spinner_stopped = True
            await spinner.stop()

    async def _stream() -> None:
        async for event in agent.astream(prompt):
            await _stop_spinner()
            renderer.on_event(event)
        await _stop_spinner()
        renderer.finish()

    task = asyncio.create_task(_stream())
    try:
        await task
    except asyncio.CancelledError:
        await _stop_spinner()
    except KeyboardInterrupt:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await _stop_spinner()
    finally:
        await _stop_spinner()
