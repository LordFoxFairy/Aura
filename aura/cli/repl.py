"""Async REPL — the main loop that drives the agent from user input."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

from rich.console import Console

from aura.cli.commands import dispatch
from aura.cli.render import Renderer
from aura.cli.spinner import ThinkingSpinner
from aura.core.agent import Agent

InputFn = Callable[[str], Awaitable[str]]


async def _default_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def run_repl_async(
    agent: Agent,
    *,
    input_fn: InputFn | None = None,
    console: Console | None = None,
) -> None:
    _input = input_fn if input_fn is not None else _default_input
    _console = console if console is not None else Console()
    renderer = Renderer(_console)

    while True:
        try:
            line = await _input("aura> ")
        except (EOFError, KeyboardInterrupt):
            _console.print()
            return

        result = dispatch(line, agent)
        if result.handled:
            if result.kind == "exit":
                return
            if result.text:
                _console.print(result.text)
            continue

        await _run_turn(agent, line, renderer, _console)


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
