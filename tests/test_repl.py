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
