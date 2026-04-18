"""Slash command dispatcher for the REPL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aura.core.agent import Agent
from aura.core.llm import UnknownModelSpecError

CommandKind = Literal["print", "exit", "noop"]


@dataclass(frozen=True)
class CommandResult:
    handled: bool
    kind: CommandKind
    text: str


_HELP_TEXT = """Available commands:
  /help          show this message
  /exit          exit the REPL (Ctrl+D also works)
  /clear         clear the current session's history
  /model         show the current provider:model
  /model <spec>  switch model (router alias or provider:model form)

Anything else is sent as a prompt to the agent."""


def dispatch(line: str, agent: Agent) -> CommandResult:
    stripped = line.strip()
    if not stripped.startswith("/"):
        return CommandResult(handled=False, kind="noop", text="")

    parts = stripped.split(None, 1)
    cmd = parts[0]
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/help":
        return CommandResult(handled=True, kind="print", text=_HELP_TEXT)

    if cmd == "/exit":
        return CommandResult(handled=True, kind="exit", text="")

    if cmd == "/clear":
        agent.clear_session()
        return CommandResult(handled=True, kind="print", text="session cleared")

    if cmd == "/model":
        return _handle_model(arg, agent)

    return CommandResult(
        handled=True,
        kind="print",
        text=f"unknown command: {cmd} (try /help)",
    )


def _handle_model(spec: str, agent: Agent) -> CommandResult:
    if not spec:
        return CommandResult(
            handled=True, kind="print", text=_model_status(agent),
        )
    try:
        agent.switch_model(spec)
    except UnknownModelSpecError as exc:
        return CommandResult(handled=True, kind="print", text=f"error: {exc}")
    return CommandResult(handled=True, kind="print", text=f"switched to {spec}")


def _model_status(agent: Agent) -> str:
    default = agent.current_model or "?"
    aliases = sorted(agent.router_aliases)
    lines = [f"current default: {default}"]
    if aliases:
        lines.append("aliases:")
        for alias in aliases:
            lines.append(f"  {alias} -> {agent.router_aliases[alias]}")
    return "\n".join(lines)
