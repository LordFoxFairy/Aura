"""CLI surface: REPL loop, slash commands, event renderer, and permission asker."""

from aura.capabilities.commands import CommandResult, dispatch
from cli.permission import make_cli_asker
from cli.render import Renderer
from cli.repl import run_repl_async

__all__ = [
    "CommandResult",
    "Renderer",
    "dispatch",
    "make_cli_asker",
    "run_repl_async",
]
