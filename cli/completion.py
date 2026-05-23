"""Slash-command tab completion + history-path resolution for the REPL."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText

from aura.application.commands import CommandRegistry

_AURA_DIR = ".aura"
_HISTORY_FILE = "history"


class SlashCommandCompleter(Completer):
    # Registry passed as a callable so post-init commands (skills, MCP) complete live.

    def __init__(self, registry_getter: Callable[[], CommandRegistry]) -> None:
        self._registry_getter = registry_getter

    def get_completions(
        self, document: Document, complete_event: CompleteEvent | None,
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        registry = self._registry_getter()
        for cmd in registry.list():
            if not cmd.name.startswith(text):
                continue
            if cmd.argument_hint:
                display: str | FormattedText = FormattedText([
                    ("", cmd.name),
                    ("", " "),
                    ("class:completion-menu.meta", cmd.argument_hint),
                ])
            else:
                display = cmd.name
            # display_meta is one line in pt — multi-line descriptions corrupt the column.
            meta_first_line = cmd.description.split("\n", 1)[0].strip()
            yield Completion(
                cmd.name,
                start_position=-len(text),
                display=display,
                display_meta=meta_first_line,
            )


def resolve_history_path() -> Path:
    aura_dir = Path.home() / _AURA_DIR
    aura_dir.mkdir(parents=True, exist_ok=True)
    return aura_dir / _HISTORY_FILE
