"""Slash-command tab completion + history-path resolution for the REPL.

``SlashCommandCompleter`` is a ``prompt_toolkit`` :class:`Completer` that only
fires on input whose full ``text_before_cursor`` starts with ``/`` — this is
the invariant we care about for the Aura REPL (slash is always the first
character of a command). Completions are driven by a registry getter, *not* a
snapshot, so Skill and MCP commands registered post-init (e.g. after
``agent.aconnect()``) show up without reshuffling the PromptSession.

``resolve_history_path`` centralises the ``~/.aura/history`` path and its
parent-dir creation — idempotent, so repeated REPL starts don't race on
``mkdir``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText

from aura.core.commands import CommandRegistry

_AURA_DIR = ".aura"
_HISTORY_FILE = "history"


class SlashCommandCompleter(Completer):
    """Completes ``/name`` tokens against the live :class:`CommandRegistry`.

    The registry is accessed via a zero-arg callable rather than a direct
    reference so that commands registered AFTER the completer is constructed
    (Skills, MCP commands) become completable without the REPL having to
    rebuild its PromptSession.
    """

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
            # Display column: ``/name`` + dimmed ``<arg_hint>`` when the
            # command declares one. Mirrors claude-code's completion menu
            # where required args sit after the name in a muted color,
            # distinguishing the invocable token (`/skill-name`) from the
            # hint about what to type after it (`<topic>`).
            if cmd.argument_hint:
                display: str | FormattedText = FormattedText([
                    ("", cmd.name),
                    ("", " "),
                    ("class:completion-menu.meta", cmd.argument_hint),
                ])
            else:
                display = cmd.name
            # display_meta is rendered on a single line by prompt_toolkit's
            # dropdown; a multi-line description (common for skills authored
            # by LLMs that drop a paragraph into ``description``) would push
            # the column into garbage. Keep the first line only — callers
            # who want the full text can use ``/help``.
            meta_first_line = cmd.description.split("\n", 1)[0].strip()
            yield Completion(
                cmd.name,
                start_position=-len(text),
                display=display,
                display_meta=meta_first_line,
            )


def resolve_history_path() -> Path:
    """Return ``~/.aura/history``, creating the parent dir if missing.

    Idempotent (``mkdir(parents=True, exist_ok=True)``). The file itself is
    NOT touched — ``prompt_toolkit.history.FileHistory`` creates it on first
    write.
    """
    aura_dir = Path.home() / _AURA_DIR
    aura_dir.mkdir(parents=True, exist_ok=True)
    return aura_dir / _HISTORY_FILE
