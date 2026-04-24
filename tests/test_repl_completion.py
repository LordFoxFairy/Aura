"""Tests for REPL input-UX polish: slash completer + history wiring."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from prompt_toolkit.document import Document

from aura.cli.completion import SlashCommandCompleter, resolve_history_path
from aura.core.commands import CommandRegistry
from aura.core.commands.types import CommandResult

if TYPE_CHECKING:
    from aura.core.agent import Agent


class _FakeCommand:
    """Minimal Command-protocol stand-in — no Agent required."""

    source = "builtin"
    allowed_tools: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        description: str,
        argument_hint: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.argument_hint = argument_hint

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        return CommandResult(handled=True, kind="print", text="")


def _registry(*commands: _FakeCommand) -> CommandRegistry:
    r = CommandRegistry()
    for c in commands:
        r.register(c)  # type: ignore[arg-type]
    return r


def _completions(completer: SlashCommandCompleter, text: str) -> list[str]:
    doc = Document(text=text, cursor_position=len(text))
    return [c.text for c in completer.get_completions(doc, complete_event=None)]


def test_slash_completer_offers_built_in_commands() -> None:
    """Typing ``/h`` → completer yields ``/help`` (and only matching names)."""
    r = _registry(
        _FakeCommand("/help", "show help"),
        _FakeCommand("/exit", "quit the REPL"),
    )
    completer = SlashCommandCompleter(lambda: r)

    matches = _completions(completer, "/h")
    assert matches == ["/help"]


def test_slash_completer_includes_skill_commands() -> None:
    """Skill commands registered post-init are picked up by the live getter."""
    r = _registry(_FakeCommand("/help", "show help"))
    completer = SlashCommandCompleter(lambda: r)

    # Register a skill-style command AFTER the completer is built — this
    # mimics how Skills and MCP commands show up at agent.aconnect() time.
    r.register(_FakeCommand("/skill-name", "invoke skill"))  # type: ignore[arg-type]

    matches = _completions(completer, "/sk")
    assert matches == ["/skill-name"]


def test_slash_completer_respects_cursor_position() -> None:
    """Mid-line non-slash input yields no completions — we only complete
    when the whole buffer up to the cursor starts with ``/``. Mixing
    prompt text with trailing ``/help`` is NOT a slash command."""
    r = _registry(_FakeCommand("/help", "show help"))
    completer = SlashCommandCompleter(lambda: r)

    assert _completions(completer, "please run /he") == []


def test_non_slash_input_yields_no_completions() -> None:
    """Regular prompt text → empty completion stream."""
    r = _registry(_FakeCommand("/help", "show help"))
    completer = SlashCommandCompleter(lambda: r)

    assert _completions(completer, "hello agent") == []
    assert _completions(completer, "") == []


def test_slash_completer_exposes_description_in_display_meta() -> None:
    """Completion carries the command description for the dropdown's meta column."""
    r = _registry(_FakeCommand("/help", "show help"))
    completer = SlashCommandCompleter(lambda: r)

    doc = Document(text="/he", cursor_position=3)
    completions = list(completer.get_completions(doc, complete_event=None))
    assert len(completions) == 1
    # display_meta may be a plain str or FormattedText — stringify for the assert.
    assert "show help" in str(completions[0].display_meta)


def test_slash_completer_renders_argument_hint_after_name() -> None:
    """Commands with an ``argument_hint`` surface it dimmed after the name."""
    r = _registry(
        _FakeCommand(
            "/skill-name", "invoke skill", argument_hint="<topic>",
        ),
    )
    completer = SlashCommandCompleter(lambda: r)

    doc = Document(text="/sk", cursor_position=3)
    completions = list(completer.get_completions(doc, complete_event=None))
    assert len(completions) == 1
    # display is a FormattedText (list of style/text tuples) when a hint is
    # present. Stringify both the display and the raw tuples to verify both
    # the name and the hint are included.
    display = completions[0].display
    rendered = "".join(
        text for _, text, *_ in display
    ) if not isinstance(display, str) else display
    assert "/skill-name" in rendered
    assert "<topic>" in rendered


def test_slash_completer_omits_argument_hint_when_absent() -> None:
    """Commands without an ``argument_hint`` show only the name in display."""
    r = _registry(_FakeCommand("/exit", "quit"))
    completer = SlashCommandCompleter(lambda: r)

    doc = Document(text="/ex", cursor_position=3)
    completions = list(completer.get_completions(doc, complete_event=None))
    assert len(completions) == 1
    # prompt_toolkit normalizes the display to FormattedText; extract and
    # assert the rendered text has the name only (no hint placeholder).
    display = completions[0].display
    rendered = "".join(
        text for _, text, *_ in display
    ) if not isinstance(display, str) else display
    assert rendered == "/exit"
    assert "<" not in rendered


def test_slash_completer_collapses_multiline_description() -> None:
    """Multi-line descriptions (common on LLM-authored skills) render as one
    line in the meta column, not as garbled multi-line text."""
    body = (
        "Use when starting any conversation.\n"
        "Establishes how to find and use skills."
    )
    r = _registry(_FakeCommand("/using-superpowers", body))
    completer = SlashCommandCompleter(lambda: r)

    doc = Document(text="/us", cursor_position=3)
    completions = list(completer.get_completions(doc, complete_event=None))
    assert len(completions) == 1
    meta = completions[0].display_meta
    rendered = "".join(
        text for _, text, *_ in meta
    ) if not isinstance(meta, str) else meta
    assert "\n" not in rendered
    assert rendered == "Use when starting any conversation."


def test_history_file_path_under_aura_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History file resolves to ``<HOME>/.aura/history``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = resolve_history_path()
    assert path == tmp_path / ".aura" / "history"


def test_history_parent_dir_created_on_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling resolve_history_path creates ``~/.aura/`` if absent and is idempotent."""
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".aura"
    assert not target.exists()

    resolve_history_path()
    assert target.is_dir()

    # Second call must not raise even when the dir already exists.
    resolve_history_path()
    assert target.is_dir()
