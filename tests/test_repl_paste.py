"""F-05-001: bracketed-paste handling in the REPL.

Pins:
  - Small pastes (under both byte AND line thresholds) are inserted
    verbatim — no placeholder, no UX surprise for normal copy/paste.
  - Large pastes collapse to ``[paste · 12 KB · 312 lines · #N]`` in
    the visible buffer.
  - On submit, the placeholder is expanded back into the original text
    so the agent never sees the placeholder string.
  - The completion menu / autocomplete state is reset during the
    paste insert so a 200-line paste doesn't trigger a per-keystroke
    completion storm.

These tests drive a real ``PromptSession`` over ``create_pipe_input``
so the full bracketed-paste sequence (``\\x1b[200~ ... \\x1b[201~``)
flows through pt's vt100 parser, the user binding, and back out via
``prompt_async`` — same path a real terminal exercises.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from aura.cli.repl import (
    _build_mode_key_bindings,
    _build_paste_placeholder,
    _expand_paste_placeholders,
    _PasteStore,
    _should_collapse_paste,
)
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _agent(tmp_path: Path) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "db"),
    )


def _bracketed(content: str) -> str:
    """Wrap ``content`` in the VT100 bracketed-paste markers."""
    return f"\x1b[200~{content}\x1b[201~"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_should_collapse_paste_threshold_lines() -> None:
    small = "line\n" * 50
    big = "line\n" * 150
    assert not _should_collapse_paste(small)
    assert _should_collapse_paste(big)


def test_should_collapse_paste_threshold_bytes() -> None:
    small = "x" * 1024
    big = "x" * (8 * 1024 + 1)
    assert not _should_collapse_paste(small)
    assert _should_collapse_paste(big)


def test_build_paste_placeholder_format() -> None:
    content = "abc\n" * 200
    placeholder = _build_paste_placeholder(content, paste_id=1)
    # Shape: [paste · <size> <unit> · <lines> lines · #<id>]
    assert placeholder.startswith("[paste · ")
    assert placeholder.endswith("· #1]")
    assert "200 lines" in placeholder


def test_expand_paste_placeholders_substitutes() -> None:
    store = _PasteStore()
    pid = store.stash("the original 100-line paste")
    placeholder = _build_paste_placeholder(
        "the original 100-line paste", paste_id=pid,
    )
    line = f"hey look at this: {placeholder} — what do you think?"
    expanded = _expand_paste_placeholders(line, store)
    assert "the original 100-line paste" in expanded
    assert "[paste" not in expanded


def test_expand_paste_placeholders_keeps_unknown_id_verbatim() -> None:
    store = _PasteStore()
    line = "this is fake: [paste · 1 KB · 50 lines · #999]"
    expanded = _expand_paste_placeholders(line, store)
    assert expanded == line


# ---------------------------------------------------------------------------
# pt round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_small_paste_inserted_verbatim(tmp_path: Path) -> None:
    """Below threshold: paste shows up in the buffer as-is."""
    agent = _agent(tmp_path)
    paste_store = _PasteStore()
    kb = _build_mode_key_bindings(
        agent, console=None, paste_store=paste_store,
    )

    small_paste = "hello world\nsecond line\nthird"
    with create_pipe_input() as inp:
        inp.send_text(_bracketed(small_paste) + "\r")
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        result = await session.prompt_async("> ")

    assert result == small_paste
    # Nothing stashed.
    assert paste_store._next_id == 0  # noqa: SLF001
    await agent.aclose()


@pytest.mark.asyncio
async def test_large_paste_collapsed_to_placeholder(tmp_path: Path) -> None:
    """Above threshold: the buffer carries a single placeholder line."""
    agent = _agent(tmp_path)
    paste_store = _PasteStore()
    kb = _build_mode_key_bindings(
        agent, console=None, paste_store=paste_store,
    )

    big_paste = "line\n" * 150
    with create_pipe_input() as inp:
        inp.send_text(_bracketed(big_paste) + "\r")
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        # ``prompt_async`` returns the buffer's literal text — for a
        # collapsed paste that's the placeholder, NOT the original.
        result = await session.prompt_async("> ")

    assert result.startswith("[paste · ")
    assert result.endswith("· #1]")
    assert "150 lines" in result
    # The original got stashed.
    assert paste_store.get(1) == big_paste
    await agent.aclose()


@pytest.mark.asyncio
async def test_placeholder_expands_on_submit(tmp_path: Path) -> None:
    """The REPL loop re-substitutes the placeholder before dispatch."""
    agent = _agent(tmp_path)
    paste_store = _PasteStore()
    kb = _build_mode_key_bindings(
        agent, console=None, paste_store=paste_store,
    )

    big_paste = "x" * (10 * 1024)
    with create_pipe_input() as inp:
        inp.send_text("look at this: " + _bracketed(big_paste) + "\r")
        session: PromptSession[str] = PromptSession(
            key_bindings=kb, input=inp, output=DummyOutput(),
        )
        raw_line = await session.prompt_async("> ")

    # Expansion is what the REPL loop does post-prompt.
    expanded = _expand_paste_placeholders(raw_line, paste_store)
    assert big_paste in expanded
    assert "[paste · " not in expanded
    await agent.aclose()


@pytest.mark.asyncio
async def test_paste_disables_autocomplete_during_event(
    tmp_path: Path,
) -> None:
    """A large paste must not leave the buffer with a stale completion menu.

    We pre-seed ``buffer.complete_state`` (the menu-open sentinel pt
    uses) and verify the BracketedPaste binding clears it BEFORE
    inserting the placeholder. Without the clear, pt's filter step
    would re-match against the freshly inserted text and a 250-line
    paste would trigger ~1 menu re-open per keystroke worth of paste
    bytes — the exact pathology F-05-001 calls out.
    """
    from prompt_toolkit.buffer import Buffer, CompletionState
    from prompt_toolkit.document import Document
    from prompt_toolkit.keys import Keys

    agent = _agent(tmp_path)
    paste_store = _PasteStore()
    kb = _build_mode_key_bindings(
        agent, console=None, paste_store=paste_store,
    )

    big_paste = "data\n" * 250
    buf = Buffer()

    # Build the menu-open sentinel directly. The exact CompletionState
    # internals don't matter — we only care that "non-None goes in,
    # None comes out" after the paste binding runs.
    buf.complete_state = CompletionState(
        original_document=Document(text="", cursor_position=0),
        completions=[],
    )
    assert buf.complete_state is not None

    class _FakeApp:
        def invalidate(self) -> None:
            pass

        def exit(self, *, exception: BaseException | None = None) -> None:
            pass

    class _FakeEvent:
        def __init__(self, data: str) -> None:
            self.data = data
            self.current_buffer = buf
            self.app = _FakeApp()

    handlers = [b for b in kb.bindings if Keys.BracketedPaste in b.keys]
    assert handlers, "BracketedPaste binding must be installed"
    handlers[0].handler(_FakeEvent(big_paste))  # type: ignore[arg-type]

    # Buffer carries the placeholder, NOT the original 250 lines.
    assert buf.text.startswith("[paste · ")
    # Completion state was cleared by the binding — no more in-flight menu.
    assert buf.complete_state is None
    await agent.aclose()
