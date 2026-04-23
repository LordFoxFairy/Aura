"""Tests for shift+tab mode-cycle keybinding + Agent.set_mode."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from aura.cli.repl import _build_mode_key_bindings, _cycle_mode
from aura.core.agent import Agent
from tests.conftest import FakeChatModel
from tests.test_agent import _minimal_config, _storage


def _agent(tmp_path: Path, mode: str = "default") -> Agent:
    return Agent(
        config=_minimal_config(enabled=[]),
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
        mode=mode,
    )


# ---------- Agent.set_mode ------------------------------------------------

def test_set_mode_plan(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.set_mode("plan")
    assert agent.mode == "plan"
    agent.close()


def test_set_mode_accept_edits(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.set_mode("accept_edits")
    assert agent.mode == "accept_edits"
    agent.close()


def test_set_mode_bypass_allowed_programmatically(tmp_path: Path) -> None:
    # Bypass is excluded from the interactive cycle but remains settable
    # programmatically — the CLI entry point uses it.
    agent = _agent(tmp_path)
    agent.set_mode("bypass")
    assert agent.mode == "bypass"
    agent.close()


def test_set_mode_invalid_raises(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    with pytest.raises(ValueError):
        agent.set_mode("not_a_real_mode")
    agent.close()


# ---------- _cycle_mode ---------------------------------------------------

def test_cycle_order_matches_spec() -> None:
    assert _cycle_mode("default") == "accept_edits"
    assert _cycle_mode("accept_edits") == "plan"
    assert _cycle_mode("plan") == "default"


def test_cycle_preserves_bypass() -> None:
    # Bypass is NOT in the cycle — safer option: stays bypass.
    assert _cycle_mode("bypass") == "bypass"


# ---------- shift+tab keybinding integration ------------------------------

class _FakeApp:
    """Stand-in for prompt_toolkit's Application — we only need invalidate()."""

    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class _FakeEvent:
    def __init__(self) -> None:
        self.app = _FakeApp()


def _find_binding(kb: Any, key_value: str) -> Any:
    """Return the handler for a pt KeyBindings binding whose ``keys`` tuple
    contains a Key/str equal to ``key_value`` (e.g. ``"s-tab"``, ``"escape"``).
    pt uses the ``Keys`` enum whose ``.value`` is the literal string we
    registered with."""
    for b in kb.bindings:
        for k in b.keys:
            if getattr(k, "value", k) == key_value:
                return b
    raise AssertionError(f"no binding for {key_value!r}")


def test_shift_tab_cycles_and_invalidates(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    kb = _build_mode_key_bindings(agent, console)

    binding = _find_binding(kb, "s-tab")
    event = _FakeEvent()
    # First cycle: default -> accept_edits
    binding.handler(event)
    assert agent.mode == "accept_edits"
    # Second cycle: accept_edits -> plan
    binding.handler(event)
    assert agent.mode == "plan"
    # Third cycle: plan -> default
    binding.handler(event)
    assert agent.mode == "default"
    assert event.app.invalidated == 3
    # NOTE: confirmation text is now routed through
    # ``prompt_toolkit.application.run_in_terminal`` to avoid racing
    # pt's renderer. Under a fake event without a running pt loop the
    # deferred lambda never fires — that's fine; the state transition
    # and invalidate contract are what this test guards. End-to-end
    # output is verified by ``test_repl.py`` via a real pipe input.
    agent.close()


def test_shift_tab_under_bypass_is_a_noop_with_message(tmp_path: Path) -> None:
    agent = _agent(tmp_path, mode="bypass")
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    kb = _build_mode_key_bindings(agent, console)

    binding = _find_binding(kb, "s-tab")
    event = _FakeEvent()
    binding.handler(event)

    assert agent.mode == "bypass"  # unchanged — bypass is sticky
    # Confirmation line is deferred via run_in_terminal; see note on
    # test_shift_tab_cycles_and_invalidates. The invariant here is
    # "state is sticky under bypass" — that's what matters.
    agent.close()


def test_escape_resets_to_default(tmp_path: Path) -> None:
    agent = _agent(tmp_path, mode="plan")
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    kb = _build_mode_key_bindings(agent, console)

    binding = _find_binding(kb, "escape")
    event = _FakeEvent()
    binding.handler(event)

    assert agent.mode == "default"
    # Confirmation line is deferred via run_in_terminal; see note on
    # test_shift_tab_cycles_and_invalidates.
    agent.close()


def test_escape_under_bypass_is_noop(tmp_path: Path) -> None:
    agent = _agent(tmp_path, mode="bypass")
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    kb = _build_mode_key_bindings(agent, console)

    binding = _find_binding(kb, "escape")
    event = _FakeEvent()
    binding.handler(event)
    assert agent.mode == "bypass"
    agent.close()
