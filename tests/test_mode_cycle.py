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
    # Zero scrollback output — feedback lives only in the bottom_toolbar
    # (which re-reads agent.mode on each invalidate). Rapid cycling used
    # to spam dozens of "mode: X (press shift+tab …)" lines; that's gone.
    assert buf.getvalue() == ""
    agent.close()


def test_shift_tab_under_bypass_is_a_noop(tmp_path: Path) -> None:
    agent = _agent(tmp_path, mode="bypass")
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    kb = _build_mode_key_bindings(agent, console)

    binding = _find_binding(kb, "s-tab")
    event = _FakeEvent()
    binding.handler(event)

    assert agent.mode == "bypass"  # unchanged — bypass is sticky
    assert buf.getvalue() == ""    # silent no-op; startup banner warns once
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
    assert buf.getvalue() == ""
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


# ---------- Agent._prior_mode (prePlanMode restoration) -------------------


def _agent_with_plan_tools(tmp_path: Path, mode: str = "default") -> Agent:
    # Need enter_plan_mode enabled to cover the wiring end-to-end.
    return Agent(
        config=_minimal_config(enabled=["enter_plan_mode"]),
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
        mode=mode,
    )


def test_prior_mode_starts_as_none(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    # Fresh agent has no plan entries yet — nothing to restore.
    assert agent._prior_mode is None
    agent.close()


def test_enter_plan_mode_captures_prior_accept_edits(tmp_path: Path) -> None:
    agent = _agent_with_plan_tools(tmp_path, mode="accept_edits")
    tool = agent._available_tools["enter_plan_mode"]
    tool.invoke({"plan": "1. thing"})
    assert agent.mode == "plan"
    assert agent._prior_mode == "accept_edits"
    agent.close()


def test_enter_plan_mode_captures_prior_default(tmp_path: Path) -> None:
    agent = _agent_with_plan_tools(tmp_path, mode="default")
    tool = agent._available_tools["enter_plan_mode"]
    tool.invoke({"plan": "1. thing"})
    assert agent._prior_mode == "default"
    agent.close()


def test_re_entering_plan_does_not_overwrite_prior(tmp_path: Path) -> None:
    # Enter once from accept_edits → prior = accept_edits. Then re-enter
    # from plan → prior MUST stay accept_edits, not become "plan".
    agent = _agent_with_plan_tools(tmp_path, mode="accept_edits")
    tool = agent._available_tools["enter_plan_mode"]
    tool.invoke({"plan": "1. first"})
    assert agent._prior_mode == "accept_edits"
    tool.invoke({"plan": "1. refined"})  # already in plan — no-op
    assert agent._prior_mode == "accept_edits"
    agent.close()


def test_clear_session_resets_prior_mode(tmp_path: Path) -> None:
    # Regression guard: /clear must drop the captured prior so a stale
    # value from the prior plan cycle doesn't bleed into the next one.
    agent = _agent_with_plan_tools(tmp_path, mode="accept_edits")
    tool = agent._available_tools["enter_plan_mode"]
    tool.invoke({"plan": "1. thing"})
    assert agent._prior_mode == "accept_edits"
    agent.clear_session()
    assert agent._prior_mode is None
    agent.close()
