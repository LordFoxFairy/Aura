"""Tests for the ``/buddy`` slash command (T2B-sprites, v0.13).

Exercises the full render path: species+rarity+mood header, composed
5-line sprite, mood footer, plus both opt-out paths (env var + config
flag). No prompt_toolkit dependency — the command returns a plain
``CommandResult.text`` so assertions operate on a single string.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.cli.buddy import current_user_seed, generate_buddy
from aura.cli.commands import build_default_registry, dispatch
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel


def _agent(tmp_path: Path) -> Agent:
    """Construct a minimal Agent — same shape used in ``test_commands.py``."""
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


def _agent_with_buddy_disabled(tmp_path: Path) -> Agent:
    """Agent whose ``config.ui.buddy_enabled`` is False."""
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "ui": {"buddy_enabled": False},
    })
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "db"),
    )


# ---------------------------------------------------------------------------
# Happy-path render.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buddy_command_renders_species_rarity_mood(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output carries species name, rarity label, and a known mood word."""
    monkeypatch.delenv("AURA_NO_BUDDY", raising=False)
    agent = _agent(tmp_path)
    r = build_default_registry()
    result = await dispatch("/buddy", agent, r)
    assert result.handled is True
    assert result.kind == "print"

    # Rehydrate the expected buddy with the same seed source the command
    # uses — keeps the test seed-agnostic across CI environments.
    expected = generate_buddy(current_user_seed())
    assert expected.species in result.text
    assert f"({expected.rarity})" in result.text
    # Mood defaults to "idle" on a freshly-constructed state.
    assert "Mood: idle" in result.text


@pytest.mark.asyncio
async def test_buddy_command_includes_sprite_5_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output has blank line + 5 sprite lines + blank line + mood footer.

    Finds the blank line after the ``Eye: / Hat:`` header and counts 5
    non-empty sprite lines before the next blank.
    """
    monkeypatch.delenv("AURA_NO_BUDDY", raising=False)
    agent = _agent(tmp_path)
    r = build_default_registry()
    result = await dispatch("/buddy", agent, r)
    lines = result.text.splitlines()

    # Spec layout: header(0), eye/hat(1), blank(2), sprite(3..7), blank(8),
    # mood(9). Enforce exact structure so regressions are visible.
    assert len(lines) == 10, f"expected 10 lines, got {len(lines)}: {lines!r}"
    assert lines[0].startswith("Your buddy —")
    assert lines[1].lstrip().startswith("Eye:")
    assert lines[2] == ""
    # Sprite body: 5 lines, each exactly 12 columns wide.
    for i in range(3, 8):
        assert len(lines[i]) == 12, (
            f"sprite line {i} width={len(lines[i])} != 12: {lines[i]!r}"
        )
    assert lines[8] == ""
    assert lines[9].lstrip().startswith("Mood:")


# ---------------------------------------------------------------------------
# Opt-out.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buddy_command_opt_out_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AURA_NO_BUDDY=1`` short-circuits to the "disabled" marker."""
    monkeypatch.setenv("AURA_NO_BUDDY", "1")
    agent = _agent(tmp_path)
    r = build_default_registry()
    result = await dispatch("/buddy", agent, r)
    assert result.handled is True
    assert result.kind == "print"
    assert result.text == "buddy disabled"


@pytest.mark.asyncio
async def test_buddy_command_opt_out_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``config.ui.buddy_enabled=False`` also short-circuits."""
    monkeypatch.delenv("AURA_NO_BUDDY", raising=False)
    agent = _agent_with_buddy_disabled(tmp_path)
    r = build_default_registry()
    result = await dispatch("/buddy", agent, r)
    assert result.handled is True
    assert result.kind == "print"
    assert result.text == "buddy disabled"
