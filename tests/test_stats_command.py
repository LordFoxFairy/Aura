"""Tests for ``/stats`` — current-session token usage summary (V13-T2A).

Covers:
- Empty-state message when no turn has completed yet (friendly nudge, not a crash).
- Single-turn counts: totals reflect the one turn's input/output/cache.
- Multi-turn accumulation: totals sum, last-turn row shows only the latest.
- Integration with ``make_usage_tracking_hook``: running the hook twice then
  /stats shows coherent numbers end-to-end.
- Journal: ``turn_usage`` event emitted every turn with per-turn counts +
  model name extracted from ``response_metadata``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from aura.core.commands.stats import StatsCommand
from aura.core.hooks.budget import make_usage_tracking_hook
from aura.core.persistence import journal as journal_module
from aura.schemas.state import LoopState


class _StubAgent:
    """Minimal Agent-shaped object exposing ``_state`` — all /stats needs."""

    def __init__(self, state: LoopState) -> None:
        self._state = state


def _ai(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    total: int | None = None,
    model: str = "",
) -> AIMessage:
    msg = AIMessage(content="ok")
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if total is not None:
        usage["total_tokens"] = total
    msg.usage_metadata = usage  # type: ignore[assignment]
    if cache_read or model:
        meta: dict[str, Any] = {}
        if cache_read:
            meta["usage"] = {"cache_read_input_tokens": cache_read}
        if model:
            meta["model_name"] = model
        msg.response_metadata = meta
    return msg


@pytest.mark.asyncio
async def test_stats_empty_state_friendly_message() -> None:
    agent = _StubAgent(LoopState())
    out = await StatsCommand().handle("", agent)  # type: ignore[arg-type]
    assert out.handled is True
    assert out.kind == "print"
    assert "No usage recorded yet" in out.text


@pytest.mark.asyncio
async def test_stats_after_one_turn(tmp_path: Path) -> None:
    """Single hook fire → /stats shows that turn's numbers."""
    journal_module.reset()
    journal_module.configure(tmp_path / "events.jsonl")
    try:
        state = LoopState()
        hook = make_usage_tracking_hook()
        await hook(
            ai_message=_ai(input_tokens=1000, output_tokens=50, cache_read=800),
            history=[],
            state=state,
        )
        agent = _StubAgent(state)
        out = await StatsCommand().handle("", agent)  # type: ignore[arg-type]

        assert "1 turn" in out.text  # singular, no trailing "s"
        assert "1,000" in out.text  # input total
        assert "50" in out.text      # output total
        assert "800" in out.text     # cache
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_stats_accumulates_across_turns(tmp_path: Path) -> None:
    """Two hook fires → totals add, last-turn row shows the second turn only."""
    journal_module.reset()
    journal_module.configure(tmp_path / "events.jsonl")
    try:
        state = LoopState()
        hook = make_usage_tracking_hook()
        await hook(
            ai_message=_ai(input_tokens=1000, output_tokens=50, cache_read=200),
            history=[],
            state=state,
        )
        await hook(
            ai_message=_ai(input_tokens=500, output_tokens=30, cache_read=400),
            history=[],
            state=state,
        )
        agent = _StubAgent(state)
        out = await StatsCommand().handle("", agent)  # type: ignore[arg-type]

        assert "2 turns" in out.text         # plural
        assert "1,500" in out.text           # input total (1000 + 500)
        assert "80" in out.text              # output total (50 + 30)
        assert "600" in out.text             # cache total (200 + 400)
        # Last turn row should reflect ONLY the second call.
        assert "in 500" in out.text
        assert "out 30" in out.text
        assert "cache 400" in out.text
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_turn_usage_journal_event_emitted_every_turn(tmp_path: Path) -> None:
    """Every post_model hook fire should produce a ``turn_usage`` event
    carrying per-turn input / output / cache + optional model name."""
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        state = LoopState()
        state.turn_count = 1
        hook = make_usage_tracking_hook()
        await hook(
            ai_message=_ai(
                input_tokens=700,
                output_tokens=40,
                cache_read=500,
                model="claude-sonnet-4-6",
            ),
            history=[],
            state=state,
        )
        state.turn_count = 2
        await hook(
            ai_message=_ai(
                input_tokens=300,
                output_tokens=20,
                cache_read=100,
                model="claude-sonnet-4-6",
            ),
            history=[],
            state=state,
        )
        events = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        usage_events = [e for e in events if e["event"] == "turn_usage"]
        assert len(usage_events) == 2
        assert usage_events[0]["turn"] == 1
        assert usage_events[0]["input_tokens"] == 700
        assert usage_events[0]["output_tokens"] == 40
        assert usage_events[0]["cache_read_tokens"] == 500
        assert usage_events[0]["model"] == "claude-sonnet-4-6"
        assert usage_events[1]["turn"] == 2
        assert usage_events[1]["input_tokens"] == 300
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_turn_usage_event_handles_missing_model(tmp_path: Path) -> None:
    """Providers that don't expose model_name in response_metadata still
    work — model field is best-effort, falls back to empty string."""
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        state = LoopState()
        hook = make_usage_tracking_hook()
        await hook(
            ai_message=_ai(input_tokens=100, output_tokens=10),
            history=[],
            state=state,
        )
        events = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        usage = [e for e in events if e["event"] == "turn_usage"]
        assert len(usage) == 1
        assert usage[0]["model"] == ""
    finally:
        journal_module.reset()
