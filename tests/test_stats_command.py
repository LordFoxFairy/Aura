"""Tests for ``/stats`` — current-session token usage + historical replay.

Covers:
- Empty-state message when no turn has completed yet (friendly nudge, not a crash).
- Single-turn counts: totals reflect the one turn's input/output/cache.
- Multi-turn accumulation: totals sum, last-turn row shows only the latest.
- Integration with ``make_usage_tracking_hook``: running the hook twice then
  /stats shows coherent numbers end-to-end.
- Journal: ``turn_usage`` event emitted every turn with per-turn counts +
  model name extracted from ``response_metadata``.
- Historical replay (``/stats 7d`` and ``/stats all``): journal scan, per-model
  aggregation, cutoff filtering, malformed-line tolerance, friendly errors
  when no journal is configured.
"""

from __future__ import annotations

import json
import time
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

    def __init__(self, state: LoopState, *, config: object | None = None) -> None:
        self._state = state
        self._config = config


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


# ---------------------------------------------------------------------------
# V14-STATS-HISTORY — /stats 7d / /stats all  journal replay
# ---------------------------------------------------------------------------


def _seed_turn_usage(
    path: Path,
    *,
    ts: float,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    turn: int = 1,
) -> None:
    """Append one ``turn_usage`` event line to a JSONL journal file."""
    payload = {
        "ts": ts,
        "event": "turn_usage",
        "turn": turn,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


@pytest.mark.asyncio
async def test_stats_history_no_journal_configured_message(
    tmp_path: Path,
) -> None:
    """``/stats 7d`` with no live journal AND no config.log path → friendly hint."""
    journal_module.reset()
    state = LoopState()
    agent = _StubAgent(state)  # config=None → no log path discoverable
    out = await StatsCommand().handle("7d", agent)  # type: ignore[arg-type]
    assert out.handled is True
    assert "No journal configured" in out.text


@pytest.mark.asyncio
async def test_stats_history_journal_missing_friendly_message(
    tmp_path: Path,
) -> None:
    """Configured journal that hasn't been written to yet → "no file" hint."""
    journal_module.reset()
    journal_module.configure(tmp_path / "never-written.jsonl")
    try:
        state = LoopState()
        agent = _StubAgent(state)
        out = await StatsCommand().handle("7d", agent)  # type: ignore[arg-type]
        assert "not found yet" in out.text
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_stats_history_aggregates_per_model(tmp_path: Path) -> None:
    """``/stats all`` aggregates by model across multiple turns."""
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        now = time.time()
        _seed_turn_usage(
            log, ts=now, model="claude-sonnet-4-6",
            input_tokens=1000, output_tokens=50, cache_read_tokens=500,
        )
        _seed_turn_usage(
            log, ts=now, model="claude-sonnet-4-6",
            input_tokens=2000, output_tokens=80, cache_read_tokens=1500,
        )
        _seed_turn_usage(
            log, ts=now, model="claude-opus-4-7",
            input_tokens=500, output_tokens=200, cache_read_tokens=0,
        )
        state = LoopState()
        agent = _StubAgent(state)
        out = await StatsCommand().handle("all", agent)  # type: ignore[arg-type]
        assert out.handled is True
        assert out.kind == "view"
        # Sonnet sums: input 3000, output 130, cache 2000
        assert "3,000" in out.text
        assert "claude-sonnet-4-6" in out.text
        assert "claude-opus-4-7" in out.text
        # Grand total row appears only when >1 model — sanity-check it's there.
        assert "TOTAL" in out.text
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_stats_history_7d_filters_old_events(tmp_path: Path) -> None:
    """``/stats 7d`` excludes events older than 7 days."""
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        now = time.time()
        eight_days_ago = now - (8 * 86400)
        _seed_turn_usage(
            log, ts=eight_days_ago, model="old-model",
            input_tokens=99999, output_tokens=99999,
        )
        _seed_turn_usage(
            log, ts=now, model="recent-model",
            input_tokens=100, output_tokens=10,
        )
        state = LoopState()
        agent = _StubAgent(state)
        out = await StatsCommand().handle("7d", agent)  # type: ignore[arg-type]
        assert "recent-model" in out.text
        assert "old-model" not in out.text
        assert "99,999" not in out.text
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_stats_history_tolerates_malformed_lines(tmp_path: Path) -> None:
    """Mid-write crash / corrupted line / foreign JSON → silently skipped,
    surrounding valid events still aggregated."""
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        now = time.time()
        _seed_turn_usage(
            log, ts=now, model="m1", input_tokens=100, output_tokens=10,
        )
        # Append a malformed line (truncated JSON simulating a crash mid-write).
        with log.open("a", encoding="utf-8") as fh:
            fh.write('{"ts": 123, "event": "turn_usage", "input_to')  # no newline
            fh.write("\n")
            fh.write("not json at all\n")
            fh.write("\n")  # blank line
        _seed_turn_usage(
            log, ts=now, model="m1", input_tokens=200, output_tokens=20,
        )
        state = LoopState()
        agent = _StubAgent(state)
        out = await StatsCommand().handle("all", agent)  # type: ignore[arg-type]
        # Two valid events for m1: 100+200 input = 300, 10+20 = 30 output.
        assert "300" in out.text  # input total
        assert out.kind == "view"
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_stats_history_empty_window_friendly_message(
    tmp_path: Path,
) -> None:
    """Journal exists but has no ``turn_usage`` events in the window →
    friendly "nothing here yet" message instead of an empty table."""
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        # Write some non-turn_usage events so the file isn't empty but the
        # filter leaves zero rows.
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": time.time(), "event": "config_loaded",
            }) + "\n")
        state = LoopState()
        agent = _StubAgent(state)
        out = await StatsCommand().handle("7d", agent)  # type: ignore[arg-type]
        assert "No ``turn_usage`` events" in out.text
    finally:
        journal_module.reset()
