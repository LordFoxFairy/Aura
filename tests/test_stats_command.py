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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata

from aura.application.commands.stats import StatsCommand
from aura.application.hooks.budget import make_usage_tracking_hook
from aura.application.loop_state import LoopState
from aura.infrastructure.persistence import journal as journal_module


class _StubAgent:
    """Minimal AgentSession-shaped object exposing ``state`` — all /stats needs."""

    def __init__(self, state: LoopState, *, config: object | None = None) -> None:
        self.state = state
        self.config = config


def test_stats_command_owned_by_capabilities_module() -> None:
    assert StatsCommand.__module__ == "aura.application.commands.stats"


def _ai(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    total: int | None = None,
    model: str = "",
) -> AIMessage:
    msg = AIMessage(content="ok")
    computed_total = total if total is not None else input_tokens + output_tokens
    usage_meta: UsageMetadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": computed_total,
    }
    msg.usage_metadata = usage_meta
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
    agent: Any = _StubAgent(LoopState())
    # deliberately off-type arg to exercise path
    out = await StatsCommand().handle("", agent)
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
        agent: Any = _StubAgent(state)
        # deliberately off-type arg to exercise path
        out = await StatsCommand().handle("", agent)

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
        agent: Any = _StubAgent(state)
        # deliberately off-type arg to exercise path
        out = await StatsCommand().handle("", agent)

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
    agent: Any = _StubAgent(state)  # config=None → no log path discoverable
    # deliberately off-type arg to exercise path
    out = await StatsCommand().handle("7d", agent)
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
        agent: Any = _StubAgent(state)
        # deliberately off-type arg to exercise path
        out = await StatsCommand().handle("7d", agent)
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
        agent: Any = _StubAgent(state)
        # deliberately off-type arg to exercise path
        out = await StatsCommand().handle("all", agent)
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
        agent: Any = _StubAgent(state)
        # deliberately off-type arg to exercise path
        out = await StatsCommand().handle("7d", agent)
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
        agent: Any = _StubAgent(state)
        # deliberately off-type arg to exercise path
        out = await StatsCommand().handle("all", agent)
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
        agent: Any = _StubAgent(state)
        # deliberately off-type arg to exercise path
        out = await StatsCommand().handle("7d", agent)
        assert "No ``turn_usage`` events" in out.text
    finally:
        journal_module.reset()


# --------------------------------------------------------------------------
# Appended boundary coverage — mode aliases, config-path resolution, _safe_int
# coercion matrix, (unknown)-model fallback, OSError, single-model table.
# Autouse reset guards against any journal-global leak into later tests.
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _journal_isolation() -> Iterator[None]:
    """Each appended test starts and ends with no module-global journal path."""
    journal_module.reset()
    yield
    journal_module.reset()


@dataclass(frozen=True)
class _StubLog:
    """Minimal ``LogConfig``-shaped holder exposing the single ``path`` field."""

    path: str


@dataclass(frozen=True)
class _StubConfig:
    """Minimal ``AuraConfig``-shaped holder so ``cfg.log.path`` resolves."""

    log: _StubLog


@pytest.mark.parametrize("alias", ["7", "week", "7D", " 7d "])
@pytest.mark.asyncio
async def test_stats_7d_aliases_route_to_historical(
    alias: str, tmp_path: Path,
) -> None:
    """``/stats`` 7-day window must accept all documented spellings, else a
    user typing ``week`` silently falls through to the session view."""
    journal_module.configure(tmp_path / "missing.jsonl")
    agent: Any = _StubAgent(LoopState())
    out = await StatsCommand().handle(alias, agent)
    # Hits _historical → journal-missing branch (file never written).
    assert "not found yet" in out.text


@pytest.mark.parametrize("alias", ["all", "all-time", "alltime", "  ALL  "])
@pytest.mark.asyncio
async def test_stats_all_aliases_route_to_historical(
    alias: str, tmp_path: Path,
) -> None:
    """All-time spellings must reach the journal replay, not the session slot."""
    journal_module.configure(tmp_path / "missing.jsonl")
    agent: Any = _StubAgent(LoopState())
    out = await StatsCommand().handle(alias, agent)
    assert "not found yet" in out.text


@pytest.mark.parametrize("arg", ["", "garbage", "7days", "month", "0", "-1"])
@pytest.mark.asyncio
async def test_stats_unknown_arg_falls_back_to_session_view(arg: str) -> None:
    """Anything that is not a recognised window keyword must degrade to the
    current-session view, never raise — the slash arg is untrusted free text."""
    agent: Any = _StubAgent(LoopState())
    out = await StatsCommand().handle(arg, agent)
    assert out.handled is True
    assert "No usage recorded yet" in out.text


@pytest.mark.asyncio
async def test_stats_history_uses_config_log_path_when_no_live_journal(
    tmp_path: Path,
) -> None:
    """With no live journal but a configured ``log.path``, replay must read
    that file — config is the fallback source of truth for history."""
    log = tmp_path / "from-config.jsonl"
    _seed_turn_usage(
        log, ts=time.time(), model="cfg-model",
        input_tokens=120, output_tokens=8,
    )
    agent: Any = _StubAgent(
        LoopState(), config=_StubConfig(_StubLog(str(log))),
    )
    out = await StatsCommand().handle("all", agent)
    assert "cfg-model" in out.text
    assert "120" in out.text


@pytest.mark.parametrize("blank_path", ["", "   ", "\t"])
@pytest.mark.asyncio
async def test_stats_history_blank_config_path_is_no_journal(
    blank_path: str,
) -> None:
    """A whitespace-only ``log.path`` is not a usable journal — must surface
    the 'no journal configured' hint, not attempt to open ``Path('')``."""
    agent: Any = _StubAgent(
        LoopState(), config=_StubConfig(_StubLog(blank_path)),
    )
    out = await StatsCommand().handle("7d", agent)
    assert "No journal configured" in out.text


@pytest.mark.asyncio
async def test_stats_history_open_failure_is_treated_as_empty(
    tmp_path: Path,
) -> None:
    """If the journal path resolves to something unreadable (here: a directory,
    so ``open`` raises OSError), replay must degrade to the empty-window hint
    rather than propagate the OS error to the REPL."""
    a_dir = tmp_path / "is-a-directory.jsonl"
    a_dir.mkdir()
    agent: Any = _StubAgent(
        LoopState(), config=_StubConfig(_StubLog(str(a_dir))),
    )
    out = await StatsCommand().handle("all", agent)
    assert "No ``turn_usage`` events" in out.text


@pytest.mark.asyncio
async def test_stats_history_missing_model_renders_as_unknown(
    tmp_path: Path,
) -> None:
    """A turn_usage event lacking a model field must aggregate under a stable
    ``(unknown)`` bucket, never crash key lookup or drop the turn."""
    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": time.time(), "event": "turn_usage",
            "input_tokens": 30, "output_tokens": 4,
        }) + "\n")
    agent: Any = _StubAgent(LoopState())
    out = await StatsCommand().handle("all", agent)
    assert "(unknown)" in out.text
    assert "30" in out.text


@pytest.mark.parametrize("model_field", ["", "   ", 123, None])
@pytest.mark.asyncio
async def test_stats_history_blank_or_nonstr_model_is_unknown(
    model_field: object, tmp_path: Path,
) -> None:
    """Empty, whitespace, or non-string model values are all untrustworthy and
    must collapse into the single ``(unknown)`` bucket — a numeric ``model``
    from a buggy provider must not become a table row label."""
    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": time.time(), "event": "turn_usage", "model": model_field,
            "input_tokens": 11, "output_tokens": 2,
        }) + "\n")
    agent: Any = _StubAgent(LoopState())
    out = await StatsCommand().handle("all", agent)
    assert "(unknown)" in out.text


@pytest.mark.asyncio
async def test_stats_history_single_model_omits_total_row(
    tmp_path: Path,
) -> None:
    """With exactly one model the per-model row IS the total, so a redundant
    TOTAL row would be noise — it must be suppressed."""
    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    _seed_turn_usage(
        log, ts=time.time(), model="solo",
        input_tokens=100, output_tokens=10, cache_read_tokens=5,
    )
    agent: Any = _StubAgent(LoopState())
    out = await StatsCommand().handle("all", agent)
    assert "solo" in out.text
    assert "TOTAL" not in out.text
    assert "(1 turn)" in out.text  # singular title, no trailing 's'


@pytest.mark.parametrize(
    ("input_field", "output_field", "expected_input"),
    [
        (10.7, 2.9, "10"),        # float truncates toward zero
        (True, False, "0"),       # bool rejected so 'true' never becomes 1
        ("500", "x", "0"),        # numeric-looking string is still not int
        (None, None, "0"),        # missing values coerce to zero
    ],
)
@pytest.mark.asyncio
async def test_stats_history_token_coercion_matrix(
    input_field: object,
    output_field: object,
    expected_input: str,
    tmp_path: Path,
) -> None:
    """Per-turn token counts arrive from JSON of varying fidelity; coercion
    must truncate floats, reject bools, and zero out non-numerics — never
    inflate totals from a stray ``true`` or string."""
    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": time.time(), "event": "turn_usage", "model": "m",
            "input_tokens": input_field, "output_tokens": output_field,
        }) + "\n")
    agent: Any = _StubAgent(LoopState())
    out = await StatsCommand().handle("all", agent)
    # The single data row is "m  <turns>  <input>  <output> ..."; split on the
    # 2-space gutter to read the input cell exactly (index 2: model, turns, in).
    data_row = next(
        line for line in out.text.splitlines() if line.startswith("m ")
    )
    cells = [c.strip() for c in data_row.split("  ") if c.strip()]
    assert cells[2] == expected_input


@pytest.mark.asyncio
async def test_stats_history_non_numeric_ts_is_not_filtered(
    tmp_path: Path,
) -> None:
    """A 7d cutoff only applies to numeric timestamps; an event whose ``ts`` is
    a string (legacy/corrupt) must still be counted, not silently dropped."""
    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": "not-a-timestamp", "event": "turn_usage", "model": "legacy",
            "input_tokens": 42, "output_tokens": 3,
        }) + "\n")
    agent: Any = _StubAgent(LoopState())
    out = await StatsCommand().handle("7d", agent)
    assert "legacy" in out.text
    assert "42" in out.text


@pytest.mark.asyncio
async def test_stats_history_replay_is_idempotent(tmp_path: Path) -> None:
    """Replay is a pure read over the journal — invoking ``/stats all`` twice
    on an unchanged file must produce byte-identical output (no accumulation
    into per-model state across calls)."""
    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    _seed_turn_usage(
        log, ts=time.time(), model="a", input_tokens=100, output_tokens=10,
    )
    _seed_turn_usage(
        log, ts=time.time(), model="b", input_tokens=200, output_tokens=20,
    )
    agent: Any = _StubAgent(LoopState())
    first = await StatsCommand().handle("all", agent)
    second = await StatsCommand().handle("all", agent)
    assert first.text == second.text
    assert "TOTAL" in first.text  # two models → total row present
