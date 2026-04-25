"""``/stats`` — show token usage (current session + historical replay).

Three modes, picked by argument:

- ``/stats`` (no arg) — current session totals, read from
  ``state.custom["_token_stats"]`` populated by ``make_usage_tracking_hook``.
  Always works (no journal opt-in required).
- ``/stats 7d`` — last 7 days, replays ``turn_usage`` journal events.
- ``/stats all`` — all-time totals, full journal replay.

The ``turn_usage`` journal event has been emitted unconditionally since
v0.13 (see ``aura/core/hooks/budget.py::make_usage_tracking_hook``), so
historical data is on disk for any session run since then. The replay path
discovers the journal file via:

1. ``journal._path`` if currently configured (the live ``--log`` / config
   path),
2. ``agent._config.log.path`` expanded as ``~/.aura/logs/events.jsonl``
   default,
3. session log dir scan (``~/.aura/sessions/*.jsonl``-style) — best-effort.

Missing files are not errors — replay returns an empty aggregate and the
view tells the user "enable ``--log`` for historical stats."
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from aura.core.commands.types import CommandResult, CommandSource

if TYPE_CHECKING:
    from aura.core.agent import Agent


def _fmt(n: int) -> str:
    """Render ``n`` with thousands separators — ``1234567`` → ``"1,234,567"``."""
    return f"{n:,}"


class StatsCommand:
    """``/stats`` — token usage for the current session OR historical replay."""

    name = "/stats"
    description = "show token usage (current session, 7d, or all-time)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[7d|all]"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        mode = arg.strip().lower()
        if mode in {"7d", "7", "week"}:
            return self._historical(agent, days=7, label="last 7 days")
        if mode in {"all", "all-time", "alltime"}:
            return self._historical(agent, days=None, label="all-time")
        return self._current_session(agent)

    # ------------------------------------------------------------------
    # Current session — original v0.13 behaviour, untouched
    # ------------------------------------------------------------------

    def _current_session(self, agent: Agent) -> CommandResult:
        stats = agent._state.custom.get("_token_stats")
        if not isinstance(stats, dict) or not stats:
            return CommandResult(
                handled=True,
                kind="print",
                text=(
                    "No usage recorded yet — /stats becomes useful "
                    "once the agent has completed at least one turn."
                ),
            )

        turns = int(stats.get("turn_count", 0))
        total_input = int(stats.get("total_input_tokens", 0))
        total_output = int(stats.get("total_output_tokens", 0))
        total_cache = int(stats.get("total_cache_read_tokens", 0))
        last_input = int(stats.get("last_input_tokens", 0))
        last_output = int(stats.get("last_output_tokens", 0))
        last_cache = int(stats.get("last_cache_read_tokens", 0))

        grand_total = total_input + total_output

        lines = [
            f"Session tokens — {turns} turn{'s' if turns != 1 else ''}:",
            "",
            f"  Input         {_fmt(total_input):>12}   "
            f"(cache read: {_fmt(total_cache)})",
            f"  Output        {_fmt(total_output):>12}",
            f"  Total         {_fmt(grand_total):>12}",
            "",
            (
                f"  Last turn     in {_fmt(last_input)} / "
                f"out {_fmt(last_output)} / cache {_fmt(last_cache)}"
            ),
            "",
            "  /stats 7d   — last 7 days from journal",
            "  /stats all  — all-time from journal",
        ]
        return CommandResult(handled=True, kind="view", text="\n".join(lines))

    # ------------------------------------------------------------------
    # Historical replay — reads ``turn_usage`` events from the journal
    # ------------------------------------------------------------------

    def _historical(
        self, agent: Agent, *, days: int | None, label: str,
    ) -> CommandResult:
        journal_path = _resolve_journal_path(agent)
        if journal_path is None:
            return CommandResult(
                handled=True,
                kind="print",
                text=(
                    "No journal configured — historical /stats requires "
                    "``--log`` at startup or ``log.enabled: true`` in "
                    "settings.json."
                ),
            )
        if not journal_path.exists():
            return CommandResult(
                handled=True,
                kind="print",
                text=(
                    f"Journal file {journal_path} not found yet — "
                    f"historical /stats becomes useful after the first "
                    f"turn that writes a ``turn_usage`` event there."
                ),
            )

        cutoff: datetime | None = None
        if days is not None:
            cutoff = datetime.now(UTC) - timedelta(days=days)

        # Aggregate input/output/cache tokens grouped by model name. We
        # accumulate into a defaultdict so even an unseen model name on
        # the first row works without an explicit setdefault dance.
        per_model: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0, "cache_read": 0, "turns": 0},
        )
        total_turns = 0

        for event in _read_turn_usage(journal_path):
            ts = event.get("ts")
            if isinstance(ts, (int, float)) and cutoff is not None:
                event_dt = datetime.fromtimestamp(float(ts), tz=UTC)
                if event_dt < cutoff:
                    continue

            model_raw = event.get("model")
            model = (
                model_raw.strip() if isinstance(model_raw, str) and model_raw.strip()
                else "(unknown)"
            )
            row = per_model[model]
            row["input"] += _safe_int(event.get("input_tokens"))
            row["output"] += _safe_int(event.get("output_tokens"))
            row["cache_read"] += _safe_int(event.get("cache_read_tokens"))
            row["turns"] += 1
            total_turns += 1

        if total_turns == 0:
            return CommandResult(
                handled=True,
                kind="print",
                text=(
                    f"No ``turn_usage`` events in the {label} window — "
                    f"either the journal is empty, the cutoff is before "
                    f"any recorded turn, or v0.13 hadn't shipped when "
                    f"these sessions ran."
                ),
            )

        return CommandResult(
            handled=True,
            kind="view",
            text=_render_history_table(per_model, total_turns, label),
        )


def _safe_int(value: object) -> int:
    """Coerce a journal field to int, falling back to 0 on type drift.

    The journal is JSON; in practice numeric fields land as ``int`` or
    ``float``. But the file is a long-lived audit log and a future
    schema bump might change shapes — degrading to 0 is safer than
    crashing the whole replay on one weird row.
    """
    if isinstance(value, bool):
        # bool is a subtype of int in Python; explicitly reject so a
        # ``"input_tokens": true`` (which would never happen, but) doesn't
        # silently become 1.
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _resolve_journal_path(agent: Agent) -> Path | None:
    """Pick the journal file to scan.

    Precedence:
    1. The live ``aura.core.persistence.journal._path`` if non-None — that's
       what the running session is writing to right now, so its history is
       most relevant.
    2. ``agent._config.log.path`` expanded — covers the case where the
       command is invoked outside of a ``--log`` session but a config
       default path exists.
    3. ``None`` if no source produces a path. ``handle`` reports this back
       as a friendly hint instead of pretending to have data.
    """
    from aura.core.persistence import journal as journal_mod

    live = getattr(journal_mod, "_path", None)
    if isinstance(live, Path):
        return live

    cfg = getattr(agent, "_config", None)
    if cfg is None:
        return None
    log_cfg = getattr(cfg, "log", None)
    if log_cfg is None:
        return None
    raw_path = getattr(log_cfg, "path", None)
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    return Path(raw_path).expanduser()


def _read_turn_usage(path: Path) -> Iterator[dict[str, object]]:
    """Stream ``turn_usage`` events out of a JSONL journal file.

    Tolerates malformed lines (truncated mid-write on a crash, foreign
    JSON, blank rows) — they're silently skipped rather than aborting
    the whole replay. The journal is an audit log; one bad row in the
    middle should not invalidate the surrounding data.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("event") != "turn_usage":
                    continue
                yield payload
    except OSError:
        return


def _render_history_table(
    per_model: dict[str, dict[str, int]],
    total_turns: int,
    label: str,
) -> str:
    """Render per-model aggregates as a plain-text aligned table.

    Built by hand instead of via ``rich.Table`` because ``aura/core/**``
    must not import UI frameworks (architectural invariant enforced by
    ``tests/test_import_boundaries.py``). The view-kind handler at
    ``aura/cli/repl.py::_render_view`` wraps this string in a rich.Panel,
    so we get the bordered modal feel even though the table itself is
    pure ASCII. Column widths are computed from the actual content so
    the table breathes whether the user runs one tiny session or 1000.
    """
    headers = ("Model", "Turns", "Input", "Output", "Cache read", "Total")
    # Sort by total tokens desc so the heaviest-cost model is on top —
    # matches operator intuition when the question is "what cost me money".
    rows = sorted(
        per_model.items(),
        key=lambda kv: -(kv[1]["input"] + kv[1]["output"]),
    )
    grand = {"input": 0, "output": 0, "cache_read": 0}
    body: list[tuple[str, str, str, str, str, str]] = []
    for model, row in rows:
        total = row["input"] + row["output"]
        grand["input"] += row["input"]
        grand["output"] += row["output"]
        grand["cache_read"] += row["cache_read"]
        body.append((
            model,
            _fmt(row["turns"]),
            _fmt(row["input"]),
            _fmt(row["output"]),
            _fmt(row["cache_read"]),
            _fmt(total),
        ))
    show_total_row = len(rows) > 1
    if show_total_row:
        body.append((
            "TOTAL",
            _fmt(total_turns),
            _fmt(grand["input"]),
            _fmt(grand["output"]),
            _fmt(grand["cache_read"]),
            _fmt(grand["input"] + grand["output"]),
        ))

    # Compute per-column max width so values + headers line up. ``Model``
    # is left-aligned, every numeric column right-aligned. Renamed loop
    # variable to ``body_row`` so it doesn't shadow the dict-typed
    # ``row`` from the aggregate-building loop above (mypy catches the
    # collision otherwise).
    widths = [len(h) for h in headers]
    for body_row in body:
        for i, cell in enumerate(body_row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    def _format_row(rendered: tuple[str, ...]) -> str:
        cells = [
            rendered[0].ljust(widths[0]),
            *(rendered[i].rjust(widths[i]) for i in range(1, 6)),
        ]
        return "  ".join(cells)

    title = (
        f"Token usage — {label} ({total_turns} turn"
        f"{'s' if total_turns != 1 else ''})"
    )
    separator = "─" * (sum(widths) + 2 * (len(widths) - 1))
    lines = [
        title,
        "",
        _format_row(headers),
        separator,
    ]
    if show_total_row:
        lines.extend(_format_row(r) for r in body[:-1])
        lines.append(separator)
        lines.append(_format_row(body[-1]))
    else:
        lines.extend(_format_row(r) for r in body)
    return "\n".join(lines)
