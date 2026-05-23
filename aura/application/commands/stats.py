"""``/stats`` — show token usage.

Modes: bare (current session via ``state.slots.token_stats``), ``7d``
(last week via ``turn_usage`` journal replay), ``all`` (full replay).
The journal path resolves from ``journal._path`` (live) →
``agent.config.log.path`` (configured default). Missing files yield an
empty aggregate with a helpful hint rather than an error.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from aura.application.commands.types import CommandResult, CommandSource

if TYPE_CHECKING:
    from aura.core.agent import Agent


def _fmt(n: int) -> str:
    """Render ``n`` with thousands separators."""
    return f"{n:,}"


class StatsCommand:
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

    def _current_session(self, agent: Agent) -> CommandResult:
        stats = agent.state.slots.token_stats
        if stats.turn_count == 0:
            return CommandResult(
                handled=True,
                kind="print",
                text=(
                    "No usage recorded yet — /stats becomes useful "
                    "once the agent has completed at least one turn."
                ),
            )

        turns = stats.turn_count
        total_input = stats.total_input_tokens
        total_output = stats.total_output_tokens
        total_cache = stats.total_cache_read_tokens
        last_input = stats.last_input_tokens
        last_output = stats.last_output_tokens
        last_cache = stats.last_cache_read_tokens

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

        cutoff: datetime | None = (
            datetime.now(UTC) - timedelta(days=days) if days is not None else None
        )
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
    """Coerce a journal field to int; fall back to 0 on type drift.

    ``bool`` is rejected explicitly so a misshapen ``true`` doesn't silently
    become 1.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _resolve_journal_path(agent: Agent) -> Path | None:
    """Live journal path → config default → None."""
    from aura.infrastructure.persistence import journal as journal_mod

    live = getattr(journal_mod, "_path", None)
    if isinstance(live, Path):
        return live
    cfg = agent.config
    if cfg is None:
        return None
    raw_path = cfg.log.path
    if not raw_path.strip():
        return None
    return Path(raw_path).expanduser()


def _read_turn_usage(path: Path) -> Iterator[dict[str, object]]:
    """Stream ``turn_usage`` events out of a JSONL journal; skip bad lines."""
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
    """Render per-model aggregates as an aligned plain-text table.

    ASCII (not rich.Table) — ``aura/core/**`` is UI-framework-free; the
    REPL view renderer wraps the string in a rich.Panel.
    """
    headers = ("Model", "Turns", "Input", "Output", "Cache read", "Total")
    rows = sorted(
        per_model.items(),
        key=lambda kv: -(kv[1]["input"] + kv[1]["output"]),
    )
    grand = {"input": 0, "output": 0, "cache_read": 0}
    body: list[tuple[str, str, str, str, str, str]] = []
    for model, row in rows:
        grand["input"] += row["input"]
        grand["output"] += row["output"]
        grand["cache_read"] += row["cache_read"]
        body.append((
            model, _fmt(row["turns"]), _fmt(row["input"]),
            _fmt(row["output"]), _fmt(row["cache_read"]),
            _fmt(row["input"] + row["output"]),
        ))
    show_total_row = len(rows) > 1
    if show_total_row:
        body.append((
            "TOTAL", _fmt(total_turns),
            _fmt(grand["input"]), _fmt(grand["output"]),
            _fmt(grand["cache_read"]), _fmt(grand["input"] + grand["output"]),
        ))

    widths = [len(h) for h in headers]
    for body_row in body:
        for i, cell in enumerate(body_row):
            widths[i] = max(widths[i], len(cell))

    def _format_row(rendered: tuple[str, ...]) -> str:
        cells = [rendered[0].ljust(widths[0])]
        cells.extend(rendered[i].rjust(widths[i]) for i in range(1, 6))
        return "  ".join(cells)

    title = (
        f"Token usage — {label} ({total_turns} turn"
        f"{'s' if total_turns != 1 else ''})"
    )
    separator = "─" * (sum(widths) + 2 * (len(widths) - 1))
    lines = [title, "", _format_row(headers), separator]
    if show_total_row:
        lines.extend(_format_row(r) for r in body[:-1])
        lines.append(separator)
        lines.append(_format_row(body[-1]))
    else:
        lines.extend(_format_row(r) for r in body)
    return "\n".join(lines)
