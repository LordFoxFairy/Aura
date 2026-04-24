"""``/stats`` — show current-session token usage (V13-T2A, v0.13 MVP).

Scope: **current session only** — reads ``state.custom["_token_stats"]``
populated by ``make_usage_tracking_hook``. Historical aggregation across
sessions is explicitly v0.14 work (the ``turn_usage`` journal event now
emits every turn so future /stats can replay from disk without data loss).

Why "current session only" for v0.13:
- Zero scope creep — no journal reader, no date bucketing, no cross-session
  dedup by model.
- Works by default — no ``--log`` opt-in required; data source is LoopState
  which is always populated by the default usage-tracking hook.
- Matches the pattern set by the status-bar rendering at ``repl.py:330``
  which also reads ``_token_stats`` directly, so /stats and the status bar
  never drift apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aura.core.commands.types import CommandResult, CommandSource

if TYPE_CHECKING:
    from aura.core.agent import Agent


def _fmt(n: int) -> str:
    """Render ``n`` with thousands separators — ``1234567`` → ``"1,234,567"``."""
    return f"{n:,}"


class StatsCommand:
    """``/stats`` — print cumulative token usage for the current session."""

    name = "/stats"
    description = "show current-session token usage"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
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

        # Fixed-width two-column layout so the numbers align cleanly under
        # prompt_toolkit's plain-text print. ``rich.Table`` would be nicer
        # visually but adds a rendering dependency on the print path that
        # currently just emits raw strings — keep it str-only for v0.13 so
        # the command matches how the rest of the REPL renders output.
        lines = [
            f"Session tokens — {turns} turn{'s' if turns != 1 else ''}:",
            "",
            f"  Input         {_fmt(total_input):>12}   (cache read: {_fmt(total_cache)})",
            f"  Output        {_fmt(total_output):>12}",
            f"  Total         {_fmt(grand_total):>12}",
            "",
            (
                f"  Last turn     in {_fmt(last_input)} / "
                f"out {_fmt(last_output)} / cache {_fmt(last_cache)}"
            ),
        ]
        return CommandResult(handled=True, kind="print", text="\n".join(lines))
