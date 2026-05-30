"""Compact thresholds — single source of truth for callers and tests."""

from __future__ import annotations

from enum import StrEnum


class CompactionTrigger(StrEnum):
    """Trigger taxonomy for ``compact_event`` records."""

    microcompact = "microcompact"
    reactive = "reactive"
    auto = "auto"
    manual = "manual"


KEEP_LAST_N_TURNS = 3

# Sentinel -1 → derive from model context window. 0 disables auto-compact.
AUTO_COMPACT_THRESHOLD = -1

AUTO_COMPACT_HEADROOM_TOKENS = 13_000


def auto_compact_threshold_for(model_spec: str) -> int:
    """``ctx_window - AUTO_COMPACT_HEADROOM_TOKENS``, floored at 1000."""
    from aura.infrastructure.llm import get_context_window

    window = get_context_window(model_spec)
    threshold = window - AUTO_COMPACT_HEADROOM_TOKENS
    return max(1_000, threshold)


# Microcompact knobs — view-only payload compression; 5/3 fires in realistic sessions.
MICROCOMPACT_TRIGGER_PAIRS = 5
MICROCOMPACT_KEEP_RECENT = 3
MICROCOMPACT_CLEAR_MARKER = "[Old tool result content cleared]"

# High-volume / low-signal I/O — safe to compress. Excluded: subagent
# lifecycle, todo_write, skill, ask_user_question, plan-mode, MCP reads.
MICROCOMPACT_COMPACTABLE_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "write_file",
    "edit_file",
    "bash",
    "bash_background",
    "grep",
    "glob",
    "web_fetch",
    "web_search",
})
