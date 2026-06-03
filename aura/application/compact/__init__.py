"""Conversation compaction surface."""

from __future__ import annotations

from aura.application.compact.compactor import Compactor
from aura.application.compact.constants import (
    AUTO_COMPACT_THRESHOLD,
    KEEP_LAST_N_TURNS,
    MICROCOMPACT_CLEAR_MARKER,
    MICROCOMPACT_COMPACTABLE_TOOLS,
    MICROCOMPACT_KEEP_RECENT,
    MICROCOMPACT_TRIGGER_PAIRS,
    CompactionTrigger,
)
from aura.application.compact.microcompact import (
    MicrocompactPolicy,
    MicrocompactResult,
    ToolPair,
    apply_clear,
    apply_microcompact,
    find_tool_pairs,
    select_clear_ids,
)
from aura.application.compact.prompt import SUMMARY_SYSTEM, SUMMARY_USER_PREFIX
from aura.application.compact.reactive import run_compact
from aura.application.compact.result_types import CompactResult, CompactSource

__all__ = [
    "AUTO_COMPACT_THRESHOLD",
    "CompactResult",
    "CompactSource",
    "CompactionTrigger",
    "Compactor",
    "KEEP_LAST_N_TURNS",
    "MICROCOMPACT_CLEAR_MARKER",
    "MICROCOMPACT_COMPACTABLE_TOOLS",
    "MICROCOMPACT_KEEP_RECENT",
    "MICROCOMPACT_TRIGGER_PAIRS",
    "MicrocompactPolicy",
    "MicrocompactResult",
    "SUMMARY_SYSTEM",
    "SUMMARY_USER_PREFIX",
    "ToolPair",
    "apply_clear",
    "apply_microcompact",
    "find_tool_pairs",
    "run_compact",
    "select_clear_ids",
]
