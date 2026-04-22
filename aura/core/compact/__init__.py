"""Conversation compaction — public surface.

Typical usage flows through :meth:`aura.core.agent.Agent.compact` (which
wraps :func:`run_compact`). Exported here so callers can type-annotate
against :class:`CompactResult` without reaching into the submodule.
"""

from __future__ import annotations

from aura.core.compact.compact import CompactResult, CompactSource, run_compact
from aura.core.compact.constants import (
    AUTO_COMPACT_THRESHOLD,
    KEEP_LAST_N_TURNS,
    MAX_FILES_TO_RESTORE,
    MAX_TOKENS_PER_FILE,
    SUMMARY_BUDGET,
)
from aura.core.compact.prompt import SUMMARY_SYSTEM, SUMMARY_USER_PREFIX

__all__ = [
    "AUTO_COMPACT_THRESHOLD",
    "CompactResult",
    "CompactSource",
    "KEEP_LAST_N_TURNS",
    "MAX_FILES_TO_RESTORE",
    "MAX_TOKENS_PER_FILE",
    "SUMMARY_BUDGET",
    "SUMMARY_SYSTEM",
    "SUMMARY_USER_PREFIX",
    "run_compact",
]
