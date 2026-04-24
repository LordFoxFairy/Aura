"""Conversation compaction — public surface.

Typical usage flows through :meth:`aura.core.agent.Agent.compact` (which
wraps :func:`run_compact`). Exported here so callers can type-annotate
against :class:`CompactResult` without reaching into the submodule.

The microcompact surface (:func:`apply_microcompact` +
:class:`MicrocompactPolicy` + :class:`MicrocompactResult`) is a separate,
pure-function layer wired into ``_invoke_model`` between ``Context.build``
and ``self._bound.ainvoke``. It compresses old tool_use/tool_result pair
payloads in the *outgoing* prompt only — stored history is untouched.
"""

from __future__ import annotations

from aura.core.compact.compact import CompactResult, CompactSource, run_compact
from aura.core.compact.constants import (
    AUTO_COMPACT_THRESHOLD,
    KEEP_LAST_N_TURNS,
    MAX_FILES_TO_RESTORE,
    MAX_TOKENS_PER_FILE,
    MICROCOMPACT_CLEAR_MARKER,
    MICROCOMPACT_COMPACTABLE_TOOLS,
    MICROCOMPACT_KEEP_RECENT,
    MICROCOMPACT_TRIGGER_PAIRS,
)
from aura.core.compact.microcompact import (
    MicrocompactPolicy,
    MicrocompactResult,
    ToolPair,
    apply_clear,
    apply_microcompact,
    find_tool_pairs,
    select_clear_ids,
)
from aura.core.compact.prompt import SUMMARY_SYSTEM, SUMMARY_USER_PREFIX

__all__ = [
    "AUTO_COMPACT_THRESHOLD",
    "CompactResult",
    "CompactSource",
    "KEEP_LAST_N_TURNS",
    "MAX_FILES_TO_RESTORE",
    "MAX_TOKENS_PER_FILE",
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
