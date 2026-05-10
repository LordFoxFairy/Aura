"""Conversation compaction — public surface.

Typical usage flows through :meth:`aura.core.agent.Agent.compact` (which
wraps :func:`run_compact`). Exported here so callers can type-annotate
against :class:`CompactResult` without reaching into the submodule.

The microcompact surface (:func:`apply_microcompact` +
:class:`MicrocompactPolicy` + :class:`MicrocompactResult`) is a separate,
pure-function layer wired into ``_invoke_model`` between ``Context.build``
and ``self._bound.ainvoke``. It compresses old tool_use/tool_result pair
payloads in the *outgoing* prompt only — stored history is untouched.

The :class:`Compactor` Protocol (Phase 1 §3.3) collapses the three
historical compaction call sites — microcompact, reactive, auto — into
one named interface. :class:`aura.core.compact.legacy_adapter.LegacyCompactor`
satisfies it today by delegating to :func:`apply_microcompact` and
:func:`run_compact`. Phase 4 replaces the adapter with a first-class
implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.messages import BaseMessage

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
from aura.schemas.state import LoopSlots


@runtime_checkable
class Compactor(Protocol):
    """Phase 1 §3.3 — unified compaction interface.

    Three historical call sites collapse to three named methods on one
    object:

    - :meth:`microcompact` — view-only payload compression, called per
      turn between ``Context.build`` and ``model.ainvoke``. Stored
      history is untouched; only the outgoing prompt is trimmed.
    - :meth:`reactive` — full summary compaction triggered when the
      model raises a context-overflow on ``ainvoke``. Mutates stored
      history in place.
    - :meth:`auto` — post-turn threshold check; runs a summary
      compaction when ``total_tokens_used`` crosses the model-aware
      auto-compact threshold. Returns ``None`` when the threshold was
      not exceeded (so the loop can distinguish "no work to do" from
      "ran and produced these stats").

    Phase 1 ships
    :class:`aura.core.compact.legacy_adapter.LegacyCompactor` as the only
    implementation; Phase 4 introduces ``CompactConfig`` + a first-class
    implementation. The Protocol is async on every method so Phase 4 can
    swap a streaming implementation in without changing the call sites.
    """

    async def microcompact(
        self,
        messages: list[BaseMessage],
        slots: LoopSlots,
    ) -> list[BaseMessage]: ...

    async def reactive(
        self,
        history: list[BaseMessage],
        slots: LoopSlots,
    ) -> CompactResult: ...

    async def auto(
        self,
        history: list[BaseMessage],
        slots: LoopSlots,
        *,
        model: str,
    ) -> CompactResult | None: ...


__all__ = [
    "AUTO_COMPACT_THRESHOLD",
    "CompactResult",
    "CompactSource",
    "Compactor",
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
