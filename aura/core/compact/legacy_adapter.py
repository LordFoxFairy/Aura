"""LegacyCompactor — Phase 1 adapter satisfying the :class:`Compactor` Protocol.

Phase 1 introduced a unified :class:`aura.core.compact.Compactor` Protocol so
the loop has one named entry point per compaction trigger. This module
ships the only implementation — a thin adapter delegating to the existing
free functions in :mod:`aura.core.compact.compact` and
:mod:`aura.core.compact.microcompact`. Phase 4 replaces it with a
first-class implementation backed by ``CompactConfig``.

The adapter is stateless on its own: every mutable piece of state it
touches lives on the agent it was constructed with, or on
``state.slots.consecutive_compact_failures`` (the auto-compact circuit
breaker counter). Refactoring agent.py / compact.py internals is
explicitly out of scope here — Phase 4's territory.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage

from aura.core.compact.compact import CompactResult
from aura.core.compact.microcompact import (
    MicrocompactPolicy,
    apply_microcompact,
)
from aura.core.persistence import journal
from aura.schemas.state import LoopSlots

if TYPE_CHECKING:
    from aura.core.agent import Agent


# Circuit-breaker threshold: three consecutive auto-compact failures
# disable subsequent auto-firings for the session. Manual /compact
# bypasses this counter (different code path), and a successful
# auto-compact resets it. Mirrors the constant baked into Agent's
# pre-Phase-1 inline implementation — kept here so Phase 4 can move
# it onto ``CompactConfig`` without touching call sites.
_AUTO_COMPACT_CIRCUIT_BREAKER_LIMIT = 3


class LegacyCompactor:
    """Phase 1 adapter — delegates each Compactor method to today's free fns.

    Holds a back-reference to the owning :class:`aura.core.agent.Agent`
    because the existing :func:`run_compact` function reaches into agent
    internals (``_storage``, ``_context``, ``_state``, ``_model``, ``_cwd``,
    ``_hooks``, …) to perform the summary turn. That coupling is the
    artefact Phase 4 will dissolve; Phase 1 simply names the call site.

    Microcompact policy is supplied at construction (mirroring how today's
    AgentLoop receives it). ``None`` disables the microcompact path
    entirely — :meth:`microcompact` becomes a pass-through.

    Session id + turn provider are threaded so :meth:`microcompact`
    can stamp the existing ``microcompact_applied`` journal event with
    the same ``session`` + ``turn`` fields it carried before Phase 1.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        microcompact_policy: MicrocompactPolicy | None,
        session_id: str,
        turn_provider: Callable[[], int],
    ) -> None:
        self._agent = agent
        self._microcompact_policy = microcompact_policy
        self._session_id = session_id
        self._turn_provider = turn_provider

    async def microcompact(
        self,
        messages: list[BaseMessage],
        slots: LoopSlots,  # noqa: ARG002 — slots reserved for Phase 4
    ) -> list[BaseMessage]:
        """Apply view-only payload compression to ``messages``.

        Returns the (possibly compressed) message list. When the policy
        is ``None`` (feature disabled) or no pairs were cleared, returns
        the input list unchanged. Emits the same ``microcompact_applied``
        journal event the loop emitted pre-Phase-1 so observability is
        unchanged.
        """
        if self._microcompact_policy is None:
            return messages
        result = apply_microcompact(messages, self._microcompact_policy)
        if result.cleared_pair_count == 0:
            return messages
        journal.write(
            "microcompact_applied",
            session=self._session_id,
            turn=self._turn_provider(),
            cleared_pair_count=result.cleared_pair_count,
            cleared_tool_call_ids=list(result.cleared_tool_call_ids),
            cleared_positions=[
                [p.ai_idx, p.tool_idx] for p in result.cleared_pairs
            ],
        )
        return result.messages

    async def reactive(
        self,
        history: list[BaseMessage],
        slots: LoopSlots,  # noqa: ARG002 — slots reserved for Phase 4
    ) -> CompactResult:
        """Run a full summary compaction in response to context-overflow.

        Mirrors the pre-Phase-1 ``Agent._reactive_compact_callback``:
        delegate to ``Agent.compact(source="reactive")`` (the public
        surface, so test patches that replace ``Agent.compact`` keep
        working), then refresh the in-memory ``history`` list in place
        so the loop's local reference points at the post-compact
        transcript without a re-assignment.
        """
        result = await self._agent.compact(source="reactive")
        history[:] = self._agent._storage.load(self._agent.session_id)
        return result

    async def auto(
        self,
        history: list[BaseMessage],
        slots: LoopSlots,
        *,
        model: str,  # noqa: ARG002 — model is a Phase 4 hook for window-aware policy
    ) -> CompactResult | None:
        """Post-turn auto-compact threshold check + run.

        Returns the :class:`CompactResult` when compaction ran;
        ``None`` when the threshold was not exceeded OR the circuit
        breaker disabled the trigger. Mirrors the pre-Phase-1 inline
        block in ``Agent.astream`` (at the post-save site) so
        behavior — including the journal events emitted on each branch —
        is unchanged.

        On failure the consecutive-failures counter on
        ``slots.consecutive_compact_failures`` is incremented and the
        exception is re-raised; on success it is reset to 0.
        """
        threshold = self._agent._effective_auto_compact_threshold()
        if threshold <= 0:
            return None
        used = self._agent._state.total_tokens_used
        used_estimator = used == 0
        if used_estimator:
            # Defer to Agent's estimator to preserve parity (it includes
            # pinned-prefix tokens — system prompt, memory, tool schemas
            # — which a naive ``sum(estimate_message_tokens)`` over
            # history alone would miss).
            used = self._agent._estimate_history_tokens(history)
        if used <= threshold:
            return None
        failures = slots.consecutive_compact_failures
        if failures >= _AUTO_COMPACT_CIRCUIT_BREAKER_LIMIT:
            journal.write(
                "auto_compact_skipped_circuit_breaker",
                session=self._agent.session_id,
                tokens=used,
                threshold=threshold,
                consecutive_failures=failures,
                used_estimator=used_estimator,
            )
            return None
        journal.write(
            "auto_compact_triggered",
            session=self._agent.session_id,
            tokens=used,
            threshold=threshold,
            used_estimator=used_estimator,
        )
        try:
            # Route through ``Agent.compact`` (the public surface) so
            # test fixtures patching ``Agent.compact`` continue to
            # observe the auto path. The Agent method is a thin wrapper
            # over :func:`run_compact`.
            result = await self._agent.compact(source="auto")
        except Exception as exc:  # noqa: BLE001
            new_failures = failures + 1
            self._agent._state.slots = dataclasses.replace(
                self._agent._state.slots,
                consecutive_compact_failures=new_failures,
            )
            journal.write(
                "auto_compact_failed",
                session=self._agent.session_id,
                error=str(exc),
                consecutive_failures=new_failures,
            )
            raise
        else:
            self._agent._state.slots = dataclasses.replace(
                self._agent._state.slots,
                consecutive_compact_failures=0,
            )
            return result
