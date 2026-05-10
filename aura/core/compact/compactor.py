"""Phase 4 §5 — first-class :class:`Compactor` replacing the Phase 1 adapter.

This class collapses the four compaction call sites — microcompact,
reactive, auto, manual — onto one named object so the loop wiring is a
single pointer instead of three callback references. It mirrors the
pre-Phase-1 free-function behavior verbatim (microcompact via
:func:`apply_microcompact`, summary-turn compaction via
:func:`run_compact`); the value-add is unification + explicit trigger
tagging on every event so observability filters on a stable enum.

The class is otherwise stateless. Per-session counters live on
:attr:`aura.schemas.state.LoopSlots.consecutive_compact_failures` and
mutate via :func:`dataclasses.replace` (the slot is frozen). That's the
same pattern the Phase 1 adapter used and the same pattern the rest of
the typed-slot machinery uses — see ``aura/core/hooks/budget.py`` for
the equivalent on ``token_stats``.

Spec §6 precedence rules (microcompact first, reactive only on
``PromptTooLong``, auto only post-turn, length-recovery disjoint from
reactive, circuit breaker on N consecutive failures) are NOT enforced
here — the loop is the orchestrator. This class is the *executor* each
trigger calls into; the orchestrator picks which one based on its turn
state.

Phase 4 Task 4 will swap ``aura/core/loop.py`` and ``aura/core/agent.py``
from :class:`LegacyCompactor` onto this class. Until then both
implementations coexist and Task 5 deletes the legacy adapter.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from aura.config.schema import CompactConfig
from aura.core.compact.compact import CompactResult
from aura.core.compact.constants import CompactionTrigger
from aura.core.compact.microcompact import (
    MicrocompactPolicy,
    apply_microcompact,
)
from aura.core.persistence import journal
from aura.schemas.state import LoopSlots
from aura.transport.wire import compact_event_to_wire

if TYPE_CHECKING:
    from aura.core.agent import Agent

# Type alias for the AG-UI custom-event sink. The Compactor calls this
# (when set) on every method exit with the wire-format ``compact_event``
# dict. Default is ``None`` — Phase 4 Task 4 wires a real emitter from
# ``Agent.astream``; callers / tests can pass any callable.
EventEmitter = Callable[[dict[str, Any]], None]


class Compactor:
    """Phase 4 §5 — unified executor for the four compaction triggers.

    One class with four async methods (``microcompact`` / ``reactive`` /
    ``auto`` / ``manual``); each is the call site for exactly one
    :class:`CompactionTrigger`. Construction:

    - ``agent``: back-reference to the owning :class:`aura.core.agent.Agent`.
      Phase 4 keeps the coupling because the heavy lifting in
      :func:`aura.core.compact.compact.run_compact` reaches into agent
      internals (``_storage``, ``_context``, ``_state``, ``_model``,
      ``_cwd``, ``_hooks``) — dissolving that coupling is a deeper
      refactor than this Phase covers.
    - ``config``: :class:`CompactConfig` (typically
      ``agent._config.compact``). Read for ``max_consecutive_failures``
      (circuit breaker limit). Other fields are read by the underlying
      ``run_compact`` flow via :class:`SummaryCaps`.
    - ``summary_model``: kept for spec parity. The active summary model
      is whatever the Agent's ``_model`` resolves to at call time;
      ``run_compact`` reads it via the agent reference. The constructor
      argument exists so a future implementation can swap to a
      dedicated summary model without changing call sites.
    - ``microcompact_policy``: :class:`MicrocompactPolicy` instance or
      ``None``. ``None`` disables the microcompact path entirely (the
      method becomes a pass-through). Same shape the Phase 1 adapter
      took; the loop builder owns construction.
    - ``session_id`` / ``turn_provider``: thread the session id +
      current turn through to the ``microcompact_applied`` journal
      event so observability matches pre-Phase-1.
    - ``event_emitter``: optional callable invoked on every method
      exit with the wire-format ``compact_event`` dict (see
      :func:`aura.transport.wire.compact_event_to_wire`). The CLI / SSE
      adapter wires a real emitter; tests can pass a list-appender.

    State (``consecutive_compact_failures``) lives on the typed
    :class:`LoopSlots` — frozen, ``dataclasses.replace``-d on every
    write. The class itself is stateless.
    """

    def __init__(
        self,
        *,
        agent: Agent,
        config: CompactConfig,
        summary_model: BaseChatModel,
        microcompact_policy: MicrocompactPolicy | None = None,
        session_id: str,
        turn_provider: Callable[[], int],
        event_emitter: EventEmitter | None = None,
    ) -> None:
        self._agent = agent
        self._config = config
        # ``summary_model`` is held for spec parity even though
        # ``run_compact`` reads ``agent._model`` at call time. Keeping
        # it as a bound attribute means a future refactor (decouple
        # from agent) doesn't change the constructor surface.
        self._summary_model = summary_model
        self._microcompact_policy = microcompact_policy
        self._session_id = session_id
        self._turn_provider = turn_provider
        self._event_emitter = event_emitter

    # ------------------------------------------------------------------
    # Compactor Protocol surface (Phase 1 §3.3 + Phase 4 §5)
    # ------------------------------------------------------------------

    async def microcompact(
        self,
        messages: list[BaseMessage],
        slots: LoopSlots,  # noqa: ARG002 — reserved for future per-slot policy
        trigger: CompactionTrigger = CompactionTrigger.microcompact,
    ) -> list[BaseMessage]:
        """Apply view-only payload compression to ``messages``.

        Returns the (possibly compressed) message list. ``None`` policy
        + zero cleared pairs both return the input list unchanged. The
        ``microcompact_applied`` journal event is preserved for parity
        with pre-Phase-1 observability; the new ``compact_event`` is
        emitted alongside so the Phase 4 unified shape is consistent
        across all four triggers.
        """
        before = sum(_msg_chars(m) for m in messages)
        started = time.monotonic()
        if self._microcompact_policy is None:
            self._emit_event(
                trigger=trigger,
                tokens_before=before,
                tokens_after=before,
                outcome="skipped",
                duration_ms=_elapsed_ms(started),
            )
            return messages
        result = apply_microcompact(messages, self._microcompact_policy)
        if result.cleared_pair_count == 0:
            self._emit_event(
                trigger=trigger,
                tokens_before=before,
                tokens_after=before,
                outcome="skipped",
                duration_ms=_elapsed_ms(started),
            )
            return messages
        # Pre-Phase-1 parity event — kept so existing observability
        # (test_microcompact_journal_event etc.) doesn't break.
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
        after = sum(_msg_chars(m) for m in result.messages)
        self._emit_event(
            trigger=trigger,
            tokens_before=before,
            tokens_after=after,
            outcome="ok",
            duration_ms=_elapsed_ms(started),
        )
        return result.messages

    async def reactive(
        self,
        history: list[BaseMessage],
        slots: LoopSlots,  # noqa: ARG002 — reserved for circuit-breaker on reactive
        trigger: CompactionTrigger = CompactionTrigger.reactive,
    ) -> CompactResult:
        """Run a full summary compaction in response to context-overflow.

        Mirrors :meth:`LegacyCompactor.reactive` — delegates to
        ``Agent.compact(source="reactive")`` so test patches on
        ``Agent.compact`` continue to observe the reactive path, then
        refreshes the in-memory ``history`` list in place so the loop's
        local reference points at the post-compact transcript without a
        re-assignment.
        """
        before = self._agent._state.total_tokens_used
        started = time.monotonic()
        try:
            result = await self._agent.compact(source="reactive")
        except Exception:
            self._emit_event(
                trigger=trigger,
                tokens_before=before,
                tokens_after=before,
                outcome="failed",
                duration_ms=_elapsed_ms(started),
            )
            raise
        history[:] = self._agent._storage.load(self._agent.session_id)
        self._emit_event(
            trigger=trigger,
            tokens_before=result.before_tokens,
            tokens_after=result.after_tokens,
            outcome="ok",
            duration_ms=_elapsed_ms(started),
        )
        return result

    async def auto(
        self,
        history: list[BaseMessage],
        slots: LoopSlots,
        *,
        model: str,  # noqa: ARG002 — model is read via agent threshold helper
        trigger: CompactionTrigger = CompactionTrigger.auto,
    ) -> CompactResult | None:
        """Post-turn auto-compact threshold check + run.

        Returns the :class:`CompactResult` when compaction ran; ``None``
        when the threshold was not exceeded OR the circuit breaker
        disabled the trigger. On failure, increments
        ``slots.consecutive_compact_failures`` (via
        :func:`dataclasses.replace` because :class:`LoopSlots` is
        frozen) and re-raises; on success, resets it to 0.

        The circuit-breaker limit comes from
        ``self._config.max_consecutive_failures`` (Phase 4 lifted it
        from the hardcoded ``_AUTO_COMPACT_CIRCUIT_BREAKER_LIMIT`` in
        the legacy adapter).
        """
        threshold = self._agent._effective_auto_compact_threshold()
        before = self._agent._state.total_tokens_used
        started = time.monotonic()
        if threshold <= 0:
            self._emit_event(
                trigger=trigger,
                tokens_before=before,
                tokens_after=before,
                outcome="skipped",
                duration_ms=_elapsed_ms(started),
            )
            return None
        used = before
        used_estimator = used == 0
        if used_estimator:
            used = self._agent._estimate_history_tokens(history)
        if used <= threshold:
            self._emit_event(
                trigger=trigger,
                tokens_before=used,
                tokens_after=used,
                outcome="skipped",
                duration_ms=_elapsed_ms(started),
            )
            return None
        failures = slots.consecutive_compact_failures
        if failures >= self._config.max_consecutive_failures:
            journal.write(
                "auto_compact_skipped_circuit_breaker",
                session=self._agent.session_id,
                tokens=used,
                threshold=threshold,
                consecutive_failures=failures,
                used_estimator=used_estimator,
            )
            self._emit_event(
                trigger=trigger,
                tokens_before=used,
                tokens_after=used,
                outcome="skipped",
                duration_ms=_elapsed_ms(started),
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
            self._emit_event(
                trigger=trigger,
                tokens_before=used,
                tokens_after=used,
                outcome="failed",
                duration_ms=_elapsed_ms(started),
            )
            raise
        self._agent._state.slots = dataclasses.replace(
            self._agent._state.slots,
            consecutive_compact_failures=0,
        )
        self._emit_event(
            trigger=trigger,
            tokens_before=result.before_tokens,
            tokens_after=result.after_tokens,
            outcome="ok",
            duration_ms=_elapsed_ms(started),
        )
        return result

    async def manual(
        self,
        history: list[BaseMessage],  # noqa: ARG002 — manual reads from storage
        slots: LoopSlots,  # noqa: ARG002 — manual bypasses the breaker
        trigger: CompactionTrigger = CompactionTrigger.manual,
    ) -> CompactResult:
        """Run a user-invoked ``/compact`` cycle.

        Delegates to ``Agent.compact(source="manual")``. The circuit
        breaker is intentionally NOT consulted — manual is the user's
        explicit override, mirroring the pre-Phase-1 contract (see
        ``test_breaker_emits_skip_journal_event``'s
        ``test_manual_compact_bypasses_breaker`` sibling).
        """
        before = self._agent._state.total_tokens_used
        started = time.monotonic()
        try:
            result = await self._agent.compact(source="manual")
        except Exception:
            self._emit_event(
                trigger=trigger,
                tokens_before=before,
                tokens_after=before,
                outcome="failed",
                duration_ms=_elapsed_ms(started),
            )
            raise
        self._emit_event(
            trigger=trigger,
            tokens_before=result.before_tokens,
            tokens_after=result.after_tokens,
            outcome="ok",
            duration_ms=_elapsed_ms(started),
        )
        return result

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        *,
        trigger: CompactionTrigger,
        tokens_before: int,
        tokens_after: int,
        outcome: str,
        duration_ms: float,
    ) -> None:
        """Write the journal record and forward to the AG-UI emitter.

        Both sinks receive identical payloads — the wire-format dict
        produced by :func:`compact_event_to_wire`. The journal record
        carries ``session`` for grep parity with the rest of the
        ``compact_*`` event family.
        """
        payload = compact_event_to_wire(
            trigger=str(trigger),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            outcome=outcome,
            duration_ms=duration_ms,
        )
        journal.write(
            "compact_event",
            session=self._session_id,
            trigger=str(trigger),
            tokens_before=int(tokens_before),
            tokens_after=int(tokens_after),
            outcome=outcome,
            duration_ms=float(duration_ms),
        )
        if self._event_emitter is not None:
            # Defensive: emitter failure must not abort a compact cycle
            # (the journal record is the system-of-record; the
            # emitter is a UI niceness).
            with contextlib.suppress(Exception):
                self._event_emitter(payload)


def _elapsed_ms(started_monotonic: float) -> float:
    return (time.monotonic() - started_monotonic) * 1000.0


def _msg_chars(message: BaseMessage) -> int:
    """Cheap proxy for "size of this message" — content length in chars.

    Used by :meth:`Compactor.microcompact` to populate
    ``tokens_before`` / ``tokens_after`` on the ``compact_event``.
    Microcompact doesn't have a real token count handy (the loop
    estimator runs at the prompt level, not on individual messages),
    and a char-count proxy is good enough for a UI indicator. Tool
    calls aren't counted — microcompact only clears tool-result
    payloads, so the AIMessage tool_calls structure is unchanged.
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        # Multimodal content blocks — sum string-ish parts.
        total = 0
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    total += len(text)
            elif isinstance(block, str):
                total += len(block)
        return total
    return 0


__all__ = ["Compactor", "EventEmitter"]
