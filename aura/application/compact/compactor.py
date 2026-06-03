"""Compactor: one object, four trigger entry points (microcompact/reactive/auto/manual)."""

from __future__ import annotations

import contextlib
import dataclasses
import time
from collections.abc import Callable
from typing import Literal, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from aura.application.compact.constants import CompactionTrigger
from aura.application.compact.microcompact import (
    MicrocompactPolicy,
    apply_microcompact,
)
from aura.application.compact.result_types import CompactResult
from aura.application.loop_state import LoopSlots, LoopState
from aura.config.schema import CompactConfig
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.wire.serialize import compact_event_to_wire


class _CompactorSession(Protocol):
    """Narrow contract Compactor needs from AgentSession."""

    @property
    def state(self) -> LoopState: ...
    @property
    def storage(self) -> SessionStorage: ...
    @property
    def session_id(self) -> str: ...

    async def compact(
        self, *, source: Literal["manual", "auto", "reactive"] = "manual"
    ) -> CompactResult: ...

    def effective_auto_compact_threshold(self) -> int: ...
    def estimate_history_tokens(self, history: list[BaseMessage]) -> int: ...


EventEmitter = Callable[[dict[str, object]], None]


class Compactor:

    def __init__(
        self,
        *,
        agent: _CompactorSession,
        config: CompactConfig,
        summary_model: BaseChatModel,
        microcompact_policy: MicrocompactPolicy | None = None,
        session_id: str,
        turn_provider: Callable[[], int],
        event_emitter: EventEmitter | None = None,
    ) -> None:
        self._agent = agent
        self._config = config
        self._summary_model = summary_model
        self._microcompact_policy = microcompact_policy
        self._session_id = session_id
        self._turn_provider = turn_provider
        self._event_emitter = event_emitter

    async def microcompact(
        self,
        messages: list[BaseMessage],
        slots: LoopSlots,  # noqa: ARG002
        trigger: CompactionTrigger = CompactionTrigger.microcompact,
    ) -> list[BaseMessage]:
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
        slots: LoopSlots,  # noqa: ARG002
        trigger: CompactionTrigger = CompactionTrigger.reactive,
    ) -> CompactResult:
        """Context-overflow compaction; refreshes ``history`` in place."""
        before = self._agent.state.total_tokens_used
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
        history[:] = self._agent.storage.load(self._agent.session_id)
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
        model: str,  # noqa: ARG002
        trigger: CompactionTrigger = CompactionTrigger.auto,
    ) -> CompactResult | None:
        """Post-turn threshold-driven run; ``None`` = no work / circuit breaker open."""
        threshold = self._agent.effective_auto_compact_threshold()
        before = self._agent.state.total_tokens_used
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
            used = self._agent.estimate_history_tokens(history)
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
            self._agent.state.slots = dataclasses.replace(
                self._agent.state.slots,
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
        self._agent.state.slots = dataclasses.replace(
            self._agent.state.slots,
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
        history: list[BaseMessage],  # noqa: ARG002
        slots: LoopSlots,  # noqa: ARG002
        trigger: CompactionTrigger = CompactionTrigger.manual,
    ) -> CompactResult:
        """User-invoked ``/compact``; bypasses the circuit breaker by spec."""
        before = self._agent.state.total_tokens_used
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

    def _emit_event(
        self,
        *,
        trigger: CompactionTrigger,
        tokens_before: int,
        tokens_after: int,
        outcome: str,
        duration_ms: float,
    ) -> None:
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
            with contextlib.suppress(Exception):
                self._event_emitter(dict(payload))


def _elapsed_ms(started_monotonic: float) -> float:
    return (time.monotonic() - started_monotonic) * 1000.0


def _msg_chars(message: BaseMessage) -> int:
    content = message.content
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
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
