"""Periodic cheap-model digest of a running subagent's transcript, written to TaskProgress."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from aura.infrastructure.persistence import journal

if TYPE_CHECKING:
    from aura.application.tasks.store import TasksStore


# 30s default: cheap-amortized over a multi-minute run, low parent-observability lag.
DEFAULT_INTERVAL_SEC: float = 30.0

# Float seconds; ``<=0`` disables. Re-read each call so a live process picks up env changes.
_INTERVAL_ENV_VAR = "AURA_AGENT_SUMMARY_INTERVAL_SEC"

# Cap transcript chars to keep the cheap-model call cheap; tail-keep favors most-recent state.
_TRANSCRIPT_INPUT_CAP_CHARS = 4_000

# Trailing messages kept: 8 ≈ last prompt + 2-3 tool round-trips + recent assistant turn.
_TRANSCRIPT_TAIL_MESSAGES = 8

# Fixed single-sentence, action-focused summary prompt.
SUMMARY_PROMPT_TEMPLATE = (
    "You are observing a subagent running on behalf of a parent agent. "
    "Below is the subagent's recent transcript. In ONE concise sentence "
    "(max ~20 words), describe what the subagent has accomplished so far "
    "or what it is currently doing. Do not editorialize, do not speculate, "
    "do not add prefixes like 'The subagent is'. Just the action.\n\n"
    "Transcript:\n{transcript}"
)


def _resolve_interval(override: float | None) -> float:
    """Resolve interval: ``override`` → env var → default; ``<=0`` → 0.0 (disabled)."""
    if override is not None:
        return max(0.0, override)
    raw = os.environ.get(_INTERVAL_ENV_VAR)
    if raw is not None:
        try:
            parsed = float(raw)
        except ValueError:
            journal.write(
                "agent_summary_env_invalid",
                var=_INTERVAL_ENV_VAR,
                value=raw,
            )
            return DEFAULT_INTERVAL_SEC
        return max(0.0, parsed)
    return DEFAULT_INTERVAL_SEC


def _format_transcript(messages: list[BaseMessage]) -> str:
    """Tail-truncate + char-cap the recent transcript for the summary model; tool output trimmed."""
    tail = messages[-_TRANSCRIPT_TAIL_MESSAGES:]
    rendered_lines: list[str] = []
    for msg in tail:
        role: str
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = f"tool({getattr(msg, 'name', '?')})"
        else:
            role = msg.type if hasattr(msg, "type") else "msg"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, ToolMessage) and len(content) > 200:
            content = content[:200] + "…"
        rendered_lines.append(f"[{role}] {content}")
    rendered = "\n".join(rendered_lines)
    if len(rendered) > _TRANSCRIPT_INPUT_CAP_CHARS:
        # Tail-keep (drop the head): most-recent state is what the summary should describe.
        rendered = "…\n" + rendered[-(_TRANSCRIPT_INPUT_CAP_CHARS - 2):]
    return rendered


async def _run_one_summary(
    *,
    task_id: str,
    store: TasksStore,
    transcript_provider: Callable[[], list[BaseMessage]],
    summary_model: BaseChatModel,
) -> None:
    """One tick: read transcript, invoke model, write summary; errors journaled + swallowed."""
    try:
        messages = transcript_provider()
    except Exception as exc:  # noqa: BLE001  # log + swallow; logging path must never crash caller
        journal.write(
            "agent_summary_transcript_error",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return
    if not messages:
        return
    transcript_text = _format_transcript(messages)
    if not transcript_text.strip():
        return
    prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript_text)
    try:
        ai = await summary_model.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:  # noqa: BLE001  # log + swallow; logging path must never crash caller
        journal.write(
            "agent_summary_invoke_error",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return
    text = ai.content if isinstance(ai.content, str) else str(ai.content)
    cleaned = text.strip()
    if not cleaned:
        return
    store.update_summary(task_id, cleaned)
    journal.write(
        "agent_summary_updated",
        task_id=task_id,
        summary_chars=len(cleaned),
    )


def _is_terminal(store: TasksStore, task_id: str) -> bool:
    """True once the record left ``running`` (or vanished); cheap per-tick status read."""
    rec = store.get(task_id)
    if rec is None:
        return True
    return rec.status != "running"


async def run_summary_loop(
    *,
    task_id: str,
    store: TasksStore,
    transcript_provider: Callable[[], list[BaseMessage]],
    summary_model_factory: Callable[[], BaseChatModel],
    interval_sec: float | None = None,
) -> None:
    """Tick until the record is terminal or the task is cancelled; ``<=0`` interval disables."""
    interval = _resolve_interval(interval_sec)
    if interval <= 0:
        journal.write(
            "agent_summary_disabled",
            task_id=task_id,
            reason="interval_zero",
        )
        return
    # Lazy model build: defer SDK construction to the first tick so a bad spec doesn't crash spawn.
    summary_model: BaseChatModel | None = None
    while not _is_terminal(store, task_id):
        # ``asyncio.sleep`` is the abort cancel point; CancelledError unwinds to cleanup.
        await asyncio.sleep(interval)
        if _is_terminal(store, task_id):
            return
        if summary_model is None:
            try:
                summary_model = summary_model_factory()
            except Exception as exc:  # noqa: BLE001  # cleanup path must not propagate
                journal.write(
                    "agent_summary_model_factory_error",
                    task_id=task_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                # No model: exit rather than retry every tick.
                return
        await _run_one_summary(
            task_id=task_id,
            store=store,
            transcript_provider=transcript_provider,
            summary_model=summary_model,
        )


class AgentSummarizer:
    """Owns the summary background asyncio.Task; ``start`` schedules it, ``stop`` cancels it."""

    def __init__(
        self,
        *,
        task_id: str,
        store: TasksStore,
        transcript_provider: Callable[[], list[BaseMessage]],
        summary_model_factory: Callable[[], BaseChatModel],
        interval_sec: float | None = None,
    ) -> None:
        self._task_id = task_id
        self._store = store
        self._transcript_provider = transcript_provider
        self._summary_model_factory = summary_model_factory
        self._interval_sec = interval_sec
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Schedule the summary loop on the current event loop; idempotent."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            run_summary_loop(
                task_id=self._task_id,
                store=self._store,
                transcript_provider=self._transcript_provider,
                summary_model_factory=self._summary_model_factory,
                interval_sec=self._interval_sec,
            )
        )

    async def stop(self) -> None:
        """Cancel the summary task and await cleanup; no-op if never started or already done."""
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    @property
    def done(self) -> bool:
        """True when the summary loop has exited (cleanly or cancelled)."""
        return self._task is None or self._task.done()


TranscriptProvider = Callable[[], list[BaseMessage]]
SummaryModelFactory = Callable[[], BaseChatModel]
