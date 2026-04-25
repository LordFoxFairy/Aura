"""AgentSummarizer — periodic LLM-driven digest of a running subagent.

Round 7QS. Mirrors claude-code's ``startAgentSummarization``
(``src/services/AgentSummary/agentSummary.ts``):

- A background asyncio task per running subagent. Wakes every
  :data:`DEFAULT_INTERVAL_SEC` seconds (configurable via
  ``AURA_AGENT_SUMMARY_INTERVAL_SEC`` — set to ``0`` to disable).
- Each tick: pulls the child's recent transcript (last few tool calls +
  most recent assistant message), feeds it to a CHEAP summary model
  with a fixed one-sentence prompt, writes the result to
  :attr:`TaskProgress.latest_summary` via
  :meth:`TasksStore.update_summary`.
- Exits when the underlying :class:`TaskRecord` reaches a terminal
  state (``record.status != "running"``) or when the surrounding
  asyncio task is cancelled — whichever comes first. Polling the
  record's status keeps the implementation independent of any
  asyncio.Event-style terminal signal the store may expose later.

Cost trade-off: one cheap-model invoke every 30s per running child.
For a long-running subagent that takes 5 minutes that's ~10 calls of
Haiku-tier pricing — negligible relative to the main work. The kill
switch (``interval_sec=0`` or env var set to ``0``) is the operator
escape hatch.

The summary model is the SAME one ``web_fetch`` uses
(``llm.make_summary_model_factory``) so users get one config knob for
both. Without ``cfg.web_fetch.summary_model`` set, the factory falls
back to the main model — workable but expensive; the journal records
the fallback so operators can see they're paying main-model rates.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from aura.core.persistence import journal

if TYPE_CHECKING:
    from aura.core.tasks.store import TasksStore


# Interval default — matches claude-code's 30s tick in
# ``agentSummary.ts``. Long enough that a Haiku-class summary call is
# cheap-amortized over a multi-minute subagent run; short enough that
# parent observability doesn't lag the child's reality by a meaningful
# fraction of its lifetime.
DEFAULT_INTERVAL_SEC: float = 30.0

# Env var override. Float seconds. ``<=0`` disables the summarizer
# entirely (the run loop returns immediately). Parsed at every
# ``run_summary_loop`` call site so a long-lived process can pick up
# an env change without restart.
_INTERVAL_ENV_VAR = "AURA_AGENT_SUMMARY_INTERVAL_SEC"

# Hard cap on transcript characters fed to the summary model — keeps
# the cheap-model call genuinely cheap even when a long subagent has
# accumulated dozens of tool messages. Tail-keep (last N chars) so
# the summary describes the MOST RECENT state of work, which is what
# operators want to see.
_TRANSCRIPT_INPUT_CAP_CHARS = 4_000

# How many trailing transcript messages to consider — beyond this we
# shed older context. 8 is enough to capture the last user prompt +
# 2-3 tool round-trips + the most recent assistant turn.
_TRANSCRIPT_TAIL_MESSAGES = 8

# The fixed prompt template. Single-sentence, action-focused — mirrors
# claude-code's "describe most recent action in 3-5 words" but slightly
# longer because Aura's CLI surface has more room than CC's progress
# pill.
SUMMARY_PROMPT_TEMPLATE = (
    "You are observing a subagent running on behalf of a parent agent. "
    "Below is the subagent's recent transcript. In ONE concise sentence "
    "(max ~20 words), describe what the subagent has accomplished so far "
    "or what it is currently doing. Do not editorialize, do not speculate, "
    "do not add prefixes like 'The subagent is'. Just the action.\n\n"
    "Transcript:\n{transcript}"
)


def _resolve_interval(override: float | None) -> float:
    """Pick the effective summary interval (seconds).

    Precedence:
      1. Explicit ``override`` kwarg (tests inject sub-second ticks).
      2. ``AURA_AGENT_SUMMARY_INTERVAL_SEC`` env var.
      3. :data:`DEFAULT_INTERVAL_SEC`.

    Returns ``0.0`` when the resolved value is ``<= 0``, signalling
    "disabled". Returning a sentinel (rather than ``None``) keeps
    :func:`run_summary_loop`'s top-level branch flat.
    """
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
    """Render a tail-truncated, char-capped transcript for the summary model.

    Walks the last :data:`_TRANSCRIPT_TAIL_MESSAGES` messages, prefixes
    each with a role marker, and trims the head if the rendered total
    exceeds :data:`_TRANSCRIPT_INPUT_CAP_CHARS`. Tool messages are
    abbreviated to their first 200 chars — full tool output is rarely
    useful for "what is the agent doing" and dominates the byte budget.
    """
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
        # Tail-keep (drop the head) — most recent state is what the
        # summarizer should describe.
        rendered = "…\n" + rendered[-(_TRANSCRIPT_INPUT_CAP_CHARS - 2):]
    return rendered


async def _run_one_summary(
    *,
    task_id: str,
    store: TasksStore,
    transcript_provider: Callable[[], list[BaseMessage]],
    summary_model: BaseChatModel,
) -> None:
    """One tick: read transcript, invoke model, write summary on the record.

    All exceptions are swallowed + journaled — a flaky summary model
    must never derail the underlying subagent. Empty transcripts
    short-circuit (no point invoking the model for nothing).
    """
    try:
        messages = transcript_provider()
    except Exception as exc:  # noqa: BLE001
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
    except Exception as exc:  # noqa: BLE001
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
    """True once the underlying record has left the ``running`` state.

    Reading the record on every tick is cheap (dict lookup + attr
    read) and avoids depending on any asyncio.Event signal the store
    might expose. ``None`` (record cleared mid-flight, shouldn't
    happen but be defensive) also counts as terminal so the loop
    exits cleanly.
    """
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
    """Periodic-summary loop for one subagent.

    Runs until the task record reaches a terminal state, or until the
    surrounding asyncio task is cancelled. At every interval it calls
    :func:`_run_one_summary`. The summary model is built lazily on the
    first tick so a misconfigured spec doesn't crash spawn — the failure
    surfaces as a journal entry only on the first attempted tick.

    ``interval_sec`` overrides env / default; pass a sub-second value
    in tests to drive the loop deterministically.

    No-op when the resolved interval is ``<= 0`` (disabled).
    """
    interval = _resolve_interval(interval_sec)
    if interval <= 0:
        journal.write(
            "agent_summary_disabled",
            task_id=task_id,
            reason="interval_zero",
        )
        return
    # Lazy model build — defer SDK construction until the first tick so
    # a misconfigured ``cfg.web_fetch.summary_model`` doesn't blow up
    # ``run_task`` on every spawn. Factory is memoized; subsequent
    # ticks reuse the same instance.
    summary_model: BaseChatModel | None = None
    while not _is_terminal(store, task_id):
        # ``asyncio.sleep`` is the cancel point for parent abort; the
        # CancelledError that lands here propagates out of the loop and
        # the surrounding ``finally`` in run_task awaits cleanup.
        await asyncio.sleep(interval)
        if _is_terminal(store, task_id):
            return
        if summary_model is None:
            try:
                summary_model = summary_model_factory()
            except Exception as exc:  # noqa: BLE001
                journal.write(
                    "agent_summary_model_factory_error",
                    task_id=task_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                # Without a model we can't summarize — exit the loop
                # rather than retry every tick.
                return
        await _run_one_summary(
            task_id=task_id,
            store=store,
            transcript_provider=transcript_provider,
            summary_model=summary_model,
        )


class AgentSummarizer:
    """Convenience wrapper that owns the summary background asyncio.Task.

    Construction is lazy: :meth:`start` schedules the task on the
    current event loop; :meth:`stop` cancels it. The class isn't
    strictly required (callers can also call :func:`run_summary_loop`
    directly under :func:`asyncio.create_task`) but having a named
    handle simplifies ``run_task``'s ``finally`` cleanup and mirrors
    claude-code's ``AgentSummaryService`` shape.
    """

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
        """Schedule the summary loop on the current event loop.

        Idempotent — repeat ``start()`` calls are no-ops. The loop
        exits naturally when the task record turns terminal, so most
        callers don't need :meth:`stop`; :meth:`stop` exists for
        explicit teardown (parent abort cascade, tests).
        """
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
        """Cancel the summary task and await its cleanup.

        No-op when ``start()`` was never called or the task has already
        finished. Swallows ``CancelledError`` from the awaited task —
        cancellation is the expected path and re-raising would force
        every caller into a try/except.
        """
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


# Re-export type aliases for downstream call sites.
TranscriptProvider = Callable[[], list[BaseMessage]]
SummaryModelFactory = Callable[[], BaseChatModel]
