"""run_task — the subagent dispatch core.

Invariants:

- Designed to be scheduled via ``asyncio.create_task`` and NEVER awaited
  by the tool that spawned it (fire-and-forget). The ``task_create`` tool
  records the handle so the parent Agent can cancel it on close.
- Exceptions bubbling out of the subagent are caught and written to the
  record's ``error`` field; they do NOT propagate to the parent's loop.
- ``CancelledError`` is the exception that DOES propagate — it's the
  parent telling us "stop now." We mark the record cancelled first so
  ``/tasks`` reflects reality, then re-raise.

Wall-clock timeout (U4)
-----------------------
Every subagent run is wrapped in an ``asyncio.timeout`` with a
defense-in-depth ceiling so a stalled model call / pathological tool /
infinite small-sleep loop can't leave a task in ``running`` forever.
Defaults to :data:`DEFAULT_SUBAGENT_TIMEOUT_SEC` (5 minutes); operators
can override via the ``AURA_SUBAGENT_TIMEOUT_SEC`` environment variable
(float, ``0`` or negative disables the ceiling — escape hatch for
long-running agents). On timeout the record is flipped to ``failed``
with ``error="subagent_timeout: exceeded <N>s"`` and the
``subagent_timeout`` journal event fires.

Round 4F additions
------------------
- ``transcript_storage`` kwarg — when supplied, the child's full message
  list is flushed to ``<storage_root>/subagents/subagent-<task_id>.jsonl``
  on EVERY terminal branch (success / cancel / timeout / failed). The
  resolved on-disk path is pinned onto the TaskRecord via
  :meth:`TasksStore.set_transcript_path` so ``task_get`` can surface it.
  Persisting through every termination branch (not just success) is
  deliberate: a debugger triaging a failed subagent needs the trail.

Round 7QS additions
-------------------
- A token-tracking ``post_model`` hook is installed on the child's
  HookChain right after :meth:`SubagentFactory.spawn` returns. Each
  AIMessage carries provider-side ``usage_metadata`` which we forward
  to :meth:`TasksStore.record_token_usage`. The hook is forgiving:
  missing usage_metadata is a no-op (FakeChatModel et al.), negative
  values clamp to 0 (defensive against a misconfigured proxy).
- An optional :class:`AgentSummarizer` ticks every
  ``summary_interval_sec`` (or :data:`AURA_AGENT_SUMMARY_INTERVAL_SEC`,
  or :data:`agent_summary.DEFAULT_INTERVAL_SEC`), writing the cheap
  model's one-sentence digest onto ``progress.latest_summary``. Pass
  ``summary_interval_sec=0`` to disable for a specific run (the
  token-tracking tests rely on this).

Matches the spirit of claude-code's ``AbortController`` contract
(``src/tools/AgentTool/runAgent.ts``) — claude-code uses an abort
controller instead of a wallclock, but the escape is the same: a
stalled child never blocks the parent loop forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, messages_to_dict

from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.store import TasksStore
from aura.schemas.events import Final, ToolCallStarted

# Default wall-clock ceiling for a single subagent run. 5 minutes is loose
# enough that legitimate research / multi-turn explorations finish well
# under the cap, while tight enough that an operator doesn't watch a
# silently-stuck task for hours before noticing. Picked to match
# ``PermissionsConfig.prompt_timeout_sec`` (same order of magnitude —
# "long enough to not fight legitimate use, short enough to matter").
DEFAULT_SUBAGENT_TIMEOUT_SEC: float = 300.0

# Environment variable override. Float seconds. ``<= 0`` disables the
# ceiling (escape hatch for specialized long-running subagents — use at
# your own risk; a stuck task can still be killed by ``task_stop`` or
# ``Agent.close``). Parsed at every call site so a long-lived process can
# pick up an env change without restart.
_TIMEOUT_ENV_VAR = "AURA_SUBAGENT_TIMEOUT_SEC"


def _resolve_timeout(override: float | None) -> float | None:
    """Pick the effective wallclock timeout.

    Precedence (highest first):

    1. Explicit ``override`` kwarg to :func:`run_task` — tests inject a
       sub-second ceiling so the regression suite doesn't have to wait 5
       minutes to observe the timeout branch.
    2. ``AURA_SUBAGENT_TIMEOUT_SEC`` environment variable.
    3. :data:`DEFAULT_SUBAGENT_TIMEOUT_SEC`.

    Returns ``None`` when the resolved value is ``<= 0``, meaning "no
    timeout — run until natural completion / cancellation." Keeping the
    escape hatch explicit is intentional: operators running overnight
    investigations shouldn't have to patch the module, and an errant
    ``0`` shouldn't fire a deny-all timeout on every task.
    """
    if override is not None:
        return override if override > 0 else None
    raw = os.environ.get(_TIMEOUT_ENV_VAR)
    if raw is not None:
        try:
            parsed = float(raw)
        except ValueError:
            # Malformed env — journal and fall through to default. Not a
            # hard error: we're in a fire-and-forget path where raising
            # would orphan the TaskRecord in ``running``.
            journal.write(
                "subagent_timeout_env_invalid",
                var=_TIMEOUT_ENV_VAR,
                value=raw,
            )
        else:
            return parsed if parsed > 0 else None
    return DEFAULT_SUBAGENT_TIMEOUT_SEC


def _make_token_observer(store: TasksStore, task_id: str) -> Any:
    """Build a post_model hook that forwards usage_metadata into the store.

    The hook is forgiving:
      - ``ai_message.usage_metadata is None`` → no-op (FakeChatModel etc.).
      - Negative values clamp inside :meth:`TasksStore.record_token_usage`.
      - Exceptions are caught + journaled — a flaky observer must NEVER
        derail the child's astream.
    """
    async def _observe(
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],
        state: Any,
        **_: Any,
    ) -> None:
        try:
            usage = getattr(ai_message, "usage_metadata", None)
            if not usage:
                return
            in_t = int(usage.get("input_tokens", 0) or 0)
            out_t = int(usage.get("output_tokens", 0) or 0)
            store.record_token_usage(
                task_id, input_tokens=in_t, output_tokens=out_t,
            )
        except Exception as exc:  # noqa: BLE001
            journal.write(
                "subagent_token_observer_error",
                task_id=task_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    return _observe


def _flush_transcript(
    *,
    transcript_storage: SessionStorage,
    task_id: str,
    messages: list[BaseMessage],
    store: TasksStore,
) -> Path | None:
    """Write the child's transcript as JSONL under ``<root>/subagents/``.

    Returns the path written (and pins it on the TaskRecord), or ``None``
    on failure. Failures are journaled but never re-raised — losing a
    transcript file is strictly less bad than stranding the child's
    terminal mark on a write error.
    """
    try:
        # Use the storage's underlying path to derive the root. ``:memory:``
        # storage gives Path(":memory:"), whose .parent is Path("."). That's
        # acceptable for tests; production callers always use a real path.
        root = getattr(transcript_storage, "_path", None)
        if root is None:
            return None
        sub_dir = Path(root).parent / "subagents"
        sub_dir.mkdir(parents=True, exist_ok=True)
        path = sub_dir / f"subagent-{task_id}.jsonl"
        # JSONL: one message per line. ``messages_to_dict`` is the same
        # shape SessionStorage.save uses, so a future load round-trips
        # via ``messages_from_dict``.
        payloads = messages_to_dict(messages)
        with path.open("w", encoding="utf-8") as fh:
            for payload in payloads:
                fh.write(json.dumps(payload, ensure_ascii=False))
                fh.write("\n")
        # Best-effort: register the transcript with the storage's
        # ``write_subagent_transcript`` index when present. A future
        # storage upgrade will surface this through the API; until
        # then we just write the raw file. Whichever approach the
        # caller wires, the task_get-visible path is identical.
        register = getattr(
            transcript_storage, "write_subagent_transcript", None,
        )
        if callable(register):
            try:
                register(task_id, messages)
            except Exception as exc:  # noqa: BLE001
                # Index write failure is non-fatal — the file is on disk.
                journal.write(
                    "subagent_transcript_index_error",
                    task_id=task_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
        store.set_transcript_path(task_id, path)
        return path
    except Exception as exc:  # noqa: BLE001
        journal.write(
            "subagent_transcript_flush_error",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return None


async def run_task(
    store: TasksStore,
    factory: SubagentFactory,
    task_id: str,
    *,
    timeout_sec: float | None = None,
    transcript_storage: SessionStorage | None = None,
    summary_interval_sec: float | None = None,
) -> None:
    record = store.get(task_id)
    if record is None:
        return
    # Resolve the ceiling ONCE per call so an env flip mid-run doesn't
    # race against the timeout context manager. Pass the resolved number
    # into the error message + journal so the operator sees exactly how
    # long we waited.
    effective_timeout = _resolve_timeout(timeout_sec)
    # V13-T1C: subagent_start fires before spawn so operators observing the
    # journal see a clear "run_task began for X" signal even if spawn itself
    # raises (the lifecycle pairs with one of subagent_completed /
    # subagent_cancelled / subagent_timeout / subagent_failed, all of which
    # now carry a ``duration_sec`` field computed from ``start_monotonic``).
    start_monotonic = time.monotonic()
    journal.write(
        "subagent_start",
        task_id=task_id,
        agent_type=record.agent_type or "general-purpose",
        prompt_chars=len(record.prompt),
    )
    # Bug fix (integration test): ``factory.spawn(...)`` USED to live
    # outside the try/except, so if spawn itself raised (e.g. ValueError
    # from ``agent_type="explore"`` requiring a tool the parent hadn't
    # enabled), ``mark_failed`` was never called and the TaskRecord
    # stayed permanently in ``running`` status. Move spawn into the try;
    # guard ``agent.close()`` with a None-check so the except blocks
    # don't trip over an unbound name when spawn itself failed.
    agent: Any = None
    final_text = ""
    summarizer: Any = None
    try:
        # Try the new spec-aware spawn signature first; fall back to the
        # legacy positional signature for tests that subclass SubagentFactory
        # without accepting ``model_spec``.
        try:
            agent = factory.spawn(
                record.prompt,
                agent_type=record.agent_type or "general-purpose",
                task_id=task_id,
                model_spec=record.model_spec or None,
            )
        except TypeError as exc:
            if "model_spec" in str(exc):
                agent = factory.spawn(
                    record.prompt,
                    agent_type=record.agent_type or "general-purpose",
                    task_id=task_id,
                )
            else:
                raise
        # Round 7QS — token tracking. Install the post_model observer
        # on the child's HookChain. Best-effort: a child Agent whose
        # ``_hooks`` doesn't expose post_model (very minimal stub in
        # some tests) is silently skipped.
        child_hooks = getattr(agent, "_hooks", None)
        if child_hooks is not None and hasattr(child_hooks, "post_model"):
            child_hooks.post_model.append(_make_token_observer(store, task_id))
        # Round 7QS — periodic summary. Build the cheap-model factory
        # via ``llm.make_summary_model_factory`` (memoized). Disabled
        # paths: ``summary_interval_sec=0`` (test injection),
        # AURA_AGENT_SUMMARY_INTERVAL_SEC=0 env, or factory missing.
        from aura.core import llm as _llm_mod
        from aura.core.services.agent_summary import AgentSummarizer

        _make_summary_factory = getattr(
            _llm_mod, "make_summary_model_factory", None,
        )
        if _make_summary_factory is not None:
            # ``cfg.web_fetch.summary_model`` is the canonical config
            # path for the cheap model; absent → factory falls back
            # to the main model. The cfg attribute may not exist
            # (``AuraConfig`` doesn't validate the ``web_fetch`` key
            # today), so we read defensively.
            cfg = getattr(agent, "_config", None)
            wf_cfg = getattr(cfg, "web_fetch", None) if cfg is not None else None
            summary_spec = (
                getattr(wf_cfg, "summary_model", None) if wf_cfg is not None else None
            )
            main_model = getattr(agent, "_model", None)
            if main_model is not None:
                summary_factory = _make_summary_factory(
                    cfg, main_model, summary_spec=summary_spec,
                )

                # Transcript provider reads from the child Agent's live
                # storage on every tick — the child's history is
                # ``save``d on every astream ainvoke, so a summarizer
                # tick mid-run sees the latest tool round-trips even
                # before run_task's terminal _capture_child_messages
                # mirrors them onto the TaskRecord.
                child_storage = getattr(agent, "_storage", None)
                child_session_id = getattr(agent, "_session_id", None)

                def _transcript_provider() -> list[BaseMessage]:
                    if child_storage is not None and child_session_id is not None:
                        try:
                            return list(child_storage.load(child_session_id))
                        except Exception:  # noqa: BLE001
                            pass
                    rec = store.get(task_id)
                    return list(rec.messages) if rec is not None else []

                summarizer = AgentSummarizer(
                    task_id=task_id,
                    store=store,
                    transcript_provider=_transcript_provider,
                    summary_model_factory=summary_factory,
                    interval_sec=summary_interval_sec,
                )
                summarizer.start()
        # ``asyncio.timeout(None)`` is a no-op context — the escape hatch
        # (env / kwarg set ``<= 0``) flows through as ``None`` and we get
        # the legacy "wait forever" behavior with no extra indirection.
        async with asyncio.timeout(effective_timeout):
            async for event in agent.astream(record.prompt):
                if isinstance(event, ToolCallStarted):
                    # Progress is for the PARENT's ``task_get`` polling —
                    # cheap write, stays in parent's event loop, no cross-process
                    # sync. Child tool invocations are the most meaningful unit
                    # of "what is this subagent doing right now".
                    store.record_activity(task_id, event.name)
                elif isinstance(event, Final):
                    final_text = event.message
        # Capture the child's full transcript onto the TaskRecord BEFORE
        # the terminal mark fires — a listener that wants to surface the
        # transcript count needs it on the record at fire time.
        await _capture_child_messages(agent, store, task_id)
        store.mark_completed(task_id, final_text)
    except asyncio.CancelledError:
        await _capture_child_messages(agent, store, task_id)
        store.mark_cancelled(task_id)
        journal.write(
            "subagent_cancelled",
            task_id=task_id,
            duration_sec=round(time.monotonic() - start_monotonic, 3),
        )
        if transcript_storage is not None:
            _flush_transcript(
                transcript_storage=transcript_storage,
                task_id=task_id,
                messages=_load_child_messages(agent, store, task_id),
                store=store,
            )
        if summarizer is not None:
            await summarizer.stop()
        if agent is not None:
            # ``aclose`` is the async-safe teardown. ``close()`` on a live
            # event loop raises ``RuntimeError`` when the child Agent has
            # an MCP manager — today subagents don't wire MCP so the old
            # sync ``close()`` happened to work, but the moment a subagent
            # inherits MCP (an expected near-future change) every terminal
            # branch here would crash ``run_task`` mid-cleanup, orphaning
            # connections and leaving the TaskRecord in ``running``.
            await agent.aclose()
        raise
    except TimeoutError as exc:
        # Wall-clock ceiling hit. ``asyncio.timeout`` raises TimeoutError
        # after the enclosed scope is cancelled — the child's astream got
        # a CancelledError, its own except-CancelledError branches cleaned
        # up, and now we convert that into a ``failed`` record so the
        # parent's task_output sees a terminal state with a clear reason
        # rather than "stuck running forever".
        await _capture_child_messages(agent, store, task_id)
        err_msg = (
            f"subagent_timeout: exceeded {effective_timeout}s "
            f"(set AURA_SUBAGENT_TIMEOUT_SEC to override; "
            f"<=0 disables the ceiling)"
        )
        store.mark_failed(task_id, err_msg)
        journal.write(
            "subagent_timeout",
            task_id=task_id,
            timeout_sec=effective_timeout,
            duration_sec=round(time.monotonic() - start_monotonic, 3),
            # Preserve the underlying cause for deep debugging; mostly
            # ``TimeoutError()`` with no message, but a future re-raise
            # inside astream could carry something more useful.
            cause=f"{type(exc).__name__}: {exc}",
        )
        if transcript_storage is not None:
            _flush_transcript(
                transcript_storage=transcript_storage,
                task_id=task_id,
                messages=_load_child_messages(agent, store, task_id),
                store=store,
            )
        if summarizer is not None:
            await summarizer.stop()
        if agent is not None:
            await agent.aclose()
    except Exception as exc:  # noqa: BLE001
        # spawn-time failures land here too (``factory.spawn`` raised) —
        # ``agent`` is still ``None`` in that case, which is fine: we
        # just mark the record failed without a transcript.
        if agent is not None:
            await _capture_child_messages(agent, store, task_id)
        err_msg = f"{type(exc).__name__}: {exc}"
        store.mark_failed(task_id, err_msg)
        journal.write(
            "subagent_failed",
            task_id=task_id,
            error=err_msg,
            duration_sec=round(time.monotonic() - start_monotonic, 3),
        )
        if transcript_storage is not None and agent is not None:
            _flush_transcript(
                transcript_storage=transcript_storage,
                task_id=task_id,
                messages=_load_child_messages(agent, store, task_id),
                store=store,
            )
        if summarizer is not None:
            await summarizer.stop()
        if agent is not None:
            await agent.aclose()
    else:
        journal.write(
            "subagent_completed",
            task_id=task_id,
            duration_sec=round(time.monotonic() - start_monotonic, 3),
            final_text_chars=len(final_text),
        )
        if transcript_storage is not None and agent is not None:
            _flush_transcript(
                transcript_storage=transcript_storage,
                task_id=task_id,
                messages=_load_child_messages(agent, store, task_id),
                store=store,
            )
        if summarizer is not None:
            await summarizer.stop()
        if agent is not None:
            await agent.aclose()


async def _capture_child_messages(
    agent: Any, store: TasksStore, task_id: str,
) -> None:
    """Pull the child's full message list onto the TaskRecord.

    ``run_task`` uses the agent's astream events to drive progress, but
    the assistant + tool messages live on the child's
    :class:`SessionStorage`. We snapshot them onto the record so
    listeners (Round 4F) and ``task_get(include_messages=True)`` see
    the complete transcript without an extra storage lookup.
    """
    if agent is None:
        return
    try:
        storage = getattr(agent, "_storage", None)
        session_id = getattr(agent, "session_id", None) or getattr(
            agent, "_session_id", None,
        )
        if storage is None or session_id is None:
            return
        msgs = storage.load(session_id)
        rec = store.get(task_id)
        if rec is None:
            return
        # Replace rather than extend — the child's storage IS the
        # authoritative source; running incrementally via append would
        # double-count after a /clear.
        rec.messages = list(msgs)
    except Exception as exc:  # noqa: BLE001
        journal.write(
            "subagent_capture_messages_error",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )


def _load_child_messages(
    agent: Any, store: TasksStore, task_id: str,
) -> list[BaseMessage]:
    """Read the child's transcript for transcript-flush.

    Prefers the snapshot pinned by :func:`_capture_child_messages` (so a
    cancel branch that already ran capture doesn't double-load); falls
    back to a fresh read from the child's storage.
    """
    rec = store.get(task_id)
    if rec is not None and rec.messages:
        return list(rec.messages)
    if agent is None:
        return []
    with contextlib.suppress(Exception):
        storage = getattr(agent, "_storage", None)
        session_id = getattr(agent, "session_id", None) or getattr(
            agent, "_session_id", None,
        )
        if storage is None or session_id is None:
            return []
        return list(storage.load(session_id))
    return []
