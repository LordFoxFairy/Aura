"""LocalAgentTask — in-process child Agent runner, fire-and-forget.

Translates claude-code's ``tasks/LocalAgentTask`` into Python. The
runner owns one child :class:`~aura.core.agent.Agent` instance,
drives a single prompt through ``astream``, and writes the terminal
outcome onto the :class:`~aura.application.tasks.store.TasksStore` record.
Fire-and-forget: the runner schedules itself via
``asyncio.create_task`` and the parent never awaits the resulting
asyncio.Task directly — cancellation flows through :meth:`abort`.

This module owns the bulk of subagent lifecycle plumbing (transcript
flush, metadata flush, periodic summary, token observer) so the
``run.py`` driver can stay thin and route between runner topologies.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage

from aura.application.tasks.store import TasksStore
from aura.domain.task import TaskRecord
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage
from aura.schemas.events import Final, ToolCallStarted

if TYPE_CHECKING:
    from aura.application.tasks.factory import SubagentFactory


# 5 minute defense-in-depth ceiling. Matches the spirit of claude-code's
# AbortController contract — a stuck child must never block the parent
# forever. Env override + sub-zero escape hatch kept for operators
# running specialised long jobs.
DEFAULT_SUBAGENT_TIMEOUT_SEC: float = 300.0
_TIMEOUT_ENV_VAR = "AURA_SUBAGENT_TIMEOUT_SEC"


def resolve_timeout(override: float | None) -> float | None:
    """Pick the effective wallclock timeout (None == disabled).

    Precedence: explicit override > env var > default. ``<= 0`` flows
    through as ``None`` (no cap). Malformed env values journal + fall
    through to the default rather than raising — we're in a
    fire-and-forget path and stranding the TaskRecord in ``running``
    would be worse than ignoring a typo.
    """
    if override is not None:
        return override if override > 0 else None
    raw = os.environ.get(_TIMEOUT_ENV_VAR)
    if raw is not None:
        try:
            parsed = float(raw)
        except ValueError:
            journal.write(
                "subagent_timeout_env_invalid",
                var=_TIMEOUT_ENV_VAR,
                value=raw,
            )
        else:
            return parsed if parsed > 0 else None
    return DEFAULT_SUBAGENT_TIMEOUT_SEC


def make_token_observer(store: TasksStore, task_id: str) -> Any:
    """Build a post_model hook that forwards usage_metadata into the store.

    The hook is forgiving: missing ``usage_metadata`` (FakeChatModel et
    al.) is a no-op; an exception from the observer journals + returns
    so the child's astream keeps going.
    """
    async def _observe(
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],  # noqa: ARG001 - protocol compliance
        state: Any,  # noqa: ARG001
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


def flush_transcript(
    *,
    transcript_storage: SessionStorage,
    task_id: str,
    messages: list[BaseMessage],
    store: TasksStore,
    parent_session_id: str = "",
    cwd: str = "",
) -> Path | None:
    """Write the child's transcript JSONL via the storage's path API."""
    try:
        register = getattr(
            transcript_storage, "write_subagent_transcript", None,
        )
        if not callable(register):
            return None
        cwd_arg: Path | None = Path(cwd) if cwd else None
        parent_arg: str | None = parent_session_id or None
        path: Path = register(
            task_id,
            messages,
            parent_session_id=parent_arg,
            cwd=cwd_arg,
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


def maybe_cleanup_completed_transcript(
    *,
    agent: Any,
    transcript_storage: SessionStorage,
    task_id: str,
    parent_session_id: str,
    cwd: str,
) -> None:
    """Delete a successfully-completed subagent's transcript + meta files.

    Opt-in via ``ToolsConfig.cleanup_completed_subagent_transcripts``;
    default is False (claude-code parity — files stay so /resume +
    post-mortem inspection still work). Only success calls this;
    failed / cancelled / timeout transcripts always survive.
    """
    try:
        cfg = getattr(agent, "_config", None)
        tools_cfg = getattr(cfg, "tools", None) if cfg is not None else None
        if not getattr(tools_cfg, "cleanup_completed_subagent_transcripts", False):
            return
        cwd_arg: Path | None = Path(cwd) if cwd else None
        parent_arg: str | None = parent_session_id or None
        for fn_name in ("subagent_transcript_path", "subagent_metadata_path"):
            path_fn = getattr(transcript_storage, fn_name, None)
            if not callable(path_fn):
                continue
            target: Path = path_fn(
                task_id, parent_session_id=parent_arg, cwd=cwd_arg,
            )
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                journal.write(
                    "subagent_transcript_cleanup_error",
                    task_id=task_id,
                    file=fn_name,
                    error=f"{type(exc).__name__}: {exc}",
                )
        journal.write(
            "subagent_transcript_cleaned",
            task_id=task_id,
            parent_session_id=parent_arg or "",
        )
    except Exception as exc:  # noqa: BLE001
        journal.write(
            "subagent_transcript_cleanup_error",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )


def flush_metadata(
    *,
    transcript_storage: SessionStorage,
    record: TaskRecord,
    parent_session_id: str,
    cwd: str,
) -> Path | None:
    """Write the ``.meta.json`` companion file alongside the transcript."""
    try:
        path_fn = getattr(transcript_storage, "subagent_metadata_path", None)
        if path_fn is None:
            journal.write(
                "subagent_metadata_skipped",
                task_id=record.id,
                reason="storage_missing_subagent_metadata_path",
            )
            return None
        cwd_arg: Path | None = Path(cwd) if cwd else None
        parent_arg: str | None = parent_session_id or None
        meta_path: Path = path_fn(
            record.id,
            parent_session_id=parent_arg,
            cwd=cwd_arg,
        )
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        started_iso = datetime.fromtimestamp(
            record.started_at, tz=UTC,
        ).isoformat()
        ended_iso: str | None = None
        if record.finished_at is not None:
            ended_iso = datetime.fromtimestamp(
                record.finished_at, tz=UTC,
            ).isoformat()
        meta = {
            "agent_type": record.agent_type or "general-purpose",
            "task_id": record.id,
            "description": record.description,
            "model_spec": record.model_spec or "",
            "parent_session_id": parent_session_id or "",
            "cwd": cwd or "",
            "started_at": record.started_at,
            "started_at_iso": started_iso,
            "ended_at": record.finished_at,
            "ended_at_iso": ended_iso,
            "status": record.status,
            "input_tokens": record.progress.input_tokens,
            "output_tokens": record.progress.output_tokens,
        }
        tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        tmp.replace(meta_path)
        return meta_path
    except Exception as exc:  # noqa: BLE001
        journal.write(
            "subagent_metadata_flush_error",
            task_id=record.id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return None


async def capture_child_messages(
    agent: Any, store: TasksStore, task_id: str,
) -> None:
    """Pull the child's full message list onto the TaskRecord."""
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
        rec.messages = list(msgs)
    except Exception as exc:  # noqa: BLE001
        journal.write(
            "subagent_capture_messages_error",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )


def load_child_messages(
    agent: Any, store: TasksStore, task_id: str,
) -> list[BaseMessage]:
    """Read the child's transcript for transcript-flush."""
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


class LocalAgentTask:
    """In-process subagent runner — one prompt, one terminal outcome.

    Use via :meth:`start` to schedule the asyncio task, :meth:`abort` to
    cancel it, and :meth:`wait_for_terminal` to await its natural
    completion. The runner publishes terminal state onto the
    :class:`TasksStore` so external listeners (parent Agent's task
    notifications) see the same record regardless of which runner
    topology was used.
    """

    def __init__(
        self,
        *,
        store: TasksStore,
        factory: SubagentFactory,
        task_id: str,
        timeout_sec: float | None = None,
        transcript_storage: SessionStorage | None = None,
        summary_interval_sec: float | None = None,
        parent_session_id: str | None = None,
        cwd: str | None = None,
    ) -> None:
        self._store = store
        self._factory = factory
        self._task_id = task_id
        self._timeout_sec = timeout_sec
        self._transcript_storage = transcript_storage
        self._summary_interval_sec = summary_interval_sec
        self._parent_session_id = parent_session_id
        self._cwd = cwd
        self._task: asyncio.Task[None] | None = None

    def start(self) -> asyncio.Task[None]:
        """Schedule the runner coroutine; return the handle.

        Idempotent — a second ``start()`` returns the existing handle.
        """
        if self._task is None:
            self._task = asyncio.create_task(
                self._run(),
                name=f"aura-local-agent-{self._task_id[:8]}",
            )
        return self._task

    async def wait_for_terminal(self) -> None:
        """Await the scheduled task. No-op if never started or already done."""
        if self._task is None:
            return
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    def abort(self) -> None:
        """Cancel the scheduled task. Idempotent — done tasks ignore the cancel."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def _run(self) -> None:
        await _run_local_agent(
            store=self._store,
            factory=self._factory,
            task_id=self._task_id,
            timeout_sec=self._timeout_sec,
            transcript_storage=self._transcript_storage,
            summary_interval_sec=self._summary_interval_sec,
            parent_session_id=self._parent_session_id,
            cwd=self._cwd,
        )


async def _run_local_agent(
    *,
    store: TasksStore,
    factory: SubagentFactory,
    task_id: str,
    timeout_sec: float | None,
    transcript_storage: SessionStorage | None,
    summary_interval_sec: float | None,
    parent_session_id: str | None,
    cwd: str | None,
) -> None:
    """Body of the local-agent run.

    Kept as a module-level function (not an instance method) so the
    legacy ``run_task`` thin wrapper in :mod:`aura.application.tasks.run` can
    call it directly without instantiating :class:`LocalAgentTask` —
    preserves existing callers that scheduled ``asyncio.create_task(
    run_task(...))`` without a class wrapper.
    """
    record = store.get(task_id)
    if record is None:
        return
    resolved_parent_session_id = parent_session_id or ""
    resolved_cwd = cwd or os.getcwd()
    effective_timeout = resolve_timeout(timeout_sec)
    # F-07-005 — parent abort cascade. Watcher coroutine fires the
    # local task's cancel when the parent's AbortController fires.
    parent_abort = factory.abort_event
    abort_watcher: asyncio.Task[None] | None = None
    if parent_abort is not None:
        current_task = asyncio.current_task()

        async def _watch_parent_abort() -> None:
            assert parent_abort is not None
            await parent_abort.wait()
            if current_task is not None and not current_task.done():
                current_task.cancel()

        abort_watcher = asyncio.create_task(
            _watch_parent_abort(),
            name=f"aura-subagent-abort-watch-{task_id[:8]}",
        )

    start_monotonic = time.monotonic()
    journal.write(
        "subagent_start",
        task_id=task_id,
        agent_type=record.agent_type or "general-purpose",
        prompt_chars=len(record.prompt),
    )
    agent: Any = None
    final_text = ""
    summarizer: Any = None
    try:
        # spawn() may raise on agent_type/model errors — keep it inside
        # the try so a spawn-time failure flips the record to ``failed``
        # rather than stranding it in ``running``.
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
        child_hooks = getattr(agent, "_hooks", None)
        if child_hooks is not None and hasattr(child_hooks, "post_model"):
            child_hooks.post_model.append(make_token_observer(store, task_id))
        # Round 7QS — periodic summary (cheap-model digest tick).
        from aura.application.services.agent_summary import AgentSummarizer
        from aura.infrastructure import llm as _llm_mod

        _make_summary_factory = getattr(
            _llm_mod, "make_summary_model_factory", None,
        )
        if _make_summary_factory is not None:
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
        async with asyncio.timeout(effective_timeout):
            async for event in agent.astream(record.prompt):
                if isinstance(event, ToolCallStarted):
                    store.record_activity(task_id, event.name)
                elif isinstance(event, Final):
                    final_text = event.message
        await capture_child_messages(agent, store, task_id)
        store.mark_completed(task_id, final_text)
    except asyncio.CancelledError:
        await capture_child_messages(agent, store, task_id)
        store.mark_cancelled(task_id)
        journal.write(
            "subagent_cancelled",
            task_id=task_id,
            duration_sec=round(time.monotonic() - start_monotonic, 3),
        )
        if transcript_storage is not None:
            flush_transcript(
                transcript_storage=transcript_storage,
                task_id=task_id,
                messages=load_child_messages(agent, store, task_id),
                store=store,
                parent_session_id=resolved_parent_session_id,
                cwd=resolved_cwd,
            )
            flush_metadata(
                transcript_storage=transcript_storage,
                record=record,
                parent_session_id=resolved_parent_session_id,
                cwd=resolved_cwd,
            )
        if summarizer is not None:
            await summarizer.stop()
        if agent is not None:
            await agent.aclose()
        raise
    except TimeoutError as exc:
        await capture_child_messages(agent, store, task_id)
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
            cause=f"{type(exc).__name__}: {exc}",
        )
        if transcript_storage is not None:
            flush_transcript(
                transcript_storage=transcript_storage,
                task_id=task_id,
                messages=load_child_messages(agent, store, task_id),
                store=store,
                parent_session_id=resolved_parent_session_id,
                cwd=resolved_cwd,
            )
            flush_metadata(
                transcript_storage=transcript_storage,
                record=record,
                parent_session_id=resolved_parent_session_id,
                cwd=resolved_cwd,
            )
        if summarizer is not None:
            await summarizer.stop()
        if agent is not None:
            await agent.aclose()
    except Exception as exc:  # noqa: BLE001
        if agent is not None:
            await capture_child_messages(agent, store, task_id)
        err_msg = f"{type(exc).__name__}: {exc}"
        store.mark_failed(task_id, err_msg)
        journal.write(
            "subagent_failed",
            task_id=task_id,
            error=err_msg,
            duration_sec=round(time.monotonic() - start_monotonic, 3),
        )
        if transcript_storage is not None and agent is not None:
            flush_transcript(
                transcript_storage=transcript_storage,
                task_id=task_id,
                messages=load_child_messages(agent, store, task_id),
                store=store,
                parent_session_id=resolved_parent_session_id,
                cwd=resolved_cwd,
            )
            flush_metadata(
                transcript_storage=transcript_storage,
                record=record,
                parent_session_id=resolved_parent_session_id,
                cwd=resolved_cwd,
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
            flush_transcript(
                transcript_storage=transcript_storage,
                task_id=task_id,
                messages=load_child_messages(agent, store, task_id),
                store=store,
                parent_session_id=resolved_parent_session_id,
                cwd=resolved_cwd,
            )
            flush_metadata(
                transcript_storage=transcript_storage,
                record=record,
                parent_session_id=resolved_parent_session_id,
                cwd=resolved_cwd,
            )
            maybe_cleanup_completed_transcript(
                agent=agent,
                transcript_storage=transcript_storage,
                task_id=task_id,
                parent_session_id=resolved_parent_session_id,
                cwd=resolved_cwd,
            )
        if summarizer is not None:
            await summarizer.stop()
        if agent is not None:
            await agent.aclose()
    finally:
        if abort_watcher is not None and not abort_watcher.done():
            abort_watcher.cancel()


__all__ = [
    "DEFAULT_SUBAGENT_TIMEOUT_SEC",
    "LocalAgentTask",
    "capture_child_messages",
    "flush_metadata",
    "flush_transcript",
    "load_child_messages",
    "make_token_observer",
    "maybe_cleanup_completed_transcript",
    "resolve_timeout",
]
