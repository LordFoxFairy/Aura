"""In-process child AgentSession runner; fire-and-forget, cancellation via :meth:`abort`."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time

from aura.application.tasks.runners.local_agent_io import (
    capture_child_messages,
    flush_metadata,
    flush_transcript,
    load_child_messages,
    maybe_cleanup_completed_transcript,
)
from aura.application.tasks.runners.local_agent_support import (
    DEFAULT_SUBAGENT_TIMEOUT_SEC,
    make_token_observer,
    resolve_timeout,
)
from aura.application.tasks.spawn_port import SpawnedAgent, SpawnPort
from aura.application.tasks.store import TasksStore
from aura.domain.events import Final, ToolCallStarted
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage


class LocalAgentTask:
    """In-process subagent runner — one prompt, one terminal outcome."""

    def __init__(
        self,
        *,
        store: TasksStore,
        factory: SpawnPort,
        task_id: str,
        timeout_sec: float | None = None,
        transcript_storage: SessionStorage | None = None,
        parent_session_id: str | None = None,
        cwd: str | None = None,
    ) -> None:
        self._store = store
        self._factory = factory
        self._task_id = task_id
        self._timeout_sec = timeout_sec
        self._transcript_storage = transcript_storage
        self._parent_session_id = parent_session_id
        self._cwd = cwd
        self._task: asyncio.Task[None] | None = None

    def start(self) -> asyncio.Task[None]:
        """Schedule the runner coroutine; idempotent."""
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
        await run_local_agent(
            store=self._store,
            factory=self._factory,
            task_id=self._task_id,
            timeout_sec=self._timeout_sec,
            transcript_storage=self._transcript_storage,
            parent_session_id=self._parent_session_id,
            cwd=self._cwd,
        )


async def run_local_agent(
    *,
    store: TasksStore,
    factory: SpawnPort,
    task_id: str,
    timeout_sec: float | None,
    transcript_storage: SessionStorage | None,
    parent_session_id: str | None,
    cwd: str | None,
) -> None:
    """Body of the local-agent run; module-level so :func:`run_task` calls it directly."""
    record = store.get(task_id)
    if record is None:
        return
    resolved_parent_session_id = parent_session_id or ""
    resolved_cwd = cwd or os.getcwd()
    effective_timeout = resolve_timeout(timeout_sec)
    # Parent abort cascade: cancel the local task when the parent's Event fires.
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
    store.record_started(task_id)
    agent: SpawnedAgent | None = None
    final_text = ""

    async def _finalize_run(*, require_agent: bool, cleanup: bool) -> None:
        if transcript_storage is not None and (agent is not None or not require_agent):
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
            if cleanup and agent is not None:
                maybe_cleanup_completed_transcript(
                    agent=agent,
                    transcript_storage=transcript_storage,
                    task_id=task_id,
                    parent_session_id=resolved_parent_session_id,
                    cwd=resolved_cwd,
                )
        if agent is not None:
            await agent.aclose()

    try:
        agent = factory.spawn(
            record.prompt,
            agent_type=record.agent_type or "general-purpose",
            task_id=task_id,
            model_spec=record.model_spec or None,
        )
        agent.hooks.post_model.append(make_token_observer(store, task_id))
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
        await _finalize_run(require_agent=False, cleanup=False)
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
        await _finalize_run(require_agent=False, cleanup=False)
    except Exception as exc:  # noqa: BLE001  # cleanup path must not propagate
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
        await _finalize_run(require_agent=True, cleanup=False)
    else:
        journal.write(
            "subagent_completed",
            task_id=task_id,
            duration_sec=round(time.monotonic() - start_monotonic, 3),
            final_text_chars=len(final_text),
        )
        await _finalize_run(require_agent=True, cleanup=True)
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
