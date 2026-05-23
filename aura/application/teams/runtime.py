"""Teammate runtime — the long-lived loop that drives one teammate Agent.

Spawned once by :meth:`TeamManager.add_member`; exits on ``stop_event.set()``
(graceful) or ``abort.abort()`` (cascade from leader). Two entry points:

- ``InProcessBackend.spawn`` schedules :func:`run_teammate` on the leader's
  loop.
- ``PaneBackend.spawn`` runs :func:`run_teammate_main` inside a subprocess.

Loop: seed prompt (if any) → wait for new-message signal → ack unseen →
on ``shutdown_request`` confirm + exit cleanly, else feed envelope-wrapped
messages to ``Agent.astream``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

from aura.application.teams.mailbox import (
    FileMailboxNotifier,
    Mailbox,
    MailboxNotifier,
)
from aura.domain.abort import AbortController, AbortException
from aura.domain.team import TeamMessage
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage
from aura.schemas.events import Final, PermissionAudit, ToolCallProgress, ToolCallStarted

if TYPE_CHECKING:
    from aura.application.tasks.store import TasksStore
    from aura.core.agent import Agent

# Mailbox wait slice (s). For QueueMailboxNotifier the wait returns
# instantly on signal so this only bounds the idle interval; for
# FileMailboxNotifier (pane subprocess) it caps a single poll window.
_WAIT_SLICE_SEC: float = 5.0


def _task_tracking(agent: Agent) -> tuple[TasksStore, str] | None:
    store = getattr(agent, "_teammate_tasks_store", None)
    task_id = getattr(agent, "_teammate_task_id", None)
    if store is None or not isinstance(task_id, str) or task_id == "":
        return None
    return store, task_id


def _record_teammate_note(agent: Agent, activity: str) -> None:
    tracking = _task_tracking(agent)
    if tracking is None:
        return
    store, task_id = tracking
    store.record_activity_note(task_id, activity)


def _format_envelope(messages: list[TeamMessage]) -> str:
    """Wrap each TeamMessage into ``<from-{sender}>…</from-{sender}>``."""
    return "\n\n".join(
        f"<from-{m.sender}>\n{m.body}\n</from-{m.sender}>" for m in messages
    )


async def _drive_one_turn(
    *,
    agent: Agent,
    prompt: str,
    abort: AbortController,
    storage: SessionStorage,
    team_id: str,
    member_name: str,
) -> str:
    """Run one ``Agent.astream`` pass; return the Final text.

    Streams a one-line digest per event into the teammate transcript and
    forwards tool / permission activity to the TasksStore. ``AbortException``
    propagates with a journal entry; ``Final`` is the natural-completion
    marker.
    """
    final_text = ""
    transcript = storage.team_transcript_path(team_id, member_name)
    try:
        _record_teammate_note(agent, "turn_started")
        async for event in agent.astream(prompt, abort=abort):
            tracking = _task_tracking(agent)
            if tracking is not None:
                store, task_id = tracking
                if isinstance(event, ToolCallStarted):
                    store.record_activity(task_id, event.name)
                elif isinstance(event, ToolCallProgress):
                    chunk = event.chunk.strip()
                    if chunk:
                        store.record_activity_note(
                            task_id, f"{event.name}:{event.stream}> {chunk}",
                        )
                elif isinstance(event, PermissionAudit):
                    store.record_activity_note(task_id, f"permission:{event.tool}")
            with (
                contextlib.suppress(OSError),
                transcript.open("a", encoding="utf-8") as f,
            ):
                f.write(f"{int(time.time())} {type(event).__name__} ")
                if isinstance(event, Final):
                    f.write(event.message[:500])
                f.write("\n")
            if isinstance(event, Final):
                _record_teammate_note(agent, "final")
                final_text = event.message
    except AbortException:
        journal.write("team_runtime_aborted", team_id=team_id, member=member_name)
        raise
    return final_text


async def _wait_for_message(
    notifier: MailboxNotifier,
    member_name: str,
    stop_event: asyncio.Event,
    timeout: float,
) -> bool:
    """Await ``notifier`` OR ``stop_event``; return ``True`` if a message wins.

    ``asyncio.wait`` races the two so a stop set during an idle window
    exits the loop at the next iteration without waiting out the full
    slice. Cancellation of either pending task is suppressed.
    """
    wait_task = asyncio.create_task(notifier.wait_new(member_name, timeout=timeout))
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        done, pending = await asyncio.wait(
            {wait_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for t in (wait_task, stop_task):
            if not t.done():
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
    if stop_task in done:
        return False
    return bool(wait_task.result())


async def run_teammate(
    *,
    agent: Agent,
    team_id: str,
    member_name: str,
    storage: SessionStorage,
    stop_event: asyncio.Event,
    abort: AbortController,
    seed_prompt: str | None = None,
    notifier: MailboxNotifier | None = None,
) -> None:
    """Long-lived loop: drain mailbox, run agent, repeat.

    ``notifier`` decouples wake-up cadence from storage: callers pass a
    :class:`~aura.application.teams.mailbox.QueueMailboxNotifier` for
    in-process teammates (instant wake on signal) and let the default
    :class:`FileMailboxNotifier` poll the JSONL for pane subprocesses.

    Exit conditions: ``stop_event`` set (graceful boundary), ``abort.aborted``
    propagates as ``AbortException``, or ``asyncio.CancelledError`` re-raises
    after best-effort cleanup. A ``shutdown_request`` resolves the manager's
    per-member ack future (in-process) and writes a ``shutdown_response`` to
    the leader inbox (pane fallback) before exiting.
    """
    mailbox = Mailbox(storage, team_id)
    if notifier is None:
        notifier = FileMailboxNotifier(mailbox)
    journal.write("team_runtime_started", team_id=team_id, member=member_name)
    try:
        if seed_prompt is not None and seed_prompt.strip():
            # Seed runs once without mailbox round-trip; not re-replayed on restart.
            with contextlib.suppress(AbortException):
                await _drive_one_turn(
                    agent=agent, prompt=seed_prompt, abort=abort,
                    storage=storage, team_id=team_id, member_name=member_name,
                )
        while not stop_event.is_set() and not abort.aborted:
            has_msg = await _wait_for_message(
                notifier, member_name, stop_event, _WAIT_SLICE_SEC,
            )
            if not has_msg:
                continue
            unseen = mailbox.read_unseen(member_name)
            if not unseen:
                continue
            mailbox.ack(member_name, [m.msg_id for m in unseen])
            shutdown = next(
                (m for m in unseen if m.kind == "shutdown_request"), None,
            )
            if shutdown is not None:
                journal.write(
                    "team_runtime_shutdown",
                    team_id=team_id, member=member_name, sender=shutdown.sender,
                )
                manager = getattr(agent, "team", None)
                if manager is not None and getattr(manager, "is_active", False):
                    # In-process: resolve the leader's per-member ack future
                    # directly — no inbox round-trip needed.
                    confirm = getattr(manager, "confirm_shutdown", None)
                    if callable(confirm):
                        with contextlib.suppress(Exception):
                            confirm(member_name, body=shutdown.body)
                    # Pane fallback: subprocess can't share futures with the
                    # leader, so the manager still observes the response via
                    # its inbox. ``send`` is idempotent w.r.t. confirm above.
                    with contextlib.suppress(Exception):
                        manager.send(
                            sender=member_name, recipient="leader",
                            body=f"shutting down: {shutdown.body}",
                            kind="shutdown_response",
                        )
                break
            text_msgs = [m for m in unseen if m.kind == "text"]
            if not text_msgs:
                continue
            try:
                await _drive_one_turn(
                    agent=agent, prompt=_format_envelope(text_msgs), abort=abort,
                    storage=storage, team_id=team_id, member_name=member_name,
                )
            except AbortException:
                break
            except Exception as exc:  # noqa: BLE001
                # A single turn failure shouldn't kill the teammate;
                # the leader can /team remove to escalate.
                journal.write(
                    "team_runtime_turn_failed",
                    team_id=team_id, member=member_name,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
    except asyncio.CancelledError:
        journal.write("team_runtime_cancelled", team_id=team_id, member=member_name)
        raise
    finally:
        journal.write("team_runtime_exited", team_id=team_id, member=member_name)


async def run_teammate_main(
    *,
    team_id: str,
    member_name: str,
    storage_root: str,
    agent_type: str = "general-purpose",
    model_name: str | None = None,
    system_prompt: str | None = None,
    seed_prompt: str | None = None,
) -> int:
    """Subprocess entrypoint for the pane backend.

    Builds a fresh Agent rooted at ``storage_root`` and drives
    :func:`run_teammate` against the same JSONL mailbox the leader writes
    to. Communication is exclusively via the on-disk mailbox + ``.seen``
    cursor — no IPC channel back to the leader.

    Returns ``0`` on clean shutdown, non-zero on startup failure.
    """
    # Lazy imports keep the in-process backend's hot path cold-start cheap.
    from pathlib import Path

    from aura.config.loader import load_config
    from aura.core.agent import build_agent
    from aura.domain.team import TeammateMember as _Member
    from aura.infrastructure.persistence.storage import SessionStorage as _Storage

    config = load_config()
    if model_name:
        # Override the router default so build_agent picks up the per-member spec.
        config = config.model_copy(
            update={"router": {**config.router, "default": model_name}},
        )
    storage = _Storage(Path(storage_root) / "index.sqlite")
    agent = build_agent(config, session_id=f"team-{team_id}-{member_name}")
    # Synthesize a TeammateMember so a future caller can pass a richer
    # record without changing this signature.
    _ = _Member(
        name=member_name, agent_type=agent_type,
        model_name=model_name, system_prompt=system_prompt,
    )
    stop_event = asyncio.Event()
    abort = AbortController()
    try:
        await run_teammate(
            agent=agent, team_id=team_id, member_name=member_name,
            storage=storage, stop_event=stop_event, abort=abort,
            seed_prompt=seed_prompt,
        )
    finally:
        with contextlib.suppress(Exception):
            await agent.aclose()
    return 0
