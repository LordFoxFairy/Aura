"""Long-lived loop driving one teammate AgentSession against its JSONL mailbox."""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

from aura.application.session import AgentSession, build_agent
from aura.application.tasks.store import TasksStore
from aura.application.teams.mailbox import (
    FileMailboxNotifier,
    Mailbox,
    MailboxNotifier,
)
from aura.application.teams.team_port import TeamPort
from aura.config.loader import load_config
from aura.domain.abort import AbortController, AbortException
from aura.domain.events import Final, PermissionAudit, ToolCallProgress, ToolCallStarted
from aura.domain.team import TeamMessage
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage

_WAIT_SLICE_SEC: float = 5.0


def _task_tracking(agent: AgentSession) -> tuple[TasksStore, str] | None:
    binding = agent.teammate
    if binding is None:
        return None
    return binding.tasks_store, binding.task_id


def _record_teammate_note(agent: AgentSession, activity: str) -> None:
    tracking = _task_tracking(agent)
    if tracking is None:
        return
    store, task_id = tracking
    store.record_activity_note(task_id, activity)


def _format_envelope(messages: list[TeamMessage]) -> str:
    return "\n\n".join(
        f"<from-{m.sender}>\n{m.body}\n</from-{m.sender}>" for m in messages
    )


async def _drive_one_turn(
    *,
    agent: AgentSession,
    prompt: str,
    abort: AbortController,
    storage: SessionStorage,
    team_id: str,
    member_name: str,
) -> str:
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
    wait_task = asyncio.create_task(notifier.wait_new(member_name, timeout=timeout))
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        done, _ = await asyncio.wait(
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
    agent: AgentSession,
    team_id: str,
    member_name: str,
    storage: SessionStorage,
    stop_event: asyncio.Event,
    abort: AbortController,
    seed_prompt: str | None = None,
    notifier: MailboxNotifier | None = None,
) -> None:
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
                manager: TeamPort | None = agent.team
                if manager is not None and manager.is_active:
                    # In-process: future ack; pane subprocess: inbox round-trip.
                    with contextlib.suppress(Exception):
                        manager.confirm_shutdown(member_name, body=shutdown.body)
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
                # Per-turn failure logged but does not tear down the teammate.
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
    config = load_config()
    if model_name:
        config = config.model_copy(
            update={"router": {**config.router, "default": model_name}},
        )
    storage = SessionStorage(Path(storage_root) / "index.sqlite")
    agent = build_agent(config, session_id=f"team-{team_id}-{member_name}")
    del agent_type, system_prompt
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
