"""Lifecycle owner of one team per leader AgentSession: record, mailbox, runtime tasks."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from aura.application.session import AgentSession
from aura.application.tasks.spawn import SubagentSpawner
from aura.application.tasks.store import TasksStore
from aura.application.teams.lifecycle import LifecycleEmitter
from aura.application.teams.mailbox import (
    Mailbox,
    MailboxNotifier,
    QueueMailboxNotifier,
)
from aura.application.teams.runtime import run_teammate
from aura.application.teams.state import BackendHandleLike, Member, TeamError
from aura.application.teams.view import TeamViewBuilder
from aura.application.teams.view_types import TeammateMemberStatus, TeamViewSnapshot
from aura.domain.abort import AbortController
from aura.domain.task import TaskRecord
from aura.domain.team import (
    BROADCAST_RECIPIENT,
    MAX_BODY_CHARS,
    MAX_MEMBERS,
    TEAM_LEADER_NAME,
    BackendType,
    TeammateMember,
    TeamMessage,
    TeamMessageKind,
    TeamRecord,
)
from aura.domain.team_memory import redact_secrets
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.teams.in_process import InProcessBackend, InProcessHandle
from aura.infrastructure.teams.registry import BackendUnavailable, get_backend
from aura.infrastructure.teams.types import TeammateBackend
from aura.infrastructure.wire.events import CoordinationEvent
from aura.infrastructure.wire.serialize import team_message_to_wire

_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]+$")


# Only the in-process backend handle carries an asyncio task to await; pane
# handles run out-of-process and expose none.
@runtime_checkable
class _HasTask(Protocol):
    task: asyncio.Task[None]


class TeammateRunner(Protocol):
    def __call__(
        self,
        *,
        agent: AgentSession,
        team_id: str,
        member_name: str,
        storage: SessionStorage,
        stop_event: asyncio.Event,
        abort: AbortController,
        seed_prompt: str | None = None,
        notifier: MailboxNotifier | None = None,
    ) -> Coroutine[Any, Any, None]: ...


@dataclass(frozen=True)
class _PreparedMember:
    """Common prefix of (a)add_member: validated, persisted, spawned, pre-dispatch."""

    member: TeammateMember
    record: TaskRecord
    child: AgentSession
    abort: AbortController
    stop_event: asyncio.Event
    backend: TeammateBackend
    slot: Member
    team_id: str


def _slugify(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", raw.strip()).strip("-_")
    if not cleaned:
        raise TeamError(f"name {raw!r} has no slugifiable characters")
    return cleaned


class TeamManager:
    def __init__(
        self,
        *,
        leader: AgentSession,
        storage: SessionStorage,
        factory: SubagentSpawner[AgentSession],
        running_aborts: dict[str, AbortController],
        tasks_store: TasksStore,
        runtime_runner: TeammateRunner | None = None,
    ) -> None:
        self._leader = leader
        self._storage = storage
        self._factory = factory
        self._running_aborts = running_aborts
        self._tasks_store = tasks_store
        self._runtime_runner = (
            run_teammate if runtime_runner is None else runtime_runner
        )
        self._team: TeamRecord | None = None
        self._runtimes: dict[str, asyncio.Task[None]] = {}
        self._teardown_tasks: set[asyncio.Task[object]] = set()
        self._members: dict[str, Member] = {}
        self._mailbox_notifier = QueueMailboxNotifier()
        self._teammate_terminal_intents: dict[str, str] = {}
        self._session_created_teams: set[str] = set()
        self._lifecycle = LifecycleEmitter(
            team=lambda: self._team, members=self._members,
        )
        self._view = TeamViewBuilder(
            team=lambda: self._team,
            members=self._members,
            storage=storage,
            tasks_store=tasks_store,
        )

    @property
    def team(self) -> TeamRecord | None:
        return self._team

    @property
    def is_active(self) -> bool:
        return self._team is not None

    @property
    def pending_protocol_events(self) -> tuple[CoordinationEvent, ...]:
        return self._lifecycle.pending

    @property
    def storage(self) -> SessionStorage:
        return self._storage

    def post_message(self, msg: TeamMessage) -> None:
        """Public surface for sibling-pane shutdown signals; mirrors :meth:`_post`."""
        self._post(msg)

    def mailbox(self) -> Mailbox:
        if self._team is None:
            raise TeamError("no team is active")
        return Mailbox(self._storage, self._team.team_id)

    def create_team(self, name: str) -> TeamRecord:
        if self._team is not None:
            raise TeamError(
                f"one team per leader: {self._team.team_id!r} is already active",
            )
        team_id = _slugify(name)
        existing = set(self._storage.list_team_ids())
        if team_id in existing:
            suffix = 2
            while f"{team_id}-{suffix}" in existing:
                suffix += 1
            team_id = f"{team_id}-{suffix}"
        record = TeamRecord(
            team_id=team_id,
            name=name,
            leader_session_id=self._leader.session_id,
            cwd=str(self._leader.cwd),
        )
        self._team = record
        self._persist()
        self._session_created_teams.add(team_id)
        self._lifecycle.reset_team_state()
        self._members.clear()
        self._lifecycle.emit_team("active")
        journal.write(
            "team_created",
            team_id=team_id,
            name=name,
            leader_session=self._leader.session_id,
        )
        return record

    def delete_team(self) -> None:
        if self._team is None:
            return
        # Snapshot before mutation: remove_member rewrites ``members`` in place.
        for member in list(self._team.members):
            with contextlib.suppress(TeamError):
                self.remove_member(member.name, force=True)
        team_id = self._team.team_id
        self._lifecycle.emit_team("draining")
        self._lifecycle.emit_team("terminated")
        self._session_created_teams.discard(team_id)
        team_dir = self._storage.team_root(team_id)
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(team_dir)
        self._team = None
        self._lifecycle.reset_team_state()
        self._members.clear()
        journal.write("team_deleted", team_id=team_id, dir_removed=True)

    async def cleanup_session_teams(self) -> None:
        if not self._session_created_teams:
            return
        team_ids = list(self._session_created_teams)
        for m in list(self._members.values()):
            if m.task_id is not None:
                self._mark_teammate_cancelled(m.task_id)
        for task in list(self._runtimes.values()):
            if not task.done():
                task.cancel()
        if self._runtimes:
            await asyncio.gather(
                *[t for t in self._runtimes.values() if not t.done()],
                return_exceptions=True,
            )
        backends = [m.backend for m in self._members.values() if m.backend is not None]
        if backends:
            # Pane backends own tmux panes; force_kill must run before rmtree.
            await asyncio.gather(
                *[h.force_kill() for h in backends],
                return_exceptions=True,
            )
            for m in self._members.values():
                m.backend = None
        for team_id in team_ids:
            team_dir = self._storage.team_root(team_id)
            try:
                shutil.rmtree(team_dir)
            except FileNotFoundError:
                pass
            except OSError as exc:
                journal.write(
                    "team_session_cleanup_error",
                    team_id=team_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            journal.write("team_session_cleanup", team_id=team_id)
            self._session_created_teams.discard(team_id)
        if self._team is not None and self._team.team_id not in self._session_created_teams:
            self._team = None
            self._runtimes.clear()
            for m in self._members.values():
                m.task_id = None
                m.agent = None
                m.stop_event = None
            self._teammate_terminal_intents.clear()

    def _prepare_member(
        self,
        name: str,
        *,
        agent_type: str,
        system_prompt: str | None,
        model_name: str | None,
        seed_prompt: str | None,
        backend_type: BackendType,
    ) -> _PreparedMember:
        if self._team is None:
            raise TeamError("no team is active; call create_team first")
        if name == TEAM_LEADER_NAME:
            raise TeamError(f"member name {name!r} is reserved for the leader")
        if name == BROADCAST_RECIPIENT:
            raise TeamError(
                f"member name {name!r} is reserved for broadcast routing",
            )
        if not _SLUG_RE.match(name):
            raise TeamError(
                f"member name {name!r} must match {_SLUG_RE.pattern}",
            )
        if any(m.name == name for m in self._team.members):
            raise TeamError(f"member {name!r} already exists in team")
        if len(self._team.members) >= MAX_MEMBERS:
            raise TeamError(
                f"team has reached MAX_MEMBERS={MAX_MEMBERS}; "
                "remove a member before adding another",
            )
        # Resolve before state mutation so an unsupported pane env fails fast.
        try:
            backend = get_backend(backend_type)
        except BackendUnavailable as exc:
            raise TeamError(str(exc)) from exc
        if model_name is not None:
            self._factory.validate_model_spec(model_name)
        member = TeammateMember(
            name=name,
            agent_type=agent_type,
            system_prompt=system_prompt,
            model_name=model_name,
            backend_type=backend_type,
        )
        self._team.members.append(member)
        self._persist()
        prompt_for_task = seed_prompt or "(idle teammate; awaiting messages)"
        task_model_spec = (
            model_name
            if model_name is not None
            else self._factory.parent_model_spec
        )
        record = self._tasks_store.create(
            description=f"teammate: {name}",
            prompt=prompt_for_task,
            kind="teammate",
            agent_type=agent_type,
            metadata={"team_id": self._team.team_id, "member": name},
            model_spec=task_model_spec,
        )
        child = self._factory.spawn(
            prompt_for_task,
            agent_type=agent_type,
            task_id=record.id,
            model_spec=model_name,
        )
        child.join_team(
            manager=self,
            member_name=name,
            task_id=record.id,
            tasks_store=self._tasks_store,
        )
        slot = self._members.setdefault(name, Member())
        slot.agent = child
        slot.task_id = record.id
        # Abort controller must register before runtime starts so a same-tick cascade finds it.
        abort = AbortController()
        self._running_aborts[record.id] = abort
        return _PreparedMember(
            member=member,
            record=record,
            child=child,
            abort=abort,
            stop_event=asyncio.Event(),
            backend=backend,
            slot=slot,
            team_id=self._team.team_id,
        )

    def _finalize_member(
        self,
        prepared: _PreparedMember,
        handle: BackendHandleLike,
        *,
        agent_type: str,
        backend_type: BackendType,
    ) -> TeammateMember:
        member = prepared.member
        prepared.slot.stop_event = prepared.stop_event
        prepared.slot.backend = handle
        if member.tmux_pane_id is not None:
            # Pane backend may have stamped tmux_pane_id during spawn.
            self._persist()
        self._lifecycle.emit_member(member.name, "ready")
        self._lifecycle.emit_member(member.name, "idle")
        journal.write(
            "team_member_added",
            team_id=prepared.team_id,
            member=member.name,
            agent_type=agent_type,
            backend_type=backend_type,
            task_id=prepared.record.id,
            tmux_pane_id=member.tmux_pane_id,
        )
        return member

    def add_member(
        self,
        name: str,
        *,
        agent_type: str = "general-purpose",
        system_prompt: str | None = None,
        model_name: str | None = None,
        seed_prompt: str | None = None,
        backend_type: BackendType = "in_process",
    ) -> TeammateMember:
        prepared = self._prepare_member(
            name,
            agent_type=agent_type,
            system_prompt=system_prompt,
            model_name=model_name,
            seed_prompt=seed_prompt,
            backend_type=backend_type,
        )
        if backend_type == "pane":
            raise TeamError(
                "pane backend must be added via aadd_member() from an "
                "async context (sync add_member supports in_process only)",
            )
        record = prepared.record
        child = prepared.child
        abort = prepared.abort
        stop_event = prepared.stop_event
        self._lifecycle.emit_member(name, "starting")
        handle: BackendHandleLike
        if self._runtime_runner is not run_teammate:
            # Tests inject a runner directly; bypass the backend dispatch.
            task: asyncio.Task[None] = asyncio.create_task(
                self._runtime_runner(
                    agent=child,
                    team_id=prepared.team_id,
                    member_name=name,
                    storage=self._storage,
                    stop_event=stop_event,
                    abort=abort,
                    seed_prompt=seed_prompt,
                ),
                name=f"aura-teammate-{name}",
            )
            self._register_runtime_task(task, record.id, abort)
            handle = InProcessHandle(
                task=task,
                stop_event=stop_event,
                abort=abort,
            )
        else:
            assert isinstance(prepared.backend, InProcessBackend)
            handle = prepared.backend.spawn_sync(
                team_id=prepared.team_id,
                member=prepared.member,
                agent=child,
                manager=self,
                storage=self._storage,
                stop_event=stop_event,
                abort=abort,
                seed_prompt=seed_prompt,
                notifier=self._mailbox_notifier,
            )
            self._register_runtime_task(handle.task, record.id, abort)
        return self._finalize_member(
            prepared,
            handle,
            agent_type=agent_type,
            backend_type=backend_type,
        )

    async def aadd_member(
        self,
        name: str,
        *,
        agent_type: str = "general-purpose",
        system_prompt: str | None = None,
        model_name: str | None = None,
        seed_prompt: str | None = None,
        backend_type: BackendType = "in_process",
    ) -> TeammateMember:
        if backend_type != "pane":
            return self.add_member(
                name,
                agent_type=agent_type,
                system_prompt=system_prompt,
                model_name=model_name,
                seed_prompt=seed_prompt,
                backend_type=backend_type,
            )
        prepared = self._prepare_member(
            name,
            agent_type=agent_type,
            system_prompt=system_prompt,
            model_name=model_name,
            seed_prompt=seed_prompt,
            backend_type=backend_type,
        )
        record = prepared.record
        abort = prepared.abort
        self._lifecycle.emit_member(name, "starting")
        handle = await prepared.backend.spawn(
            team_id=prepared.team_id,
            member=prepared.member,
            agent=prepared.child,
            manager=self,
            storage=self._storage,
            stop_event=prepared.stop_event,
            abort=abort,
            seed_prompt=seed_prompt,
            notifier=self._mailbox_notifier,
        )
        if isinstance(handle, _HasTask):
            self._register_runtime_task(handle.task, record.id, abort)
        return self._finalize_member(
            prepared,
            handle,
            agent_type=agent_type,
            backend_type=backend_type,
        )

    # 5 s matches the teammate runtime's mailbox-wait slice; below it fights the cadence.
    DEFAULT_SHUTDOWN_GRACE_SEC: float = 5.0

    def remove_member(
        self,
        name: str,
        *,
        force: bool = False,
        timeout_sec: float | None = None,
    ) -> None:
        if self._team is None:
            raise TeamError("no team is active")
        if not any(m.name == name for m in self._team.members):
            raise TeamError(f"member {name!r} not found in team")
        if force:
            self._teardown_member(name, send_request=False, journal_force=True)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            # No loop (sync unit tests): degrade to synchronous teardown.
            self._teardown_member(
                name, send_request=True, journal_force=True,
            )
            return
        waiter = loop.create_task(
            self.aremove_member(name, timeout_sec=timeout_sec),
            name=f"aura-team-shutdown-{name}",
        )
        self._members.setdefault(name, Member()).shutdown_waiter = waiter
        def _prune(_t: asyncio.Task[bool]) -> None:
            slot = self._members.get(name)
            if slot is not None:
                slot.shutdown_waiter = None
            with contextlib.suppress(Exception):
                if not _t.cancelled():
                    _t.exception()
        waiter.add_done_callback(_prune)

    async def aremove_member(
        self,
        name: str,
        *,
        force: bool = False,
        timeout_sec: float | None = None,
    ) -> bool:
        if self._team is None:
            raise TeamError("no team is active")
        if not any(m.name == name for m in self._team.members):
            raise TeamError(f"member {name!r} not found in team")
        if force:
            self._teardown_member(name, send_request=False, journal_force=True)
            return False
        timeout = (
            timeout_sec
            if timeout_sec is not None
            else self.DEFAULT_SHUTDOWN_GRACE_SEC
        )
        team_id = self._team.team_id
        # Drop row first; runtime must live long enough to resolve the ack future.
        idx = next(
            i for i, m in enumerate(self._team.members) if m.name == name
        )
        self._team.members.pop(idx)
        self._persist()
        self._lifecycle.emit_member(name, "draining")
        loop = asyncio.get_running_loop()
        ack_future: asyncio.Future[bool] = loop.create_future()
        slot = self._members.setdefault(name, Member())
        slot.shutdown_ack = ack_future
        with contextlib.suppress(Exception):
            self._post(TeamMessage(
                msg_id=uuid.uuid4().hex,
                sender=TEAM_LEADER_NAME,
                recipient=name,
                body="shutdown",
                kind="shutdown_request",
            ))
        stop_event = slot.stop_event
        if stop_event is not None:
            if slot.task_id is not None:
                self._set_teammate_cancel_intent(slot.task_id)
            stop_event.set()
        try:
            acked = await asyncio.wait_for(
                asyncio.shield(ack_future), timeout=timeout,
            )
        except TimeoutError:
            acked = False
        finally:
            ack_slot = self._members.get(name)
            if ack_slot is not None:
                ack_slot.shutdown_ack = None
        if acked:
            journal.write(
                "team_member_shutdown_ack_received",
                team_id=team_id,
                member=name,
            )
            self._lifecycle.emit_member(name, "terminated")
            self._teardown_member(
                name,
                send_request=False,
                journal_force=False,
                already_acked=True,
            )
            return True
        journal.write(
            "team_member_shutdown_force_killed",
            team_id=team_id,
            member=name,
            timeout_sec=timeout,
        )
        self._lifecycle.emit_member(name, "terminated", reason="forced_timeout")
        self._teardown_member(name, send_request=False, journal_force=True)
        return False

    def confirm_shutdown(self, member_name: str, *, body: str = "") -> None:
        del body
        slot = self._members.get(member_name)
        fut = None if slot is None else slot.shutdown_ack
        if fut is None or fut.done():
            return
        with contextlib.suppress(Exception):
            fut.set_result(True)

    def _teardown_member(
        self,
        name: str,
        *,
        send_request: bool,
        journal_force: bool,
        already_acked: bool = False,
    ) -> None:
        if self._team is None:
            return
        idx = next(
            (i for i, m in enumerate(self._team.members) if m.name == name),
            -1,
        )
        if idx >= 0:
            self._team.members.pop(idx)
            self._persist()
        # Runtime handles drop here; lifecycle_state + shutdown_waiter outlive teardown.
        slot = self._members.get(name) or Member()
        task_id, slot.task_id = slot.task_id, None
        stop_event, slot.stop_event = slot.stop_event, None
        pending_ack, slot.shutdown_ack = slot.shutdown_ack, None
        backend_handle, slot.backend = slot.backend, None
        agent, slot.agent = slot.agent, None
        if pending_ack is not None and not pending_ack.done():
            with contextlib.suppress(Exception):
                pending_ack.set_result(False)
        if task_id is not None:
            self._mark_teammate_cancelled(task_id)
        if send_request and task_id is not None:
            with contextlib.suppress(Exception):
                self._post(TeamMessage(
                    msg_id=uuid.uuid4().hex,
                    sender=TEAM_LEADER_NAME,
                    recipient=name,
                    body="shutdown",
                    kind="shutdown_request",
                ))
        if stop_event is not None:
            stop_event.set()
        if task_id is not None and not already_acked:
            controller = self._running_aborts.get(task_id)
            if controller is not None and not controller.aborted:
                controller.abort("teammate_removed")
            self._running_aborts.pop(task_id, None)
            handle = self._runtimes.get(task_id)
            if handle is not None and not handle.done():
                handle.cancel()
        if backend_handle is not None and not already_acked:
            with contextlib.suppress(Exception):
                # force_kill is async; skip when no loop runs (task cancel above suffices).
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None:
                    self._track_teardown(backend_handle.force_kill())
        if agent is not None:
            with contextlib.suppress(Exception):
                self._track_teardown(agent.aclose())
        if journal_force:
            journal.write(
                "team_member_removed",
                team_id=self._team.team_id,
                member=name,
                forced=not already_acked,
            )

    def _track_teardown(self, coro: Coroutine[Any, Any, object]) -> None:
        # Hold a ref so the fire-and-forget teardown task isn't GC'd mid-flight.
        task = asyncio.ensure_future(coro)
        self._teardown_tasks.add(task)
        task.add_done_callback(self._teardown_tasks.discard)

    def _set_teammate_cancel_intent(self, task_id: str) -> None:
        if task_id in self._runtimes:
            self._teammate_terminal_intents[task_id] = "cancelled"

    def _mark_teammate_cancelled(self, task_id: str) -> None:
        self._set_teammate_cancel_intent(task_id)
        self._tasks_store.mark_cancelled(task_id)

    def _register_runtime_task(
        self,
        task: asyncio.Task[None],
        task_id: str,
        abort: AbortController,
    ) -> None:
        """Track the teammate task; finalize (drop refs) when it exits."""
        task.add_done_callback(
            lambda t: self._finalize_runtime_task(task_id, t, abort)
        )
        self._runtimes[task_id] = task

    def _finalize_runtime_task(
        self,
        task_id: str,
        task: asyncio.Task[None],
        abort: AbortController,
    ) -> None:
        self._runtimes.pop(task_id, None)
        self._running_aborts.pop(task_id, None)
        intent = self._teammate_terminal_intents.pop(task_id, None)
        if intent == "cancelled" or task.cancelled() or abort.aborted:
            self._tasks_store.mark_cancelled(task_id)
            return
        exc = task.exception()
        if exc is not None:
            self._tasks_store.mark_failed(
                task_id,
                f"{type(exc).__name__}: {exc}",
            )
            return
        self._tasks_store.mark_completed(task_id, "")

    def list_members(self) -> list[TeammateMember]:
        if self._team is None:
            return []
        return list(self._team.members)

    def view_state(self, team_id: str | None = None) -> TeamViewSnapshot:
        return self._view.view_state(team_id)

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        body: str,
        kind: TeamMessageKind = "text",
    ) -> list[TeamMessage]:
        if self._team is None:
            raise TeamError("no team is active")
        if not body.strip():
            raise TeamError("body must contain at least one non-whitespace char")
        if len(body) > MAX_BODY_CHARS:
            raise TeamError(
                f"body length {len(body)} exceeds MAX_BODY_CHARS={MAX_BODY_CHARS}",
            )
        # Redact before any persistence: JSONL inbox is the highest-leak surface.
        body = redact_secrets(body)
        recipients: list[str]
        if recipient == BROADCAST_RECIPIENT:
            recipients = [m.name for m in self._team.members]
            if not recipients:
                raise TeamError("broadcast: team has no members")
        elif recipient == TEAM_LEADER_NAME:
            recipients = [TEAM_LEADER_NAME]
        else:
            if not any(m.name == recipient for m in self._team.members):
                raise TeamError(f"unknown recipient {recipient!r}")
            recipients = [recipient]
        sent: list[TeamMessage] = []
        mailbox = self.mailbox()
        for rcpt in recipients:
            msg = TeamMessage(
                msg_id=uuid.uuid4().hex,
                sender=sender,
                recipient=rcpt,
                body=body,
                kind=kind,
            )
            mailbox.append(msg)
            self._mailbox_notifier.signal(rcpt)
            sent.append(msg)
            self._lifecycle.append(
                team_message_to_wire(msg, team_id=self._team.team_id),
            )
        return sent

    def _post(self, msg: TeamMessage) -> None:
        # Internal shutdown plumbing; UI shows teardown via member_lifecycle, not raw mailbox.
        msg = msg.model_copy(update={"body": redact_secrets(msg.body)})
        self.mailbox().append(msg)
        self._mailbox_notifier.signal(msg.recipient)

    def drain_protocol_events(self) -> list[CoordinationEvent]:
        return self._lifecycle.drain()

    def _persist(self) -> None:
        if self._team is None:
            return
        path = self._storage.team_config_path(self._team.team_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = self._team.model_dump_json(indent=2)
        try:
            with tmp.open("w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                with contextlib.suppress(OSError):
                    os.fsync(f.fileno())
            tmp.replace(path)
        except OSError as exc:
            # Transient FS failure surfaces via journal; agent stays alive.
            journal.write(
                "team_persist_failed",
                team_id=self._team.team_id,
                error=f"{type(exc).__name__}: {exc}",
            )

__all__ = [
    "Member",
    "TeamError",
    "TeamManager",
    "TeammateMemberStatus",
    "TeamViewSnapshot",
]
