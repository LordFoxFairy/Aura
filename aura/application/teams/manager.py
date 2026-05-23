"""TeamManager — owns the lifecycle of one team per leader Agent.

Single-process, single-event-loop. The leader Agent holds at most one
``TeamManager``; the manager owns the on-disk ``TeamRecord``, the mailbox
handles, and the asyncio.Task handles for each teammate runtime.

Out of scope here: CLI/slash dispatch (``aura.application.commands.team``),
the runtime loop (``aura.application.teams.runtime``), and permission policy
(handed to ``SubagentFactory.spawn``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aura.application.tasks.factory import SubagentFactory
from aura.application.tasks.store import TasksStore
from aura.application.teams.mailbox import Mailbox, QueueMailboxNotifier
from aura.domain.abort import AbortController
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
from aura.infrastructure.wire.event_dto import WireEvent

if TYPE_CHECKING:
    from aura.core.agent import Agent
    from aura.infrastructure.teams.types import BackendHandle

# Slug pattern for team_id / member name (ASCII alnum + ``-`` / ``_``);
# filesystem-safe on every platform we support.
_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class TeammateMemberStatus:
    """Per-member projection used by :meth:`TeamManager.view_state`.

    Status ∈ ``"active"`` / ``"shutting-down"`` / ``"dead"``. ``model_spec``
    is ``None`` when the teammate inherits the leader's default;
    ``last_active`` is ``None`` until the first tracked event.
    """

    name: str
    agent_type: str
    model_spec: str | None
    status: str
    tokens_used: int
    last_active: float | None


@dataclass(frozen=True)
class TeamViewSnapshot:
    """Aggregated read-only projection of a team's state for ``/team view``.

    Built by :meth:`TeamManager.view_state`. Members come from the
    in-memory :class:`TeamRecord`; recent messages come from disk
    (the union of every per-recipient JSONL inbox), sorted by
    ``sent_at`` descending and capped at ``RECENT_MESSAGE_CAP``.
    Subagent + transcript counts are best-effort directory walks —
    the renderer treats them as informational, not load-bearing.
    """

    team_id: str
    name: str
    members: list[TeammateMemberStatus]
    recent_messages: list[TeamMessage]
    subagent_count: int
    transcript_count: int


_RECENT_MESSAGE_CAP: int = 10


class TeamError(ValueError):
    """Domain error for invariant violations. Surfaces to the LLM as a
    ToolError and to the CLI as a printable string."""


def _slugify(raw: str) -> str:
    """Reduce ``raw`` to a filesystem-safe slug; raise on empty result."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", raw.strip()).strip("-_")
    if not cleaned:
        raise TeamError(f"name {raw!r} has no slugifiable characters")
    return cleaned


class TeamManager:
    """Lifecycle owner for one leader Agent's team.

    One team per leader. ``add_member`` spawns the teammate runtime via the
    injected ``runtime_runner`` so tests can substitute a fake; production
    wires :func:`aura.application.teams.runtime.run_teammate`.
    """

    def __init__(
        self,
        *,
        leader: Agent,
        storage: SessionStorage,
        factory: SubagentFactory,
        running_aborts: dict[str, AbortController],
        tasks_store: TasksStore,
        runtime_runner: Any = None,
    ) -> None:
        self._leader = leader
        self._storage = storage
        self._factory = factory
        self._running_aborts = running_aborts
        self._tasks_store = tasks_store
        # Lazy import — runtime imports manager types, so the inverse import
        # would close the cycle.
        if runtime_runner is None:
            from aura.application.teams.runtime import run_teammate
            self._runtime_runner = run_teammate
        else:
            self._runtime_runner = runtime_runner
        self._team: TeamRecord | None = None
        # task_id -> runtime asyncio.Task. Pruned by ``_finalize_runtime_task``.
        self._runtimes: dict[str, asyncio.Task[None]] = {}
        self._member_task_ids: dict[str, str] = {}
        self._member_agents: dict[str, Agent] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        # Per-member shutdown ack future; resolved by ``confirm_shutdown``
        # when the in-process runtime consumes a ``shutdown_request``.
        # Pane subprocesses cannot reach this future across the process
        # boundary; ``aremove_member`` falls back to inbox-poll there.
        self._shutdown_acks: dict[str, asyncio.Future[bool]] = {}
        # In-flight ``aremove_member`` waiter tasks; tests can await these.
        self._shutdown_waiters: dict[str, asyncio.Task[bool]] = {}
        # In-process mailbox wake-up channel. ``send`` / ``_post`` signal
        # the recipient's per-member event so the runtime exits its
        # ``wait_new`` instantly. Pane recipients ignore signals (they
        # poll the JSONL from a separate process).
        self._mailbox_notifier = QueueMailboxNotifier()
        # Backend-agnostic shutdown handles. The in-process backend's handle
        # wraps the same task we track in ``_runtimes`` — duplication is
        # intentional so ``cleanup_session_teams`` walks ``_member_backends``
        # uniformly without dispatching on backend_type.
        self._member_backends: dict[str, BackendHandle] = {}
        # Explicit terminal state selected by manager-owned lifecycle
        # transitions; consulted in the runtime done callback.
        self._teammate_terminal_intents: dict[str, str] = {}
        # team_ids created in this process — ``cleanup_session_teams`` rms
        # any entries left behind on Agent.aclose.
        self._session_created_teams: set[str] = set()
        # Live coordination queue for protocol adapters; orthogonal to mailbox.
        self._pending_protocol_events: list[WireEvent] = []

    # ------------------------------------------------------------------
    # Lifecycle: create / delete / lookup
    # ------------------------------------------------------------------

    @property
    def team(self) -> TeamRecord | None:
        return self._team

    @property
    def is_active(self) -> bool:
        return self._team is not None

    @property
    def pending_protocol_events(self) -> tuple[WireEvent, ...]:
        """Snapshot of queued team coordination wire events."""
        return tuple(self._pending_protocol_events)

    def mailbox(self) -> Mailbox:
        """Return a Mailbox bound to the live team. Raises if no team."""
        if self._team is None:
            raise TeamError("no team is active")
        return Mailbox(self._storage, self._team.team_id)

    def create_team(self, name: str) -> TeamRecord:
        """Create the (one) team this leader owns.

        Persists ``config.json`` immediately so a crash before the first
        ``add_member`` still leaves a recoverable state on disk.
        """
        if self._team is not None:
            raise TeamError(
                f"team {self._team.team_id!r} is already active "
                "(Phase A: one team per leader)",
            )
        team_id = _slugify(name)
        # Collision check: a previous session may have left a team folder
        # behind. Append ``-2`` / ``-3`` until free, mirroring claude-code.
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
        # Track for session-end cleanup — claude-code parity (gh-32730).
        # Removed by :meth:`delete_team` when the user explicitly tears
        # down; otherwise consumed by :meth:`cleanup_session_teams`.
        self._session_created_teams.add(team_id)
        journal.write(
            "team_created",
            team_id=team_id,
            name=name,
            leader_session=self._leader.session_id,
        )
        return record

    def delete_team(self) -> None:
        """Tear down every teammate AND remove the team directory from disk.

        Aligned with claude-code's ``TeamDeleteTool`` (post gh-32730):
        explicit delete is destructive — config.json + inbox/ + transcripts/
        are all removed. ``_session_created_teams`` membership is cleared
        so the session-end cleanup doesn't double-rm.
        """
        if self._team is None:
            return
        # Snapshot members BEFORE cancellation so iteration is stable
        # while we mutate ``_runtimes`` / ``_member_task_ids``.
        for member in list(self._team.members):
            with contextlib.suppress(TeamError):
                self.remove_member(member.name, force=True)
        team_id = self._team.team_id
        self._session_created_teams.discard(team_id)
        team_dir = self._storage.team_root(team_id)
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(team_dir)
        self._team = None
        journal.write("team_deleted", team_id=team_id, dir_removed=True)

    async def cleanup_session_teams(self) -> None:
        """Remove every team this session created that wasn't explicitly deleted.

        Mirrors claude-code's ``cleanupSessionTeams`` (``utils/swarm/
        teamHelpers.ts:576``) — invoked from ``Agent.aclose`` so an
        operator who forgot to ``/team delete`` doesn't leak orphan
        directories. Best-effort: missing directories are tolerated,
        rmtree errors are journaled but do not propagate.
        """
        if not self._session_created_teams:
            return
        # Snapshot before mutation; the rmtree loop clears the set entry
        # by entry so a partial failure doesn't strand the surviving
        # entries.
        team_ids = list(self._session_created_teams)
        # Best-effort: cancel any still-running runtime tasks first so
        # rmtree doesn't race with an active poll loop.
        for task_id in list(self._member_task_ids.values()):
            self._mark_teammate_cancelled(task_id)
        for task in list(self._runtimes.values()):
            if not task.done():
                task.cancel()
        if self._runtimes:
            await asyncio.gather(
                *[t for t in self._runtimes.values() if not t.done()],
                return_exceptions=True,
            )
        # Pane backends also need a force_kill so their tmux panes
        # close before we ``rm -rf`` the team directory. ``force_kill``
        # is idempotent — handles already cleaned by ``_teardown_member``
        # become no-ops.
        if self._member_backends:
            await asyncio.gather(
                *[h.force_kill() for h in self._member_backends.values()],
                return_exceptions=True,
            )
            self._member_backends.clear()
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
        # Clear in-memory state if the live team was among the cleaned set.
        if self._team is not None and self._team.team_id not in self._session_created_teams:
            # Already pruned from the set above; clear runtime state too.
            self._team = None
            self._runtimes.clear()
            self._member_task_ids.clear()
            self._member_agents.clear()
            self._stop_events.clear()
            self._teammate_terminal_intents.clear()

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

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
        """Spawn a teammate, register its runtime task, persist the record.

        ``seed_prompt`` is the FIRST message the teammate consumes — it
        skips the mailbox and is fed directly into the first ``astream``
        iteration so a freshly-added teammate doesn't need a separate
        ``send_message`` to get going. Optional; ``None`` means "wait
        idle until the leader sends something".

        ``backend_type`` selects the runtime strategy:
        ``"in_process"`` (default) runs the teammate as an asyncio task
        on the leader's loop; ``"pane"`` spawns a real subprocess inside
        a freshly-split tmux pane. The pane backend raises
        :class:`TeamError` early when the environment doesn't support
        it (no ``$TMUX`` / no ``tmux`` on PATH).

        Returns the persisted :class:`TeammateMember`. Raises
        :class:`TeamError` on duplicate name, ``MAX_MEMBERS`` overflow,
        invalid slug, or no active team.
        """
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
        # Resolve the backend BEFORE we mutate state so a misrouted
        # ``backend_type="pane"`` outside tmux fails fast without
        # leaving an orphan TaskRecord / member row.
        from aura.infrastructure.teams.registry import (
            BackendUnavailable,
            get_backend,
        )
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
        # Register a TaskRecord for the teammate so /tasks + journal +
        # observability tooling all see it. ``kind="teammate"`` keeps it
        # distinct from one-shot subagents in /tasks output.
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
        # Build the child Agent up-front so we can plumb the per-team
        # context (team manager, session_id) onto it before the runtime
        # picks it up. The factory installs the permission hook with the
        # SAME RuleSet / SafetyPolicy / mode-provider that the leader
        # uses — bit-for-bit inheritance.
        child = self._factory.spawn(
            prompt_for_task,
            agent_type=agent_type,
            task_id=record.id,
            model_spec=model_name,
        )
        # Stamp the teammate identity onto the Agent so the SendMessage
        # tool can resolve the (team_id, sender) pair without reaching
        # back through the manager's private state.
        child.join_team(manager=self, member_name=name)
        object.__setattr__(child, "_teammate_task_id", record.id)
        object.__setattr__(child, "_teammate_tasks_store", self._tasks_store)
        self._member_agents[name] = child
        self._member_task_ids[name] = record.id
        # Allocate the abort controller and register it with the
        # leader's running_aborts BEFORE the runtime starts so a parent
        # cascade arriving in the same scheduler tick still finds it.
        abort = AbortController()
        self._running_aborts[record.id] = abort
        # Per-runtime stop event — set by remove_member so the loop can
        # exit between mailbox polls without waiting for the next abort.
        stop_event = asyncio.Event()
        # Spawn path:
        #
        # 1. If a custom ``runtime_runner`` was injected (tests), use the
        #    legacy direct ``asyncio.create_task`` flow so the test's
        #    runner shape (``async def(**kwargs) -> None``) keeps working
        #    bit-for-bit. The handle wraps the resulting task with the
        #    same in-process semantics the registry would have produced.
        # 2. Otherwise dispatch to the in-process backend's ``spawn_sync``
        #    helper — a synchronous wrapper around the same
        #    ``asyncio.create_task`` call so we don't have to drive an
        #    async coroutine from this sync method. Pane backend MUST
        #    use :meth:`aadd_member` from an async context.
        from aura.application.teams.runtime import run_teammate as _default_runner
        from aura.infrastructure.teams.in_process import (
            InProcessBackend as _InProcessBackend,
        )
        from aura.infrastructure.teams.in_process import (
            InProcessHandle as _InProcessHandle,
        )
        if backend_type == "pane":
            raise TeamError(
                "pane backend must be added via aadd_member() from an "
                "async context (sync add_member supports in_process only)",
            )
        handle: BackendHandle
        if self._runtime_runner is not _default_runner:
            # Legacy path for tests: run the injected coroutine directly.
            def _cleanup(_t: asyncio.Task[None]) -> None:
                self._finalize_runtime_task(record.id, _t, abort)
            task: asyncio.Task[None] = asyncio.create_task(
                self._runtime_runner(
                    agent=child,
                    team_id=self._team.team_id,
                    member_name=name,
                    storage=self._storage,
                    stop_event=stop_event,
                    abort=abort,
                    seed_prompt=seed_prompt,
                ),
                name=f"aura-teammate-{name}",
            )
            task.add_done_callback(_cleanup)
            self._runtimes[record.id] = task
            handle = _InProcessHandle(
                task=task,
                stop_event=stop_event,
                abort=abort,
            )
        else:
            # Production path — dispatch via the in-process backend
            # singleton (sync helper). Pane already short-circuited above.
            assert isinstance(backend, _InProcessBackend)
            handle = backend.spawn_sync(
                team_id=self._team.team_id,
                member=member,
                agent=child,
                manager=self,
                storage=self._storage,
                stop_event=stop_event,
                abort=abort,
                seed_prompt=seed_prompt,
                notifier=self._mailbox_notifier,
            )
            in_proc_task = handle.task
            def _cleanup(_t: asyncio.Task[None]) -> None:
                self._finalize_runtime_task(record.id, _t, abort)
            in_proc_task.add_done_callback(_cleanup)
            self._runtimes[record.id] = in_proc_task
        # Stash the stop_event on the manager so remove_member can fire
        # it without re-allocating; a member->event map avoids leaking
        # the event into the runtime's call signature.
        self._stop_events[name] = stop_event
        # Backend handle is the canonical shutdown surface; index by name.
        self._member_backends[name] = handle
        # If the backend mutated ``member.tmux_pane_id`` (pane only),
        # persist the updated record so a crash leaves recoverable state.
        if member.tmux_pane_id is not None:
            self._persist()
        journal.write(
            "team_member_added",
            team_id=self._team.team_id,
            member=name,
            agent_type=agent_type,
            backend_type=backend_type,
            task_id=record.id,
            tmux_pane_id=member.tmux_pane_id,
        )
        return member

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
        """Async-native variant of :meth:`add_member`.

        Required by the pane backend whose ``spawn`` awaits
        ``subprocess.run`` via ``asyncio.to_thread``; the sync
        ``add_member`` cannot drive that from inside a running event
        loop. The in-process backend works identically through either
        entry point.

        Implementation defers to :meth:`add_member` for non-pane
        backends; for pane it performs the same sequence but awaits the
        backend spawn directly without any loop juggling.
        """
        if backend_type != "pane":
            return self.add_member(
                name,
                agent_type=agent_type,
                system_prompt=system_prompt,
                model_name=model_name,
                seed_prompt=seed_prompt,
                backend_type=backend_type,
            )
        # Pane path — async-native.
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
        from aura.infrastructure.teams.registry import (
            BackendUnavailable,
            get_backend,
        )
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
        child.join_team(manager=self, member_name=name)
        object.__setattr__(child, "_teammate_task_id", record.id)
        object.__setattr__(child, "_teammate_tasks_store", self._tasks_store)
        self._member_agents[name] = child
        self._member_task_ids[name] = record.id
        abort = AbortController()
        self._running_aborts[record.id] = abort
        stop_event = asyncio.Event()
        handle = await backend.spawn(
            team_id=self._team.team_id,
            member=member,
            agent=child,
            manager=self,
            storage=self._storage,
            stop_event=stop_event,
            abort=abort,
            seed_prompt=seed_prompt,
            notifier=self._mailbox_notifier,
        )
        task = getattr(handle, "task", None)
        if isinstance(task, asyncio.Task):
            def _cleanup(_t: asyncio.Task[None]) -> None:
                self._finalize_runtime_task(record.id, _t, abort)
            task.add_done_callback(_cleanup)
            self._runtimes[record.id] = task
        self._stop_events[name] = stop_event
        self._member_backends[name] = handle
        if member.tmux_pane_id is not None:
            self._persist()
        journal.write(
            "team_member_added",
            team_id=self._team.team_id,
            member=name,
            agent_type=agent_type,
            backend_type=backend_type,
            task_id=record.id,
            tmux_pane_id=member.tmux_pane_id,
        )
        return member

    #: Default per-member graceful-shutdown grace window. The teammate
    #: runtime polls its mailbox in 5-second slices, so 5s is the floor
    #: that avoids fighting that cadence; tests override via the
    #: ``timeout_sec`` kwarg on :meth:`aremove_member`.
    DEFAULT_SHUTDOWN_GRACE_SEC: float = 5.0

    def remove_member(
        self,
        name: str,
        *,
        force: bool = False,
        timeout_sec: float | None = None,
    ) -> None:
        """Sync entry point — graceful by default, force-kill if requested.

        Phase A.1 split:

        - ``force=True`` (or no running event loop available) →
          synchronous force-kill: append a ``shutdown_request`` for
          observability, fire ``stop_event``, abort + cancel the
          runtime task. This is what :meth:`delete_team` uses.
        - ``force=False`` and a running event loop is available →
          schedule :meth:`aremove_member` as a fire-and-forget task
          on that loop and return immediately. The CLI's
          ``/team remove`` keeps its sync feel while the round-trip
          + force-kill-on-timeout cleanup runs in the background.
          Tests and async callers that want a handle should call
          :meth:`aremove_member` directly.

        ``timeout_sec`` is forwarded to :meth:`aremove_member` for
        the graceful path; ignored when ``force=True``.
        """
        if self._team is None:
            raise TeamError("no team is active")
        if not any(m.name == name for m in self._team.members):
            raise TeamError(f"member {name!r} not found in team")
        if force:
            self._teardown_member(name, send_request=False, journal_force=True)
            return
        # Graceful path: try to schedule the async waiter on the
        # current loop. Fall back to a force-style teardown when no
        # loop is running (e.g. unit tests that drive the manager
        # without ever entering an asyncio context).
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            # No event loop — degrade to the legacy sync behaviour
            # (request + stop + abort + cancel) and journal that we
            # could not perform the round-trip wait. The teammate
            # task, if any, is already done in this scenario (tests
            # use ``_no_runtime``); production always has a loop.
            self._teardown_member(
                name, send_request=True, journal_force=True,
            )
            return
        # Fire-and-forget: a waiter task is scheduled; the caller
        # does not block. Stash the task on the manager so tests
        # / future SDK callers can observe completion if needed.
        waiter = loop.create_task(
            self.aremove_member(name, timeout_sec=timeout_sec),
            name=f"aura-team-shutdown-{name}",
        )
        self._shutdown_waiters[name] = waiter
        # Drop reference on completion so the dict doesn't leak.
        def _prune(_t: asyncio.Task[bool]) -> None:
            self._shutdown_waiters.pop(name, None)
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
        """Graceful shutdown with a per-member ack future.

        Returns ``True`` when the teammate confirmed shutdown within
        ``timeout_sec``; ``False`` when the wait timed out and the
        member was force-killed instead. ``force=True`` short-circuits
        the wait and is equivalent to :meth:`remove_member`.

        Sequence:

        1. Drop the membership row so concurrent sends raise.
        2. Allocate a per-member ``asyncio.Future`` ack channel.
        3. Append a ``shutdown_request`` to the teammate's mailbox; the
           ``QueueMailboxNotifier`` wakes the in-process runtime
           instantly.
        4. Fire the per-member ``stop_event`` (covers the no-message
           idle path).
        5. ``await`` the ack future with ``timeout_sec``. The runtime
           resolves it through :meth:`confirm_shutdown` (in-process) or,
           for pane subprocesses, by writing a ``shutdown_response`` to
           the leader inbox — pane handles still observe via their own
           inbox-poll inside the backend handle.
        6. On ack: journal + cooperative teardown.
        7. On timeout: journal + force-kill (abort + cancel).

        Idempotent: a second call with the same ``name`` raises
        ``TeamError`` (the membership row is already gone), so the
        caller can pattern-match if it wants "best-effort cleanup"
        semantics.
        """
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
        # Drop the row + allocate ack future + send request + fire stop.
        # The runtime needs to live long enough to resolve the future,
        # so abort + cancel are deferred to the timeout / teardown path.
        idx = next(
            i for i, m in enumerate(self._team.members) if m.name == name
        )
        self._team.members.pop(idx)
        self._persist()
        loop = asyncio.get_running_loop()
        ack_future: asyncio.Future[bool] = loop.create_future()
        self._shutdown_acks[name] = ack_future
        with contextlib.suppress(Exception):
            self._post(TeamMessage(
                msg_id=uuid.uuid4().hex,
                sender=TEAM_LEADER_NAME,
                recipient=name,
                body="shutdown",
                kind="shutdown_request",
            ))
        stop_event = self._stop_events.get(name)
        if stop_event is not None:
            task_id = self._member_task_ids.get(name)
            if task_id is not None:
                self._set_teammate_cancel_intent(task_id)
            stop_event.set()
        try:
            acked = await asyncio.wait_for(
                asyncio.shield(ack_future), timeout=timeout,
            )
        except TimeoutError:
            acked = False
        finally:
            self._shutdown_acks.pop(name, None)
        if acked:
            journal.write(
                "team_member_shutdown_ack_received",
                team_id=team_id,
                member=name,
            )
            self._teardown_member(
                name,
                send_request=False,
                journal_force=False,
                already_acked=True,
            )
            return True
        # Timeout — fall through to force-kill.
        journal.write(
            "team_member_shutdown_force_killed",
            team_id=team_id,
            member=name,
            timeout_sec=timeout,
        )
        self._teardown_member(name, send_request=False, journal_force=True)
        return False

    def confirm_shutdown(self, member_name: str, *, body: str = "") -> None:
        """Resolve the per-member ack future if ``aremove_member`` is waiting.

        Called by the in-process runtime when it consumes a
        ``shutdown_request``. ``body`` is accepted for symmetry with
        the legacy ``shutdown_response`` envelope but currently unused
        — the future carries a bool, and the journal already records
        the request body. Idempotent: no future ⇒ no-op (pane path or
        runtime exiting via abort).
        """
        del body
        fut = self._shutdown_acks.get(member_name)
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
        """Drop bookkeeping for ``name`` and (optionally) force-kill the runtime.

        Shared between :meth:`remove_member` and :meth:`aremove_member`.
        ``send_request=True`` posts a ``shutdown_request`` first (the
        synchronous force-kill path uses this for observability;
        graceful path already sent its own request before waiting).
        ``already_acked=True`` skips the abort+cancel (the runtime has
        exited cooperatively); otherwise we abort + cancel + aclose.
        """
        if self._team is None:
            return
        # Membership row may already be popped by aremove_member.
        idx = next(
            (i for i, m in enumerate(self._team.members) if m.name == name),
            -1,
        )
        if idx >= 0:
            self._team.members.pop(idx)
            self._persist()
        task_id = self._member_task_ids.pop(name, None)
        stop_event = self._stop_events.pop(name, None)
        # Drop any pending ack future — the member is gone, no point
        # leaving an awaiter wedged. ``aremove_member`` owns its own
        # cleanup so we only drop the orphan entry from a synchronous
        # ``remove_member(force=True)`` path.
        pending_ack = self._shutdown_acks.pop(name, None)
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
        # Backend handle teardown — covers the pane case (kill-pane) and
        # is a harmless no-op for in-process where the task cancel above
        # already did the work.
        backend_handle = self._member_backends.pop(name, None)
        if backend_handle is not None and not already_acked:
            with contextlib.suppress(Exception):
                # ``force_kill`` is async (pane awaits a tmux IPC); fire-
                # and-forget on the running loop. If no loop is active
                # (rare; sync-only test path), skip — there's nothing
                # the backend can do without a loop, and the in-process
                # task cancel above already covered that case.
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None:
                    loop.create_task(backend_handle.force_kill())
        agent = self._member_agents.pop(name, None)
        if agent is not None:
            with contextlib.suppress(Exception):
                asyncio.ensure_future(agent.aclose())
        if journal_force:
            journal.write(
                "team_member_removed",
                team_id=self._team.team_id,
                member=name,
                forced=not already_acked,
            )

    def _set_teammate_cancel_intent(self, task_id: str) -> None:
        """Remember that a running teammate should finish as cancelled."""
        if task_id in self._runtimes:
            self._teammate_terminal_intents[task_id] = "cancelled"

    def _mark_teammate_cancelled(self, task_id: str) -> None:
        """Mark a teammate TaskRecord cancelled and preserve callback intent."""
        self._set_teammate_cancel_intent(task_id)
        self._tasks_store.mark_cancelled(task_id)

    def _finalize_runtime_task(
        self,
        task_id: str,
        task: asyncio.Task[None],
        abort: AbortController,
    ) -> None:
        """Mirror a teammate runtime task's terminal outcome to TaskRecord."""
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

    # ------------------------------------------------------------------
    # Read-only aggregator (powers ``/team view`` UX)
    # ------------------------------------------------------------------

    def view_state(self, team_id: str | None = None) -> TeamViewSnapshot:
        """Aggregate a read-only snapshot of a team for the ``/team view`` UX.

        Pure read path — does NOT mutate the manager, the record, or
        any mailbox. Safe to call from a slash command handler without
        worrying about racing the runtime loop.

        ``team_id=None`` snapshots the live team (the one this manager
        owns). An explicit ``team_id`` snapshots a different team's
        on-disk state (members + inbox JSONLs); used by ``/team view
        <name>`` when the caller hasn't entered the team yet. Raises
        :class:`TeamError` when neither path resolves to a real team.

        Members come from the in-memory :class:`TeamRecord` for the
        live team, or from ``config.json`` for an off-record team.
        Recent messages are the union of every recipient's inbox JSONL
        (``leader.jsonl`` + every member's), sorted ``sent_at`` desc
        and capped at :data:`_RECENT_MESSAGE_CAP`.
        """
        if team_id is None:
            if self._team is None:
                raise TeamError("no team is active; pass team_id explicitly")
            record = self._team
        elif self._team is not None and self._team.team_id == team_id:
            record = self._team
        else:
            # Off-record snapshot — load config.json fresh. We don't
            # cache the loaded record on the manager; a second view
            # call should re-read so a concurrent writer's update is
            # picked up next time.
            path = self._storage.team_config_path(team_id)
            if not path.exists():
                raise TeamError(f"team {team_id!r} not found on disk")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise TeamError(
                    f"team {team_id!r} config is unreadable: {exc}",
                ) from exc
            record = TeamRecord.model_validate(raw)
        members = self._build_member_statuses(record)
        recent = self._collect_recent_messages(record)
        sub_count, tx_count = self._count_artifacts(record)
        return TeamViewSnapshot(
            team_id=record.team_id,
            name=record.name,
            members=members,
            recent_messages=recent,
            subagent_count=sub_count,
            transcript_count=tx_count,
        )

    def _build_member_statuses(
        self, record: TeamRecord,
    ) -> list[TeammateMemberStatus]:
        """Project ``record.members`` into status rows for ``view_state``.

        Live team: cross-reference :class:`TasksStore` for tokens +
        ``last_activity_at`` so the row reflects what the teammate has
        actually consumed since spawn. Off-record team: tokens / last
        active stay zero / ``None`` because the runtime task is gone.
        Status is ``"active"`` while the membership row is present;
        ``"dead"`` when ``is_active=False``. ``"shutting-down"`` is
        reserved for the in-flight ``aremove_member`` window — we
        approximate that by checking the per-member shutdown waiter.
        """
        out: list[TeammateMemberStatus] = []
        live = self._team is not None and self._team.team_id == record.team_id
        for m in record.members:
            tokens = 0
            last_active: float | None = None
            model_spec: str | None = m.model_name
            if live:
                task_id = self._member_task_ids.get(m.name)
                if task_id is not None:
                    rec = self._tasks_store.get(task_id)
                    if rec is not None:
                        tokens = int(rec.progress.token_count)
                        last_active = rec.progress.last_activity_at
                        # Prefer the actual-resolved model_spec the
                        # task was spawned on so the operator sees the
                        # spec the teammate is REALLY running, not the
                        # override token (which is empty for inherits).
                        resolved = getattr(rec, "model_spec", None)
                        if resolved:
                            model_spec = resolved
            if not m.is_active:
                status = "dead"
            elif live and m.name in self._shutdown_waiters:
                status = "shutting-down"
            else:
                status = "active"
            out.append(
                TeammateMemberStatus(
                    name=m.name,
                    agent_type=m.agent_type,
                    model_spec=model_spec,
                    status=status,
                    tokens_used=tokens,
                    last_active=last_active,
                ),
            )
        return out

    def _collect_recent_messages(
        self, record: TeamRecord,
    ) -> list[TeamMessage]:
        """Return last :data:`_RECENT_MESSAGE_CAP` messages across inboxes.

        Reads every per-recipient JSONL (``leader`` + every active
        member) and merges into one chronological list. Bounded by the
        cap before return so callers don't have to slice.
        """
        mailbox = Mailbox(self._storage, record.team_id)
        recipients = [TEAM_LEADER_NAME] + [m.name for m in record.members]
        gathered: list[TeamMessage] = []
        for rcpt in recipients:
            gathered.extend(mailbox.read_all(rcpt))
        gathered.sort(key=lambda m: m.sent_at, reverse=True)
        return gathered[:_RECENT_MESSAGE_CAP]

    def _count_artifacts(self, record: TeamRecord) -> tuple[int, int]:
        """Return ``(subagent_count, transcript_count)`` for ``record``.

        Subagent count is the count of distinct transcripts under the
        leader's storage root (the parent owns the subagents). Per-team
        transcript count walks the team's own ``transcripts/`` dir so
        teammate transcripts (one per member) get reported separately.
        Both are best-effort: missing directories return 0.
        """
        sub_count = 0
        with contextlib.suppress(Exception):
            sub_count = len(self._storage.list_subagent_transcripts())
        tx_count = 0
        try:
            tx_dir = self._storage.team_root(record.team_id) / "transcripts"
            if tx_dir.is_dir():
                tx_count = sum(
                    1 for p in tx_dir.iterdir()
                    if p.is_file() and p.suffix == ".jsonl"
                )
        except OSError:
            tx_count = 0
        return sub_count, tx_count

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        body: str,
        kind: TeamMessageKind = "text",
    ) -> list[TeamMessage]:
        """Append one (or N for broadcast) JSONL message lines.

        Returns the actual TeamMessage objects written so callers can
        report back ``msg_id`` / ``sent_at``. Empty list never returned —
        a recipient resolving to zero members raises ``TeamError`` so the
        caller sees the failure cleanly.
        """
        if self._team is None:
            raise TeamError("no team is active")
        if not body.strip():
            raise TeamError("body must contain at least one non-whitespace char")
        if len(body) > MAX_BODY_CHARS:
            raise TeamError(
                f"body length {len(body)} exceeds MAX_BODY_CHARS={MAX_BODY_CHARS}",
            )
        # Secret-scrub the body BEFORE it lands in any mailbox / wire
        # event. Cross-member text is the highest-leak surface — once a
        # secret hits the JSONL inbox every teammate on disk has it.
        # ``redact_secrets`` is conservative (false positives over false
        # negatives) which matches the threat model exactly.
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
            from aura.infrastructure.wire.wire import team_message_to_wire
            self._pending_protocol_events.append(
                team_message_to_wire(msg, team_id=self._team.team_id),
            )
        return sent

    def _post(self, msg: TeamMessage) -> None:
        """Internal append (skips fan-out + length checks; for control msgs).

        Control messages (shutdown_request / shutdown_response) get
        the same redaction treatment as text — the body could carry a
        teammate-supplied justification that an LLM accidentally
        pasted an API key into.
        """
        msg = msg.model_copy(update={"body": redact_secrets(msg.body)})
        self.mailbox().append(msg)
        self._mailbox_notifier.signal(msg.recipient)
        if self._team is not None:
            from aura.infrastructure.wire.wire import team_message_to_wire
            self._pending_protocol_events.append(
                team_message_to_wire(msg, team_id=self._team.team_id),
            )

    def drain_protocol_events(self) -> list[WireEvent]:
        """Pop every queued team coordination wire event, oldest first."""
        drained = list(self._pending_protocol_events)
        self._pending_protocol_events.clear()
        return drained

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

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
            # Don't crash the agent on a transient FS failure — surface
            # via journal so the operator sees the disk problem.
            journal.write(
                "team_persist_failed",
                team_id=self._team.team_id,
                error=f"{type(exc).__name__}: {exc}",
            )

__all__ = [
    "TeamError",
    "TeamManager",
    "TeammateMemberStatus",
    "TeamViewSnapshot",
]
