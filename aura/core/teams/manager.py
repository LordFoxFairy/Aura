"""TeamManager — owns the lifecycle of one team per leader Agent.

Single-process, single-event-loop. The leader Agent holds at most one
``TeamManager`` (Phase A invariant); the manager owns the disk-rooted
``TeamRecord``, the in-memory mailbox handles, and the asyncio.Task
handles for each running teammate's runtime.

Responsibilities:

- Persist ``config.json`` atomically on every membership change.
- Build child Agent instances via the parent's ``SubagentFactory`` so
  permission inheritance lands bit-for-bit (mirrors
  ``factory.py:241-269``).
- Register each teammate's ``AbortController`` in the parent's
  ``_running_aborts`` map so a parent abort cascades.
- Route ``send`` calls to the recipient's mailbox (broadcast fans out to
  every active member in declaration order).

What we deliberately do NOT do here:

- No CLI, no slash dispatch — that's :mod:`aura.core.commands.team`.
- No long-lived loop logic — that's :mod:`aura.core.teams.runtime`.
- No safety / permission policy choices — we hand the parent's
  policy snapshot to ``SubagentFactory.spawn`` and trust the existing
  pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aura.core.abort import AbortController
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.store import TasksStore
from aura.core.teams.mailbox import Mailbox
from aura.core.teams.types import (
    BROADCAST_RECIPIENT,
    MAX_BODY_CHARS,
    MAX_MEMBERS,
    TEAM_LEADER_NAME,
    TeammateMember,
    TeamMessage,
    TeamMessageKind,
    TeamRecord,
)

if TYPE_CHECKING:
    from aura.core.agent import Agent

#: Slug pattern for team_id and member name. Restricted to ASCII alnum +
#: ``-`` / ``_`` so the value is safe as a filesystem path segment on
#: every platform we support.
_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class TeammateMemberStatus:
    """Per-member projection used by :meth:`TeamManager.view_state`.

    Snapshot fields chosen to match claude-code's
    ``CoordinatorAgentStatus`` row shape: identity (``name``,
    ``agent_type``), execution profile (``model_spec``), liveness
    (``status`` ∈ ``"active"``/``"shutting-down"``/``"dead"``), and a
    pair of activity counters (``tokens_used``, ``last_active``) for
    the operator-facing "is this teammate doing anything?" summary.

    ``model_spec`` is the resolved spec the teammate's child Agent
    runs on (TaskRecord.model_spec); ``None`` when no override was
    set (the teammate inherits the leader's default). ``last_active``
    is the wall-clock float from the last journal/progress tick;
    ``None`` when the teammate has never produced a tracked event.
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


#: Cap on ``recent_messages`` returned by :meth:`TeamManager.view_state`.
#: Mirrors claude-code's ``TaskListV2`` per-task last-N tail length
#: (10 lines per pane is the readability sweet spot).
_RECENT_MESSAGE_CAP: int = 10


class TeamError(ValueError):
    """Domain error raised by :class:`TeamManager` for invariant violations.

    Surfaces to the LLM as a ToolError (via :class:`SendMessage`) and to
    the CLI as a printable string. Distinct from ``RuntimeError`` so
    callers can pattern-match on team-specific failures without
    swallowing structural bugs.
    """


def _slugify(raw: str) -> str:
    """Reduce ``raw`` to an ASCII filesystem-safe slug.

    Whitespace and other separators collapse to ``-``; non-matching
    chars drop. Empty result raises so the caller can show a clear
    "name must contain alnum chars" error rather than persist an empty
    folder name.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", raw.strip())
    cleaned = cleaned.strip("-_") or ""
    if not cleaned:
        raise TeamError(f"name {raw!r} has no slugifiable characters")
    return cleaned


class TeamManager:
    """Lifecycle owner for one leader Agent's team.

    Phase A: one team per leader. ``create_team`` rejects a second team
    while one is live (matches claude-code's "leader manages one team"
    rule). ``add_member`` spawns the teammate runtime via the injected
    ``runtime_runner`` callable so tests can substitute a fake; production
    wires :func:`aura.core.teams.runtime.run_teammate`.
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
        # Lazy import to avoid the import cycle (runtime imports manager
        # types). Tests may inject their own runner without ever touching
        # the runtime module.
        if runtime_runner is None:
            from aura.core.teams.runtime import run_teammate
            self._runtime_runner = run_teammate
        else:
            self._runtime_runner = runtime_runner
        # The single live team — Phase A invariant. ``None`` when no team
        # exists; ``create_team`` populates this and ``delete_team``
        # clears it.
        self._team: TeamRecord | None = None
        # task_id -> the asyncio.Task running that member's runtime loop.
        # Populated by ``add_member`` (so the leader's aclose can cancel
        # them) and pruned on ``remove_member`` / on natural exit via the
        # done-callback installed at spawn.
        self._runtimes: dict[str, asyncio.Task[None]] = {}
        # member_name -> task_id mapping; used to look up which abort
        # controller to fire on ``remove_member`` and to look up the
        # child Agent's session_id for transcript flushes.
        self._member_task_ids: dict[str, str] = {}
        # member_name -> Agent instance, kept so ``remove_member`` can
        # drive a clean ``aclose`` and so /team commands can introspect.
        self._member_agents: dict[str, Agent] = {}
        # member_name -> per-runtime stop event. Kept here (not closed
        # over by run_teammate) so ``remove_member`` can fire it
        # synchronously without juggling weakrefs.
        self._stop_events: dict[str, asyncio.Event] = {}
        # member_name -> in-flight ``aremove_member`` waiter task.
        # Populated when ``remove_member`` (sync) schedules a graceful
        # shutdown on the running loop; cleared by the task's done
        # callback. Tests can ``await mgr._shutdown_waiters[name]`` to
        # block on the round-trip from a sync entry point.
        self._shutdown_waiters: dict[str, asyncio.Task[bool]] = {}

    # ------------------------------------------------------------------
    # Lifecycle: create / delete / lookup
    # ------------------------------------------------------------------

    @property
    def team(self) -> TeamRecord | None:
        return self._team

    @property
    def is_active(self) -> bool:
        return self._team is not None

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
        journal.write(
            "team_created",
            team_id=team_id,
            name=name,
            leader_session=self._leader.session_id,
        )
        return record

    def delete_team(self) -> None:
        """Tear down every teammate, clear in-memory state, leave files alone.

        Files stay so post-mortem inspection works; the design proposal
        §3.3 calls this out explicitly. A future ``--purge`` flag could
        ``shutil.rmtree`` the folder, but Phase A keeps it conservative.
        """
        if self._team is None:
            return
        # Snapshot members BEFORE cancellation so iteration is stable
        # while we mutate ``_runtimes`` / ``_member_task_ids``.
        for member in list(self._team.members):
            with contextlib.suppress(TeamError):
                self.remove_member(member.name, force=True)
        team_id = self._team.team_id
        self._team = None
        journal.write("team_deleted", team_id=team_id)

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
    ) -> TeammateMember:
        """Spawn a teammate, register its runtime task, persist the record.

        ``seed_prompt`` is the FIRST message the teammate consumes — it
        skips the mailbox and is fed directly into the first ``astream``
        iteration so a freshly-added teammate doesn't need a separate
        ``send_message`` to get going. Optional; ``None`` means "wait
        idle until the leader sends something".

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
        member = TeammateMember(
            name=name,
            agent_type=agent_type,
            system_prompt=system_prompt,
            model_name=model_name,
        )
        self._team.members.append(member)
        self._persist()
        # Register a TaskRecord for the teammate so /tasks + journal +
        # observability tooling all see it. ``kind="teammate"`` keeps it
        # distinct from one-shot subagents in /tasks output.
        prompt_for_task = seed_prompt or "(idle teammate; awaiting messages)"
        record = self._tasks_store.create(
            description=f"teammate: {name}",
            prompt=prompt_for_task,
            kind="teammate",
            agent_type=agent_type,
            metadata={"team_id": self._team.team_id, "member": name},
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
        )
        # Stamp the teammate identity onto the Agent so the SendMessage
        # tool can resolve the (team_id, sender) pair without reaching
        # back through the manager's private state.
        child.join_team(manager=self, member_name=name)
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
        # Done-callback: prune state on natural completion / cancel.
        def _cleanup(_t: asyncio.Task[None]) -> None:
            self._runtimes.pop(record.id, None)
            self._running_aborts.pop(record.id, None)
            with contextlib.suppress(Exception):
                if not _t.cancelled():
                    _t.exception()
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
        # Stash the stop_event on the manager so remove_member can fire
        # it without re-allocating; a member->event map avoids leaking
        # the event into the runtime's call signature.
        self._stop_events[name] = stop_event
        journal.write(
            "team_member_added",
            team_id=self._team.team_id,
            member=name,
            agent_type=agent_type,
            task_id=record.id,
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
        """Graceful shutdown with a real ``shutdown_response`` ack.

        Returns ``True`` when the teammate's runtime emitted a
        ``shutdown_response`` to the leader's inbox within
        ``timeout_sec``; ``False`` when the wait timed out and the
        member was force-killed instead. ``force=True`` short-circuits
        the wait and is equivalent to :meth:`remove_member`.

        Sequence:

        1. Drop the membership row so concurrent sends raise.
        2. Append a ``shutdown_request`` to the teammate's mailbox.
        3. Fire the per-member ``stop_event`` so the runtime exits at
           the next poll boundary.
        4. Poll the LEADER's mailbox off-thread for a matching
           ``shutdown_response`` (sender == ``name``) within
           ``timeout_sec``.
        5. On ack: journal ``team_member_shutdown_ack_received`` and
           let the runtime exit naturally; the done-callback prunes
           the task handle and abort entry.
        6. On timeout: journal ``team_member_shutdown_force_killed``
           and force-kill the runtime via abort + cancel.

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
        # Snapshot leader-mailbox state BEFORE we send the request so
        # the watcher only counts msg_ids that arrived after this call.
        # ``team_id`` resolves the inbox path; reading the seen set is
        # cheap (small txt file) and gives us the cursor we need.
        from aura.core.teams.mailbox import Mailbox  # local import; cycle-safe.
        mailbox = Mailbox(self._storage, team_id)
        baseline_ids = {m.msg_id for m in mailbox.read_all(TEAM_LEADER_NAME)}
        # Drop the row + send request + fire stop. We do NOT abort or
        # cancel yet — the runtime needs to live long enough to emit
        # its shutdown_response.
        idx = next(
            i for i, m in enumerate(self._team.members) if m.name == name
        )
        self._team.members.pop(idx)
        self._persist()
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
            stop_event.set()
        # Poll the leader inbox for the matching ack. We deliberately
        # use ``asyncio.to_thread`` for the blocking sleep so the
        # runtime task on the same loop gets cycles to drain its
        # mailbox + write the response. Polling cadence (200ms) is
        # 25x finer than the runtime's 5s mailbox poll, so the
        # latency of an ack arriving inside the runtime's cooperative
        # exit is bounded by the runtime's own poll, not ours.
        acked = await asyncio.to_thread(
            self._wait_for_shutdown_response,
            name,
            baseline_ids,
            timeout,
        )
        if acked:
            journal.write(
                "team_member_shutdown_ack_received",
                team_id=team_id,
                member=name,
            )
            # Clean teardown — the runtime has already exited or is
            # exiting; we just prune the bookkeeping and aclose the
            # child agent.
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

    def _wait_for_shutdown_response(
        self,
        member_name: str,
        baseline_ids: set[str],
        timeout: float,
    ) -> bool:
        """Block until a matching shutdown_response arrives or timeout.

        Runs in a worker thread (``asyncio.to_thread``) so it never
        blocks the event loop. Returns ``True`` on first matching
        message, ``False`` on timeout. Identifies the ack by
        ``sender == member_name AND kind == "shutdown_response"``;
        the ``baseline_ids`` set excludes pre-existing leader inbox
        lines so a stale ack from a previous run can't false-positive.
        """
        import time

        from aura.core.teams.mailbox import Mailbox
        if self._team is None:
            return False
        mailbox = Mailbox(self._storage, self._team.team_id)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for msg in mailbox.read_all(TEAM_LEADER_NAME):
                if msg.msg_id in baseline_ids:
                    continue
                if (
                    msg.sender == member_name
                    and msg.kind == "shutdown_response"
                ):
                    return True
            time.sleep(0.05)
        return False

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
            handle = self._runtimes.get(task_id)
            if handle is not None and not handle.done():
                handle.cancel()
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
            sent.append(msg)
        return sent

    def _post(self, msg: TeamMessage) -> None:
        """Internal append (skips fan-out + length checks; for control msgs)."""
        self.mailbox().append(msg)

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

    @classmethod
    def load(
        cls,
        *,
        leader: Agent,
        storage: SessionStorage,
        factory: SubagentFactory,
        running_aborts: dict[str, AbortController],
        tasks_store: TasksStore,
        team_id: str,
    ) -> TeamManager:
        """Rehydrate a manager from disk (Phase B uses this on resume).

        Phase A doesn't auto-rehydrate on startup but the loader is here
        so tests can persist a record, instantiate a manager, and assert
        the record round-trips. The members list is loaded WITHOUT
        respawning runtimes — that's a Phase B concern.
        """
        path = storage.team_config_path(team_id)
        raw = json.loads(path.read_text(encoding="utf-8"))
        record = TeamRecord.model_validate(raw)
        mgr = cls(
            leader=leader,
            storage=storage,
            factory=factory,
            running_aborts=running_aborts,
            tasks_store=tasks_store,
        )
        mgr._team = record
        return mgr


__all__ = [
    "TeamError",
    "TeamManager",
    "TeammateMemberStatus",
    "TeamViewSnapshot",
]
