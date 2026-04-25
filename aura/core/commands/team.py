"""``/team`` slash commands — Phase A team lifecycle + view UX from the REPL.

The single ``TeamCommand`` parses the first whitespace-delimited token
of ``arg`` as the subcommand. Keeping one slash command (rather than
``/team-create``/``/team-add``/...) matches claude-code's verb-based
form and stays out of the way of the existing CLI's `/task-*` pattern.

Round V14 — added ``enter`` / ``leave`` / ``view`` / ``teammate``
verbs that mirror claude-code's ``enterTeammateView`` /
``exitTeammateView`` UX. The "active team" pointer lives on
:attr:`LoopState.custom` under ``"active_team_id"`` so:

- it survives across turns inside one REPL session,
- :meth:`Agent.clear_session` (which calls ``LoopState.reset`` →
  ``custom.clear()``) drops it (matches claude-code's clear-on-/clear),
- it's stored as a *slug*, not a human name, because two teams can
  share a display name and only the slug is unique.

Reading ``active_team_id`` via ``agent.state.custom`` is the public
path (``Agent.state`` is a documented property). It deliberately
avoids a typed property on ``Agent`` for now — Phase B will lift this
to ``Agent.active_team_id`` once the Agent surface is being touched
for sidebar / rehydration work, so we don't churn the public API for
a single REPL feature.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from aura.core.commands.types import CommandKind, CommandResult, CommandSource
from aura.core.persistence.storage import SessionStorage
from aura.core.teams.manager import (
    TeamError,
    TeamManager,
    TeamViewSnapshot,
)
from aura.core.teams.types import TeamRecord

if TYPE_CHECKING:
    from aura.core.agent import Agent


#: Slot name on :attr:`LoopState.custom` holding the slug of the team
#: the user has "entered". ``None`` / missing means no active team.
#: Stored under a stable key so the REPL renderer + every subcommand
#: agree on the same cell without a typed property to coordinate
#: through.
_ACTIVE_TEAM_KEY: str = "active_team_id"


_HELP = """\
/team <subcommand> [args]

  create <name>                    Create a team and become its leader.
  list                             List teams under the storage root.
  enter <name>                     Set <name> as the active team for this
                                   REPL session — subsequent /team add /
                                   remove / send default to it.
  leave                            Clear the active team and (if joined)
                                   detach the leader Agent from it.
  view [<name>]                    Show the active team's full status:
                                   members, recent messages, transcripts.
                                   Pass <name> to view a team you haven't
                                   entered.
  teammate <member>                Render one teammate's last 50 transcript
                                   entries in the same style as the main
                                   agent transcript.
  add <name> [agent_type] [model]  Spawn a teammate.
  remove <name>                    Graceful shutdown of a teammate.
  members                          Show live members + status.
  send <to> <body>                 Send a text message (no LLM).
  delete                           Tear down the active team.

Recipients: a member name, the literal 'leader', or 'broadcast'.
"""


#: Cap on the per-teammate transcript tail rendered by
#: ``/team teammate``. 50 entries is the sweet spot — long enough for
#: a meaningful debug trail, short enough that the REPL panel stays
#: scrollable on a 24-line terminal.
_TEAMMATE_TAIL_CAP: int = 50


def _ensure_manager(agent: Agent) -> TeamManager | None:
    """Return the agent's TeamManager, creating one if absent.

    Allocates a new manager bound to the agent's existing factory +
    storage on first use. Subsequent calls return the same instance —
    Phase A invariant of one manager per agent.
    """
    existing = getattr(agent, "_team_manager", None)
    if existing is not None:
        return existing  # type: ignore[no-any-return]
    mgr = TeamManager(
        leader=agent,
        storage=agent._storage,
        factory=agent._subagent_factory,
        running_aborts=agent._running_aborts,
        tasks_store=agent._tasks_store,
    )
    agent._team_manager = mgr  # type: ignore[attr-defined]
    return mgr


def _resolve_team_id(
    *, name: str, manager: TeamManager, agent: Agent,
) -> str | None:
    """Resolve a user-typed team handle to a stored ``team_id`` slug.

    ``name`` may be:
      - the canonical slug (matches a folder under ``teams/``);
      - the display name of the live team (``manager.team.name``);
      - the display name persisted in another team's ``config.json``.

    Returns the resolved slug or ``None`` when no team matches. Pure
    read — no side effects on the manager or storage. Order of
    precedence: live team match first, then exact-slug-on-disk, then
    name-on-disk (loads each ``config.json`` only if the prior checks
    miss, so the common path is two cheap dict lookups).
    """
    live = manager.team
    if live is not None and (name == live.team_id or name == live.name):
        return live.team_id
    on_disk = agent._storage.list_team_ids()
    if name in on_disk:
        return name
    for tid in on_disk:
        path = agent._storage.team_config_path(tid)
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            record = TeamRecord.model_validate(raw)
        except Exception:  # noqa: BLE001 — corrupt config never blocks lookup
            continue
        if record.name == name:
            return record.team_id
    return None


def _format_age(now: float, then: float | None) -> str:
    """Return a compact "X ago" string for a wall-clock float.

    Mirrors claude-code's ``formatTimeAgo`` shape: ``Ns`` / ``Nm`` /
    ``Nh`` / ``Nd``. Used by :func:`_render_view` and
    :func:`_render_teammate` so timestamps fit on one line. ``None``
    yields ``"-"`` so the caller can table-align.
    """
    if then is None:
        return "-"
    delta = max(0.0, now - then)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def _render_view(snap: TeamViewSnapshot) -> str:
    """Plain-text render of a :class:`TeamViewSnapshot` for ``kind="view"``.

    Two stacked sections — members table, recent messages — followed
    by a one-line stats summary and the footer hint pointing at
    ``/team teammate``. Plain text (not rich.Table) keeps the output
    portable: ``CommandResult.text`` flows through the REPL's view
    renderer which wraps it in a Panel; rich markup inside the panel
    would need separate escaping. The columns are space-padded to
    fixed widths so the visual alignment stays stable across teams
    with one-letter names AND teams with full-length 64-char names.
    """
    now = time.time()
    lines: list[str] = []
    lines.append(f"team: {snap.name} (id={snap.team_id})")
    lines.append("")
    if not snap.members:
        lines.append("(no members)")
    else:
        header = (
            f"  {'NAME':<20} {'TYPE':<18} {'MODEL':<28} "
            f"{'STATUS':<14} {'TOKENS':>8}  LAST"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for m in snap.members:
            model = m.model_spec or "(inherits)"
            lines.append(
                f"  {m.name:<20.20} {m.agent_type:<18.18} "
                f"{model:<28.28} {m.status:<14} "
                f"{m.tokens_used:>8}  {_format_age(now, m.last_active)}",
            )
    lines.append("")
    lines.append(f"recent messages (last {len(snap.recent_messages)}):")
    if not snap.recent_messages:
        lines.append("  (none)")
    else:
        for msg in snap.recent_messages:
            ts = _format_age(now, msg.sent_at)
            body = msg.body.replace("\n", " ")
            if len(body) > 80:
                body = body[:77] + "..."
            lines.append(
                f"  [{ts}] {msg.sender} -> {msg.recipient}: {body}",
            )
    lines.append("")
    lines.append(
        f"subagents: {snap.subagent_count}  ·  "
        f"teammate transcripts: {snap.transcript_count}",
    )
    lines.append("")
    lines.append(
        "Type /team teammate <member> to inspect a "
        "specific member's transcript.",
    )
    return "\n".join(lines)


def _render_teammate(
    *, member: str, team_id: str, lines: list[str], cap: int,
) -> str:
    """Render up to ``cap`` tail lines of ``member``'s transcript.

    The transcript JSONL is written by :func:`run_teammate` as one
    line per agent event:
        ``<unix-ts> <EventName> [Final body...]``
    We re-format each line into a one-line preview anchored on a
    human-readable timestamp + the event name + the body. Malformed
    lines pass through verbatim so a partial write never breaks the
    view (resilience contract: this is a debug surface, not the
    source of truth).
    """
    out: list[str] = []
    out.append(f"transcript: {member} (team={team_id})")
    out.append(f"showing last {len(lines)} of {cap} max")
    out.append("")
    if not lines:
        out.append("(transcript empty — teammate hasn't run yet)")
    else:
        for raw in lines:
            stripped = raw.rstrip("\n")
            parts = stripped.split(" ", 2)
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    ts_str = datetime.fromtimestamp(
                        int(parts[0]), tz=UTC,
                    ).strftime("%H:%M:%S")
                except (ValueError, OSError):
                    ts_str = parts[0]
                event = parts[1]
                body = parts[2] if len(parts) > 2 else ""
                if len(body) > 200:
                    body = body[:197] + "..."
                out.append(f"  {ts_str} {event:<14} {body}")
            else:
                out.append(f"  {stripped}")
    out.append("")
    out.append("Esc to return  ·  /team view to go back to team summary")
    return "\n".join(out)


def _read_transcript_tail(
    storage: SessionStorage, team_id: str, member: str, cap: int,
) -> list[str]:
    """Return the last ``cap`` lines of ``member``'s transcript, or ``[]``.

    Robust to missing file, missing dir, and decode errors — the
    caller surfaces "(empty)" in either case so a freshly-spawned
    teammate that hasn't produced output yet doesn't crash the view.
    """
    path = storage.team_transcript_path(team_id, member)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            all_lines = f.readlines()
    except OSError:
        return []
    return all_lines[-cap:]


class TeamCommand:
    """One slash entry, dispatch by subcommand verb."""

    name = "/team"
    description = (
        "team lifecycle (create/list/enter/leave/view/teammate/"
        "add/remove/members/send/delete)"
    )
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "<subcommand> [args]"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        if not arg.strip():
            return CommandResult(
                handled=True, kind="view", text=_HELP,
            )
        parts = arg.split(None, 1)
        verb = parts[0]
        rest = parts[1].strip() if len(parts) > 1 else ""
        try:
            text, kind = await self._dispatch(verb, rest, agent)
        except TeamError as exc:
            return CommandResult(
                handled=True, kind="print", text=f"team error: {exc}",
            )
        return CommandResult(
            handled=True, kind=cast("CommandKind", kind), text=text,
        )

    async def _dispatch(  # noqa: PLR0911,PLR0912 — tight verb table
        self, verb: str, rest: str, agent: Agent,
    ) -> tuple[str, str]:
        mgr = _ensure_manager(agent)
        assert mgr is not None  # _ensure_manager only returns None pre-init
        if verb == "help":
            return _HELP, "view"
        if verb == "create":
            if not rest:
                return "usage: /team create <name>", "print"
            record = mgr.create_team(rest)
            agent.join_team(manager=mgr)
            return (
                f"team {record.team_id!r} created (leader={agent.session_id[:8]})",
                "print",
            )
        if verb == "list":
            ids = agent._storage.list_team_ids()
            if not ids:
                return "(no teams on disk)", "print"
            active = mgr.team.team_id if mgr.team is not None else None
            lines = []
            for tid in ids:
                marker = "* " if tid == active else "  "
                lines.append(f"{marker}{tid}")
            return "\n".join(lines), "view"
        if verb == "enter":
            return await self._enter(agent, mgr, rest)
        if verb == "leave":
            return self._leave(agent, mgr)
        if verb == "view":
            return self._view(agent, mgr, rest)
        if verb == "teammate":
            return self._teammate(agent, mgr, rest)
        if verb == "add":
            tokens = rest.split()
            if not tokens:
                return (
                    "usage: /team add <name> [agent_type] [model]", "print",
                )
            name = tokens[0]
            agent_type = tokens[1] if len(tokens) > 1 else "general-purpose"
            model_name = tokens[2] if len(tokens) > 2 else None
            member = mgr.add_member(
                name, agent_type=agent_type, model_name=model_name,
            )
            return (
                f"member {member.name!r} added (agent_type={member.agent_type})",
                "print",
            )
        if verb == "remove":
            if not rest:
                return "usage: /team remove <name>", "print"
            mgr.remove_member(rest)
            return f"member {rest!r} removed", "print"
        if verb == "members":
            members = mgr.list_members()
            if not members:
                return "(no members)", "print"
            lines = []
            for m in members:
                lines.append(
                    f"{m.name:<20} {m.agent_type:<16} "
                    f"{'active' if m.is_active else 'inactive'}",
                )
            return "\n".join(lines), "view"
        if verb == "send":
            send_tokens = rest.split(None, 1)
            if len(send_tokens) < 2:
                return "usage: /team send <to> <body>", "print"
            to, body = send_tokens[0], send_tokens[1]
            sent = mgr.send(sender="leader", recipient=to, body=body)
            id_str = ", ".join(s.msg_id[:8] for s in sent)
            return (
                f"sent {len(sent)} message(s) (msg_id={id_str})", "print",
            )
        if verb == "delete":
            if mgr.team is None:
                return "(no active team)", "print"
            tid = mgr.team.team_id
            mgr.delete_team()
            agent.leave_team()
            # Drop the active-team pointer too — the team it pointed
            # at no longer exists, leaving the slot stale would surface
            # as a "team not found" the next time the renderer ran.
            agent.state.custom.pop(_ACTIVE_TEAM_KEY, None)
            return f"team {tid!r} deleted", "print"
        return f"unknown subcommand {verb!r} — try /team help", "print"

    # ------------------------------------------------------------------
    # New verbs (enter / leave / view / teammate)
    # ------------------------------------------------------------------

    async def _enter(
        self, agent: Agent, mgr: TeamManager, rest: str,
    ) -> tuple[str, str]:
        """Set the named team as the REPL's active team.

        If the named team is the live one (already created by this
        leader) we just stamp the active-team slot — no auto-join is
        needed, the leader is already a member. If the team exists on
        disk but isn't this leader's live team we cannot auto-rehydrate
        it (Phase B work — see :meth:`TeamManager.load`); we set the
        pointer and surface a hint so the operator knows ``/team add``
        / ``send`` won't work until they ``/team create`` or rehydrate.
        """
        if not rest:
            return "usage: /team enter <name>", "print"
        team_id = _resolve_team_id(name=rest, manager=mgr, agent=agent)
        if team_id is None:
            return f"team not found: {rest}", "print"
        agent.state.custom[_ACTIVE_TEAM_KEY] = team_id
        # If the leader Agent isn't in any team yet AND the resolved
        # team is the manager's live team, auto-join. We deliberately
        # don't auto-join when the team is off-record — the manager
        # only owns one team at a time (Phase A invariant) and a stale
        # join_team(manager=mgr) would point the leader at the wrong
        # record.
        live = mgr.team
        joined_msg = ""
        if (
            live is not None
            and live.team_id == team_id
            and agent.team is None
        ):
            agent.join_team(manager=mgr)
            joined_msg = " (leader joined)"
        elif live is None or live.team_id != team_id:
            joined_msg = (
                " (off-record team — /team add / send unavailable until "
                "/team create rehydrates this slug)"
            )
        return (
            f"entered team {rest!r} (id={team_id}){joined_msg}",
            "print",
        )

    def _leave(self, agent: Agent, mgr: TeamManager) -> tuple[str, str]:
        """Clear the active-team pointer and detach the leader Agent.

        Idempotent: a no-op when there's no active team and the
        leader isn't joined. The "left team X" message is what the
        REPL renders so the operator sees positive confirmation —
        otherwise the silently-cleared status line could feel like
        nothing happened.
        """
        prev = agent.state.custom.pop(_ACTIVE_TEAM_KEY, None)
        joined = agent.team is not None
        if joined:
            agent.leave_team()
        if prev is None and not joined:
            return "(no active team)", "print"
        suffix = " (leader detached)" if joined else ""
        prev_label = prev if prev is not None else "(none)"
        return f"left team {prev_label}{suffix}", "print"

    def _view(
        self, agent: Agent, mgr: TeamManager, rest: str,
    ) -> tuple[str, str]:
        """Render a :class:`TeamViewSnapshot` as a ``kind="view"`` payload.

        ``rest`` is an optional explicit team name; when missing we
        fall back to the active-team pointer. A miss on both lands as
        a clear error rather than a silent "(no data)" so the
        operator immediately knows what to do.
        """
        target_team_id: str | None = None
        if rest:
            target_team_id = _resolve_team_id(
                name=rest, manager=mgr, agent=agent,
            )
            if target_team_id is None:
                return f"team not found: {rest}", "print"
        else:
            target_team_id = agent.state.custom.get(_ACTIVE_TEAM_KEY)
            if target_team_id is None:
                return (
                    "no active team; pass /team view <name> "
                    "or /team enter <name> first",
                    "print",
                )
        snap = mgr.view_state(target_team_id)
        return _render_view(snap), "view"

    def _teammate(
        self, agent: Agent, mgr: TeamManager, rest: str,
    ) -> tuple[str, str]:
        """Render the last ``_TEAMMATE_TAIL_CAP`` transcript entries for ``rest``.

        Resolves the team from the active-team slot. Errors when no
        team is active (the operator should ``/team enter`` first) or
        when the named member isn't part of the resolved team — both
        are operator-facing so we keep the messages explicit.
        """
        if not rest:
            return "usage: /team teammate <member>", "print"
        member = rest.split()[0]
        target_team_id = agent.state.custom.get(_ACTIVE_TEAM_KEY)
        if target_team_id is None:
            return (
                "no active team; /team enter <name> first, "
                "then /team teammate <member>",
                "print",
            )
        snap = mgr.view_state(target_team_id)
        if not any(m.name == member for m in snap.members):
            valid = ", ".join(m.name for m in snap.members) or "(none)"
            return (
                f"member not found: {member} (members: {valid})",
                "print",
            )
        lines = _read_transcript_tail(
            agent._storage, target_team_id, member, _TEAMMATE_TAIL_CAP,
        )
        text = _render_teammate(
            member=member,
            team_id=target_team_id,
            lines=lines,
            cap=_TEAMMATE_TAIL_CAP,
        )
        return text, "view"


__all__ = ["TeamCommand"]
