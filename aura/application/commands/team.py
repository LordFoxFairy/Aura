"""``/team`` REPL surface; ``state.slots.active_team`` holds the slug; ``clear_session`` resets."""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import UTC, datetime

from aura.application.commands.types import CommandKind, CommandResult, CommandSource
from aura.application.teams.manager import (
    TeamError,
    TeamManager,
    TeamViewSnapshot,
)
from aura.core.agent import Agent
from aura.domain.team import BackendType, TeamRecord
from aura.infrastructure.persistence.storage import SessionStorage


def _set_active_team(agent: Agent, team_id: str | None) -> None:
    agent.state.slots = dataclasses.replace(
        agent.state.slots, active_team=team_id,
    )


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
  add <name> [agent_type] [model] [--backend <kind>]
                                   Spawn a teammate. ``--backend`` picks
                                   the runtime: ``in_process`` (default,
                                   asyncio task on the leader's loop) or
                                   ``pane`` (subprocess in a tmux split;
                                   requires ``$TMUX`` + tmux on PATH).
  remove <name>                    Graceful shutdown of a teammate.
  members                          Show live members + status.
  send <to> <body>                 Send a text message (no LLM).
  delete                           Tear down the active team.

Recipients: a member name, the literal 'leader', or 'broadcast'.
"""


_TEAMMATE_TAIL_CAP: int = 50

# Local check hints on typo before BackendUnavailable would raise.
_VALID_BACKENDS: tuple[BackendType, ...] = ("in_process", "pane")


class _AddUsageError(ValueError):
    pass


def _parse_add_args(rest: str) -> tuple[list[str], BackendType]:
    """Split ``/team add`` args; ``--backend <kind>`` may appear anywhere, positional order kept."""
    tokens = rest.split()
    backend_type: BackendType = "in_process"
    positional: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--backend":
            if i + 1 >= len(tokens):
                raise _AddUsageError(
                    "usage: /team add ... --backend <"
                    + "|".join(_VALID_BACKENDS) + ">",
                )
            raw = tokens[i + 1]
            if raw not in _VALID_BACKENDS:
                raise _AddUsageError(
                    f"unknown backend {raw!r}; expected one of "
                    + "|".join(_VALID_BACKENDS),
                )
            backend_type = raw
            i += 2
            continue
        positional.append(tok)
        i += 1
    return positional, backend_type


def _ensure_manager(agent: Agent) -> TeamManager | None:
    cached = agent._team_manager
    if isinstance(cached, TeamManager):
        return cached
    mgr = TeamManager(
        leader=agent,
        storage=agent.storage,
        factory=agent.subagent_factory,
        running_aborts=agent.running_aborts,
        tasks_store=agent.tasks_store,
    )
    agent._team_manager = mgr
    return mgr


def _resolve_team_id(
    *, name: str, manager: TeamManager, agent: Agent,
) -> str | None:
    """Resolve handle to slug; precedence: live team -> disk slug -> disk display name."""
    live = manager.team
    if live is not None and (name == live.team_id or name == live.name):
        return live.team_id
    on_disk = agent.storage.list_team_ids()
    if name in on_disk:
        return name
    for tid in on_disk:
        path = agent.storage.team_config_path(tid)
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
    """Compact ``Ns``/``Nm``/``Nh``/``Nd`` ago; ``None`` -> ``"-"`` for table alignment."""
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
    """Plain-text TeamViewSnapshot; fixed-width columns avoid view-renderer escape juggling."""
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
    """Render up to ``cap`` tail lines; malformed lines pass through (debug surface)."""
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
            handled=True, kind=kind, text=text,
        )

    async def _dispatch(  # noqa: PLR0911,PLR0912 — tight verb table
        self, verb: str, rest: str, agent: Agent,
    ) -> tuple[str, CommandKind]:
        mgr = _ensure_manager(agent)
        assert mgr is not None  # _ensure_manager only returns None pre-init
        if verb == "help":
            return _HELP, "view"
        if verb == "create":
            if not rest:
                return "usage: /team create <name>", "print"
            record = mgr.create_team(rest)
            agent.join_team(manager=mgr)
            _set_active_team(agent, record.team_id)
            return (
                f"team {record.team_id!r} created (leader={agent.session_id[:8]})",
                "print",
            )
        if verb == "list":
            ids = agent.storage.list_team_ids()
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
            try:
                positional, backend_type = _parse_add_args(rest)
            except _AddUsageError as exc:
                return str(exc), "print"
            if not positional:
                return (
                    "usage: /team add <name> [agent_type] [model] "
                    "[--backend in_process|pane]",
                    "print",
                )
            name = positional[0]
            agent_type = positional[1] if len(positional) > 1 else "general-purpose"
            model_name = positional[2] if len(positional) > 2 else None
            try:
                member = mgr.add_member(
                    name,
                    agent_type=agent_type,
                    model_name=model_name,
                    backend_type=backend_type,
                )
            except TeamError as exc:
                return f"error: {exc}", "print"
            suffix = (
                f" backend={member.backend_type}"
                if member.backend_type != "in_process"
                else ""
            )
            return (
                f"member {member.name!r} added "
                f"(agent_type={member.agent_type}{suffix})",
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
            # Drop active-team slot or next render hits "team not found" on a deleted folder.
            _set_active_team(agent, None)
            return f"team {tid!r} deleted", "print"
        return f"unknown subcommand {verb!r} — try /team help", "print"

    async def _enter(
        self, agent: Agent, mgr: TeamManager, rest: str,
    ) -> tuple[str, CommandKind]:
        """Auto-join only when resolved team is the live one; off-record just sets pointer."""
        if not rest:
            return "usage: /team enter <name>", "print"
        team_id = _resolve_team_id(name=rest, manager=mgr, agent=agent)
        if team_id is None:
            return f"team not found: {rest}", "print"
        _set_active_team(agent, team_id)
        live = mgr.team
        joined_msg = ""
        if live is not None and live.team_id == team_id and agent.team is None:
            agent.join_team(manager=mgr)
            joined_msg = " (leader joined)"
        elif live is None or live.team_id != team_id:
            joined_msg = (
                " (off-record team — /team add / send unavailable until "
                "/team create rehydrates this slug)"
            )
        return f"entered team {rest!r} (id={team_id}){joined_msg}", "print"

    def _leave(self, agent: Agent, mgr: TeamManager) -> tuple[str, CommandKind]:
        prev = agent.state.slots.active_team
        _set_active_team(agent, None)
        joined = agent.team is not None
        if joined:
            agent.leave_team()
        if prev is None and not joined:
            return "(no active team)", "print"
        suffix = " (leader detached)" if joined else ""
        return f"left team {prev or '(none)'}{suffix}", "print"

    def _view(
        self, agent: Agent, mgr: TeamManager, rest: str,
    ) -> tuple[str, CommandKind]:
        if rest:
            target_team_id = _resolve_team_id(name=rest, manager=mgr, agent=agent)
            if target_team_id is None:
                return f"team not found: {rest}", "print"
        else:
            target_team_id = agent.state.slots.active_team
            if target_team_id is None:
                return (
                    "no active team; pass /team view <name> "
                    "or /team enter <name> first",
                    "print",
                )
        return _render_view(mgr.view_state(target_team_id)), "view"

    def _teammate(
        self, agent: Agent, mgr: TeamManager, rest: str,
    ) -> tuple[str, CommandKind]:
        if not rest:
            return "usage: /team teammate <member>", "print"
        member = rest.split()[0]
        target_team_id = agent.state.slots.active_team
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
            agent.storage, target_team_id, member, _TEAMMATE_TAIL_CAP,
        )
        text = _render_teammate(
            member=member,
            team_id=target_team_id,
            lines=lines,
            cap=_TEAMMATE_TAIL_CAP,
        )
        return text, "view"


__all__ = ["TeamCommand"]
