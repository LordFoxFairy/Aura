"""``/team`` slash commands — Phase A team lifecycle from the REPL.

The single ``TeamCommand`` parses the first whitespace-delimited token
of ``arg`` as the subcommand. Keeping one slash command (rather than
``/team-create``/``/team-add``/...) matches claude-code's verb-based
form and stays out of the way of the existing CLI's `/task-*` pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from aura.core.commands.types import CommandKind, CommandResult, CommandSource
from aura.core.teams.manager import TeamError, TeamManager

if TYPE_CHECKING:
    from aura.core.agent import Agent


_HELP = """\
/team <subcommand> [args]

  create <name>                    Create a team and become its leader.
  list                             List teams under the storage root.
  add <name> [agent_type] [model]  Spawn a teammate.
  remove <name>                    Graceful shutdown of a teammate.
  members                          Show live members + status.
  send <to> <body>                 Send a text message (no LLM).
  delete                           Tear down the active team.

Recipients: a member name, the literal 'leader', or 'broadcast'.
"""


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


class TeamCommand:
    """One slash entry, dispatch by subcommand verb."""

    name = "/team"
    description = "team lifecycle (create/list/add/remove/members/send/delete)"
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
            text, kind = self._dispatch(verb, rest, agent)
        except TeamError as exc:
            return CommandResult(
                handled=True, kind="print", text=f"team error: {exc}",
            )
        return CommandResult(
            handled=True, kind=cast("CommandKind", kind), text=text,
        )

    def _dispatch(  # noqa: PLR0911,PLR0912 — tight verb table
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
            return f"team {tid!r} deleted", "print"
        return f"unknown subcommand {verb!r} — try /team help", "print"


__all__ = ["TeamCommand"]
