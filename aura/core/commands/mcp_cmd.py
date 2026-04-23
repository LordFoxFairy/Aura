"""``/mcp`` — in-REPL MCP server management.

Mirrors claude-code's ``/mcp`` command surface, adapted to Aura's slash-
command Protocol + plain-text output. One ``MCPCommand`` class parses the
arg string into a subcommand (``list`` / ``enable`` / ``disable`` /
``reconnect`` / ``help``) and delegates to the agent's
:class:`~aura.core.mcp.manager.MCPManager` where appropriate.

Output discipline: every non-trivial print is monospace-aligned plain
text — we cannot use ``rich`` here because ``aura/core/**`` is barred
from UI-framework imports (see ``test_core_does_not_import_ui_frameworks``).
A column-aligned text table degrades fine in any terminal and doesn't
double-render when the REPL eventually pipes us through its own markup
renderer.

Design notes vs claude-code:
- claude-code's ``/mcp`` opens a React modal with separate submit /
  onComplete callbacks. We collapse the whole surface into one
  :class:`CommandResult` — Aura's REPL is line-at-a-time, not a TUI,
  so a one-shot print beats a stateful modal.
- claude-code's ``/mcp enable <target>`` accepts ``all`` as a special
  target. We don't — the arg is required and must match a known server.
  Rationale: ``/mcp list`` already exposes the set; making the operator
  type one name per action keeps the dispatch log readable.
- ``/mcp reload`` is *not* implemented in this release. The manager's
  configs list is fixed at Agent construction; rebuilding it would need
  surgery on ``aura/core/agent.py`` (which is out of scope per the task
  spec). Adding it later is the natural next step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aura.core.commands.types import CommandResult, CommandSource
from aura.core.mcp.manager import MCPServerStatus

if TYPE_CHECKING:
    from aura.core.agent import Agent


_VALID_SUBCOMMANDS = ("list", "enable", "disable", "reconnect", "help")


class MCPCommand:
    """``/mcp [list|enable|disable|reconnect|help] [name]`` — MCP control."""

    name = "/mcp"
    description = "list / enable / disable / reconnect MCP servers"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[list|enable|disable|reconnect|help] [name]"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        tokens = arg.split()
        # Bare ``/mcp`` → list (matches claude-code's default modal view).
        if not tokens:
            return self._list(agent)

        sub = tokens[0]
        rest = tokens[1:]

        if sub == "list":
            return self._list(agent)
        if sub == "help":
            return _help()
        if sub in {"enable", "disable", "reconnect"}:
            if not rest:
                return CommandResult(
                    handled=True,
                    kind="print",
                    text=f"usage: /mcp {sub} <server-name>",
                )
            target = " ".join(rest).strip()
            return await self._toggle(agent, sub, target)
        return _unknown_subcommand(sub)

    # ------------------------------------------------------------------
    # list view
    # ------------------------------------------------------------------

    def _list(self, agent: Agent) -> CommandResult:
        manager = agent._mcp_manager
        # Agent may not have called ``aconnect`` yet, or no servers are
        # configured at all. Both paths collapse to an empty status list;
        # the config-empty path is the common one.
        statuses: list[MCPServerStatus] = (
            [] if manager is None else manager.status()
        )
        if not statuses:
            return CommandResult(
                handled=True,
                kind="print",
                text="(no MCP servers configured)",
            )
        return CommandResult(
            handled=True, kind="print", text=_render_table(statuses),
        )

    # ------------------------------------------------------------------
    # enable / disable / reconnect dispatch
    # ------------------------------------------------------------------

    async def _toggle(
        self, agent: Agent, action: str, target: str,
    ) -> CommandResult:
        manager = agent._mcp_manager
        if manager is None:
            return CommandResult(
                handled=True,
                kind="print",
                text="no MCP manager attached (no servers configured)",
            )
        if action == "enable":
            text = await manager.enable(target)
        elif action == "disable":
            text = await manager.disable(target)
        else:  # reconnect
            text = await manager.reconnect(target)
        return CommandResult(handled=True, kind="print", text=text)


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def _render_table(statuses: list[MCPServerStatus]) -> str:
    """Monospace-aligned table for ``/mcp list``.

    Columns: NAME · TRANSPORT · STATUS · TOOLS · RESOURCES · PROMPTS.
    Status column carries the failure message inline for ``error`` rows
    (``error: <msg>``) so the operator doesn't need a second command to
    see why a server is down. Long error messages are truncated at 60
    chars to keep rows printable; the full message stays in the journal.
    """
    header = ("NAME", "TRANSPORT", "STATUS", "TOOLS", "RESOURCES", "PROMPTS")
    rows: list[tuple[str, str, str, str, str, str]] = [header]
    for s in statuses:
        status_cell = _format_status_cell(s)
        rows.append(
            (
                s.name,
                s.transport,
                status_cell,
                str(s.tool_count),
                str(s.resource_count),
                str(s.prompt_count),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines: list[str] = []
    for row in rows:
        lines.append(
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        )
    return "\n".join(lines)


_ERR_MAX = 60


def _format_status_cell(s: MCPServerStatus) -> str:
    if s.state == "error":
        msg = s.error_message or "unknown error"
        if len(msg) > _ERR_MAX:
            msg = msg[: _ERR_MAX - 1] + "…"
        return f"error: {msg}"
    return s.state


def _help() -> CommandResult:
    text = "\n".join(
        [
            "Usage: /mcp [subcommand] [args]",
            "",
            "Subcommands:",
            "  list                     show configured MCP servers (default)",
            "  enable <name>            connect a disabled/errored server",
            "  disable <name>           disconnect a currently-connected server",
            "  reconnect <name>         force-restart a server's connection",
            "  help                     show this message",
            "",
            "Running /mcp with no args is equivalent to /mcp list.",
        ]
    )
    return CommandResult(handled=True, kind="print", text=text)


def _unknown_subcommand(sub: str) -> CommandResult:
    valid = ", ".join(_VALID_SUBCOMMANDS)
    return CommandResult(
        handled=True,
        kind="print",
        text=(
            f"unknown /mcp subcommand {sub!r}; "
            f"valid: {valid}"
        ),
    )
