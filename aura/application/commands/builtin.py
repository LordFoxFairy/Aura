"""Built-in slash commands."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from aura.application.commands.registry import CommandRegistry
from aura.application.commands.types import CommandResult, CommandSource
from aura.config.schema import AuraConfigError
from aura.infrastructure.persistence.storage import SessionMeta

if TYPE_CHECKING:
    from aura.core.agent import Agent


class HelpCommand:
    name = "/help"
    description = "show this message"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    def __init__(self, *, registry: CommandRegistry) -> None:
        self._registry = registry

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        commands = self._registry.list()
        sections: list[tuple[str, CommandSource]] = [
            ("Builtins", "builtin"),
            ("Skills", "skill"),
            ("MCP", "mcp"),
        ]
        lines = ["Available commands:"]
        for heading, source in sections:
            group = [c for c in commands if c.source == source]
            if not group:
                continue
            lines.append("")
            lines.append(f"  {heading}:")
            for cmd in group:
                hint = getattr(cmd, "argument_hint", None)
                label = f"{cmd.name} {hint}" if hint else cmd.name
                description = cmd.description.split("\n", 1)[0].strip()
                lines.append(f"    {label:<24} {description}")
        lines.append("")
        lines.append(
            "Keybindings: shift+tab cycles permission mode "
            "(default -> accept_edits -> plan) · esc resets to default."
        )
        lines.append("Anything else is sent as a prompt to the agent.")
        return CommandResult(handled=True, kind="view", text="\n".join(lines))


class ExitCommand:
    name = "/exit"
    description = "exit the REPL (Ctrl+D also works)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        return CommandResult(handled=True, kind="exit", text="")


class ClearCommand:
    name = "/clear"
    description = "clear the current session's history"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        agent.clear_session()
        return CommandResult(handled=True, kind="print", text="session cleared")


class CompactCommand:
    name = "/compact"
    description = "summarize history + preserve state"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        result = await agent.compact(source="manual")
        return CommandResult(
            handled=True,
            kind="print",
            text=(
                f"compact applied ({result.before_tokens} -> "
                f"{result.after_tokens} tokens)"
            ),
        )


class ContextCommand:
    name = "/context"
    description = "show per-section token estimates"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        from langchain_core.messages import SystemMessage

        from aura.application.compact.compact import (
            compact_summary_messages,
            estimate_compact_summary_tokens,
        )
        from aura.domain.tokens import estimate_message_tokens

        sections: dict[str, int] = {
            "system": 0,
            "memory": 0,
            "skills": 0,
            "files": 0,
            "other": 0,
        }
        for msg in agent._context.build([]):
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                content = str(content)
            tokens = estimate_message_tokens(msg)
            if isinstance(msg, SystemMessage):
                sections["system"] += tokens
            elif "<project-memory>" in content or "<nested-memory" in content:
                sections["memory"] += tokens
            elif (
                "<skills-available>" in content
                or "<skill-invoked" in content
                or "<skill-active" in content
            ):
                sections["skills"] += tokens
            elif "<recent-file" in content:
                sections["files"] += tokens
            else:
                sections["other"] += tokens

        history = agent.storage.load(agent.session_id)
        raw_history_tokens = sum(estimate_message_tokens(msg) for msg in history)
        summary_history = compact_summary_messages(agent, history)
        history_tokens = sum(estimate_message_tokens(msg) for msg in summary_history)

        tail_count = 6
        compact_candidates = (
            summary_history[:-tail_count] if len(summary_history) > tail_count else []
        )
        compact_tokens = (
            estimate_compact_summary_tokens(compact_candidates)
            if compact_candidates else 0
        )

        total = sum(sections.values()) + history_tokens
        window = agent.context_window
        pct = (total * 100 // window) if window > 0 else 0

        lines = ["Context token estimates (conservative local estimate):"]
        lines.append(f"  system   : {sections['system']:>8}")
        lines.append(f"  memory   : {sections['memory']:>8}")
        lines.append(f"  skills   : {sections['skills']:>8}")
        lines.append(f"  files    : {sections['files']:>8}")
        lines.append(f"  other    : {sections['other']:>8}")
        lines.append(f"  history  : {history_tokens:>8}")
        lines.append(f"  raw-store: {raw_history_tokens:>8}  (not sent as-is)")
        lines.append(f"  compact  : {compact_tokens:>8}  (manual /compact summary prompt)")
        lines.append("  ─────────  ────────")
        lines.append(f"  total    : {total:>8}  ({pct}% of {window})")
        return CommandResult(handled=True, kind="view", text="\n".join(lines))


class ModelCommand:
    name = "/model"
    description = "show or switch model (no arg = status)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[spec]"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        if not arg:
            return CommandResult(handled=True, kind="print", text=_model_status(agent))
        old = agent.current_model or "?"
        try:
            agent.switch_model(arg)
        except AuraConfigError as exc:
            return CommandResult(handled=True, kind="print", text=f"error: {exc}")
        new = agent.current_model or arg
        return CommandResult(
            handled=True, kind="print", text=f"model: {old} → {new}"
        )


def _model_status(agent: Agent) -> str:
    current = agent.current_model or "?"
    aliases = sorted(agent.router_aliases)
    lines = [f"current: {current}"]
    if aliases:
        width = max(len(a) for a in aliases)
        lines.append("aliases:")
        for alias in aliases:
            lines.append(f"  {alias:<{width}} → {agent.router_aliases[alias]}")
    return "\n".join(lines)


def format_relative_time(when: datetime, now: datetime | None = None) -> str:
    """Return a coarse human-readable "X ago" label for ``when``.

    Buckets: <10s → "just now", <1min → "Ns ago", <1h → "N minute(s) ago",
    <1d → "N hour(s) ago", else "N day(s) ago". Clamps ``when > now`` to
    "just now" so SQLite host clock skew doesn't raise.
    """
    current = now or datetime.now()
    diff = (current - when).total_seconds()
    if diff < 10:
        return "just now"
    if diff < 60:
        return f"{int(diff)}s ago"
    if diff < 3600:
        minutes = int(diff // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if diff < 86400:
        hours = int(diff // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(diff // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def session_label(meta: SessionMeta) -> str:
    """``<short-id>  <first-prompt-preview>`` — picker row label."""
    short = meta.session_id[:8]
    preview = meta.first_user_prompt or "(no prompt)"
    return f"session-{short}  {preview}"


class ResumeCommand:
    name = "/resume"
    description = "resume a saved session (no arg lists recent sessions)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[session_id]"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        target = arg.strip()
        if not target:
            sessions = agent.storage.list_sessions(limit=10)
            if not sessions:
                return CommandResult(
                    handled=True, kind="print", text="(no saved sessions)",
                )
            now = datetime.now()
            lines = ["recent sessions:"]
            for meta in sessions:
                lines.append(
                    f"  {session_label(meta)}  "
                    f"({format_relative_time(meta.last_used_at, now)})"
                )
            lines.append("")
            lines.append("/resume <session_id> to restore one.")
            return CommandResult(handled=True, kind="view", text="\n".join(lines))
        try:
            count = agent.resume_session(target)
        except KeyError:
            return CommandResult(
                handled=True, kind="print", text=f"session {target!r} not found",
            )
        return CommandResult(
            handled=True, kind="print",
            text=f"resumed session {target} ({count} messages)",
        )
