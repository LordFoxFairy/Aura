"""Built-in slash commands — the four migrated from the old if-else.

Each class satisfies the :class:`aura.core.commands.types.Command`
Protocol by duck-typing (no ABC inheritance).
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import TYPE_CHECKING

from aura.config.schema import AuraConfigError
from aura.core.commands.registry import CommandRegistry
from aura.core.commands.types import CommandResult, CommandSource
from aura.core.persistence.storage import SessionMeta

if TYPE_CHECKING:
    from aura.core.agent import Agent


class HelpCommand:
    """``/help`` — enumerate every registered command."""

    name = "/help"
    description = "show this message"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    def __init__(self, *, registry: CommandRegistry) -> None:
        # Keep a reference to the registry so /help always reflects the
        # live set of commands (critical once Skills/MCP register at runtime).
        self._registry = registry

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        commands = self._registry.list()
        # Group by source so the picker mirrors claude-code: Builtins →
        # Skills → MCP. Skipping an empty group keeps the output tight for
        # users who haven't wired any skills or MCP yet. Alignment is
        # preserved WITHIN each group (24-col label column, same as the
        # pre-grouping format); descriptions collapse to the first
        # non-empty line so a multi-line ``description:`` frontmatter
        # doesn't shatter the column. Mirrors the slash-completer fix
        # that already applies ``.split("\n", 1)[0].strip()`` to
        # ``display_meta``.
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
                # Collapse multi-line description to its first non-empty
                # line so a stray ``description:`` that spans multiple
                # paragraphs (common when skills are authored by LLMs)
                # doesn't shatter the 24-col alignment. Mirrors the
                # slash-completer fix at ``aura/cli/completion.py``
                # (``display_meta`` gets the same treatment).
                description = cmd.description.split("\n", 1)[0].strip()
                lines.append(f"    {label:<24} {description}")
        lines.append("")
        lines.append(
            "Keybindings: shift+tab cycles permission mode "
            "(default -> accept_edits -> plan) · esc resets to default."
        )
        lines.append("Anything else is sent as a prompt to the agent.")
        return CommandResult(
            handled=True, kind="view", text="\n".join(lines)
        )


class ExitCommand:
    """``/exit`` — signal the REPL to terminate."""

    name = "/exit"
    description = "exit the REPL (Ctrl+D also works)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        return CommandResult(handled=True, kind="exit", text="")


class ClearCommand:
    """``/clear`` — reset the current session's history."""

    name = "/clear"
    description = "clear the current session's history"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        agent.clear_session()
        return CommandResult(
            handled=True, kind="print", text="session cleared"
        )


class CompactCommand:
    """``/compact`` — summarize old history + preserve session state."""

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
    """``/context`` — print per-section token estimates for the live prompt.

    F-0910-016: introspection on what's filling the context window. Walks
    the live ``Context.build([])`` output, classifies each rendered
    HumanMessage into one of (system / memory / skills / files / history /
    other), and reports a 4-chars-per-token estimate per section plus a
    history estimate from the persisted session.
    """

    name = "/context"
    description = "show per-section token estimates"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        from langchain_core.messages import SystemMessage

        # Pinned prompt (system + memory + skills + ...): comes from
        # Context.build([]) — same path the live turn uses, so the numbers
        # match what the model sees.
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
            tokens = len(content) // 4
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

        # History: persisted messages — counted separately from "other" so
        # the operator can tell pinned-prefix bloat from conversation drift.
        history = agent._storage.load(agent.session_id)
        history_tokens = 0
        for msg in history:
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                content = str(content)
            history_tokens += len(content) // 4

        total = sum(sections.values()) + history_tokens
        window = agent.context_window
        pct = (total * 100 // window) if window > 0 else 0

        lines = ["Context token estimates (~4 chars/token):"]
        lines.append(f"  system   : {sections['system']:>8}")
        lines.append(f"  memory   : {sections['memory']:>8}")
        lines.append(f"  skills   : {sections['skills']:>8}")
        lines.append(f"  files    : {sections['files']:>8}")
        lines.append(f"  other    : {sections['other']:>8}")
        lines.append(f"  history  : {history_tokens:>8}")
        lines.append("  ─────────  ────────")
        lines.append(f"  total    : {total:>8}  ({pct}% of {window})")
        return CommandResult(
            handled=True, kind="view", text="\n".join(lines),
        )


class ModelCommand:
    """``/model [spec]`` — show or switch the current model."""

    name = "/model"
    description = "show or switch model (no arg = status)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[spec]"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        if not arg:
            return CommandResult(
                handled=True, kind="print", text=_model_status(agent)
            )
        old = agent.current_model or "?"
        try:
            agent.switch_model(arg)
        except AuraConfigError as exc:
            # Catches UnknownModelSpecError, MissingCredentialError,
            # MissingProviderDependencyError — any resolve/create failure
            # surfaces as a printable error instead of crashing the REPL.
            return CommandResult(
                handled=True, kind="print", text=f"error: {exc}"
            )
        new = agent.current_model or arg
        return CommandResult(
            handled=True, kind="print", text=f"model: {old} → {new}"
        )


def _model_status(agent: Agent) -> str:
    current = agent.current_model or "?"
    aliases = sorted(agent.router_aliases)
    lines = [f"current: {current}"]
    if aliases:
        # Align alias names so the arrows line up — matches claude-code's
        # ``/model`` picker formatting. Width from the longest alias name.
        width = max(len(a) for a in aliases)
        lines.append("aliases:")
        for alias in aliases:
            lines.append(
                f"  {alias:<{width}} → {agent.router_aliases[alias]}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resume session — Round 3A built-in slash command.
# ---------------------------------------------------------------------------


def format_relative_time(when: datetime, now: datetime | None = None) -> str:
    """Return a coarse human-readable "X ago" label for ``when``.

    Resolves to claude-code's relative buckets:

    - <10 s → ``"just now"``
    - <1 min → ``"<n>s ago"``
    - <1 h  → ``"<n> minute(s) ago"``
    - <1 d  → ``"<n> hour(s) ago"``
    - else  → ``"<n> day(s) ago"``

    Tolerates ``when > now`` (clock skew between SQLite host and caller)
    by clamping to ``"just now"`` instead of raising.
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
    """Compose a one-line picker label for a session.

    Shape: ``<short-id>  <first-prompt-preview>``. The id is truncated
    to its first 8 hex chars so the label fits the picker viewport even
    on a 40-col terminal; the picker's :func:`_truncate_to_width` then
    handles any further overflow per render.
    """
    short = meta.session_id[:8]
    preview = meta.first_user_prompt or "(no prompt)"
    return f"session-{short}  {preview}"


class ResumeCommand:
    """``/resume [session_id]`` — restore a previously persisted session.

    Without an arg this prints a list-style summary of recent sessions
    (the picker UI lives in the CLI layer; built-in commands stay UI-
    free). With an explicit id it short-circuits to
    :meth:`Agent.resume_session` and reports the message count.
    """

    name = "/resume"
    description = "resume a saved session (no arg lists recent sessions)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[session_id]"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        target = arg.strip()
        if not target:
            sessions = agent._storage.list_sessions(limit=10)
            if not sessions:
                return CommandResult(
                    handled=True, kind="print",
                    text="(no saved sessions)",
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
            return CommandResult(
                handled=True, kind="view", text="\n".join(lines),
            )
        try:
            count = agent.resume_session(target)
        except KeyError:
            return CommandResult(
                handled=True, kind="print",
                text=f"session {target!r} not found",
            )
        return CommandResult(
            handled=True, kind="print",
            text=f"resumed session {target} ({count} messages)",
        )


def restore_session_into(agent: Agent, session_id: str) -> int:
    """Deprecated: forwards to :meth:`Agent.resume_session`.

    Kept as a transitional alias so older callers (skill / MCP shims
    written against pre-v0.13 names) keep working. New code should call
    :meth:`Agent.resume_session` directly.
    """
    warnings.warn(
        "restore_session_into() is deprecated; "
        "use Agent.resume_session() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return agent.resume_session(session_id)
