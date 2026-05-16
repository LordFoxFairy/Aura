"""/export — dump the current session transcript to a file.

Two formats:
    markdown (default) — human-readable, with an envelope + per-turn
        sections and fenced code blocks for tool results.
    json — LangChain-shaped message array wrapped in a metadata envelope.

Source of truth is ``Agent._storage.load(session_id)`` — we never reach
into the live loop state or mutate storage. Failures (permission denied,
bad path, disk full) return an error :class:`CommandResult` rather than
crashing the REPL: a broken export must never take the shell down with
it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)

from aura.core.commands.types import CommandResult, CommandSource
from aura.core.persistence import journal

if TYPE_CHECKING:
    from aura.core.agent import Agent


Format = Literal["md", "json"]

_DEFAULT_DIR = Path("~/.aura/exports")
_MD_EXTS = frozenset({".md", ".markdown"})
_JSON_EXTS = frozenset({".json"})


class ExportCommand:
    """``/export [path] [--format md|json]`` — save session transcript."""

    name = "/export"
    description = "export session transcript to a file"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[path] [--format md|json]"

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        try:
            path_arg, fmt_arg = _parse_args(arg)
        except ValueError as exc:
            return CommandResult(
                handled=True, kind="print", text=f"error: {exc}",
            )

        messages = agent._storage.load(agent.session_id)
        target_path, fmt, note = _resolve_target(path_arg, fmt_arg)

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "json":
                payload = _render_json(agent, messages)
            else:
                payload = _render_markdown(agent, messages)
            target_path.write_text(payload, encoding="utf-8", newline="\n")
        except OSError as exc:
            journal.write(
                "export_failed",
                session=agent.session_id,
                path=str(target_path),
                error=f"{type(exc).__name__}: {exc}",
            )
            return CommandResult(
                handled=True,
                kind="print",
                text=f"error: could not write {target_path}: {exc}",
            )

        turns = sum(1 for m in messages if isinstance(m, HumanMessage))
        journal.write(
            "export_written",
            session=agent.session_id,
            path=str(target_path),
            format=fmt,
            turns=turns,
        )
        prefix = note + "\n" if note else ""
        return CommandResult(
            handled=True,
            kind="print",
            text=f"{prefix}exported {turns} turns to {target_path}",
        )


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------


def _parse_args(arg: str) -> tuple[str | None, Format | None]:
    """Split ``arg`` into (path, explicit-format).

    Supported shapes:
        ""                        -> (None, None)
        "path"                    -> ("path", None)
        "--format md"             -> (None, "md")
        "--format json"           -> (None, "json")
        "path --format json"      -> ("path", "json")
        "--format md path"        -> ("path", "md")
    """
    tokens = arg.split()
    path: str | None = None
    fmt: Format | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--format":
            if i + 1 >= len(tokens):
                raise ValueError("--format requires an argument (md|json)")
            value = tokens[i + 1]
            if value not in {"md", "json"}:
                raise ValueError(
                    f"unknown format {value!r}; expected 'md' or 'json'"
                )
            fmt = value  # type: ignore[assignment]
            i += 2
            continue
        if tok.startswith("--"):
            raise ValueError(f"unknown flag {tok!r}")
        if path is not None:
            raise ValueError(f"unexpected extra argument {tok!r}")
        path = tok
        i += 1
    return path, fmt


def _resolve_target(
    path_arg: str | None, fmt_arg: Format | None,
) -> tuple[Path, Format, str]:
    """Resolve the final (path, format, note) from user input.

    ``note`` is an optional hint appended to the success message — used
    when the extension didn't match a known format and we fell back to
    markdown.
    """
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    note = ""

    if path_arg is None:
        fmt: Format = fmt_arg or "md"
        ext = ".json" if fmt == "json" else ".md"
        path = _DEFAULT_DIR.expanduser() / f"aura-session-{timestamp}{ext}"
        return path, fmt, note

    path = Path(path_arg).expanduser()
    # Directory target: append the default filename inside it.
    if path.is_dir() or path_arg.endswith(("/", "\\")):
        fmt = fmt_arg or "md"
        ext = ".json" if fmt == "json" else ".md"
        path = path / f"aura-session-{timestamp}{ext}"
        return path, fmt, note

    # Infer format from extension unless --format was explicit.
    if fmt_arg is not None:
        fmt = fmt_arg
    else:
        suffix = path.suffix.lower()
        if suffix in _JSON_EXTS:
            fmt = "json"
        elif suffix in _MD_EXTS:
            fmt = "md"
        else:
            fmt = "md"
            note = (
                f"note: unknown extension {suffix!r}, writing markdown"
            )
    return path, fmt, note


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _envelope(agent: Agent, messages: list[BaseMessage]) -> dict[str, object]:
    turns = sum(1 for m in messages if isinstance(m, HumanMessage))
    return {
        "session_id": agent.session_id,
        "exported_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": agent.current_model,
        "cwd": str(agent._cwd),
        "turns": turns,
        "total_tokens": agent.state.total_tokens_used,
    }


def _render_json(agent: Agent, messages: list[BaseMessage]) -> str:
    msg_dicts: list[dict[str, object]] = []
    for m in messages:
        entry: dict[str, object] = {
            "role": m.type,
            "content": _content_as_str(m.content),
        }
        if isinstance(m, AIMessage) and m.tool_calls:
            # Keep only the LangChain-public fields so a round-trip via
            # messages_from_dict stays possible if a future consumer
            # wants to re-hydrate.
            entry["tool_calls"] = [
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                }
                for tc in m.tool_calls
            ]
        if isinstance(m, ToolMessage):
            entry["tool_call_id"] = m.tool_call_id
        msg_dicts.append(entry)
    envelope = _envelope(agent, messages)
    envelope["messages"] = msg_dicts
    return json.dumps(envelope, indent=2, ensure_ascii=False) + "\n"


def _render_markdown(agent: Agent, messages: list[BaseMessage]) -> str:
    env = _envelope(agent, messages)
    lines: list[str] = [
        "# Aura session export",
        "",
        f"- session_id: {env['session_id']}",
        f"- timestamp: {env['exported_at']}",
        f"- model: {env['model']}",
        f"- cwd: {env['cwd']}",
        f"- turns: {env['turns']}",
        f"- total tokens: {env['total_tokens']:,}",
        "",
        "---",
        "",
    ]

    turn_no = 0
    for m in messages:
        if isinstance(m, HumanMessage):
            turn_no += 1
            lines.append(f"## Turn {turn_no} (user)")
            lines.append("")
            lines.append(_content_as_str(m.content))
            lines.append("")
        elif isinstance(m, AIMessage):
            # Assistant messages before the first HumanMessage (rare but
            # possible after a compact summary) get grouped under turn 0
            # rather than crashing.
            lines.append(f"## Turn {turn_no} (assistant)")
            lines.append("")
            content = _content_as_str(m.content).strip()
            if content:
                lines.append(content)
                lines.append("")
            if m.tool_calls:
                lines.append("### Tool calls")
                for tc in m.tool_calls:
                    lines.append(_format_tool_call(tc))
                lines.append("")
        elif isinstance(m, ToolMessage):
            tool_name = m.name or "tool"
            lines.append(f"## Turn {turn_no} (tool: {tool_name})")
            lines.append("")
            body = _content_as_str(m.content)
            lang = _guess_lang(tool_name, body)
            fence = "```" + lang
            lines.append(fence)
            lines.append(body)
            lines.append("```")
            lines.append("")
        else:
            # SystemMessage or any future subclass — include verbatim so
            # the export is lossless even if we don't pretty-print it.
            lines.append(f"## Turn {turn_no} ({m.type})")
            lines.append("")
            lines.append(_content_as_str(m.content))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _format_tool_call(tc: ToolCall) -> str:
    name = tc.get("name") or "?"
    args = tc.get("args") or {}
    try:
        args_str = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        args_str = str(args)
    # Inline for short args, pretty-printed for long/multi-line ones —
    # keeps the 80% case readable, the 20% case scannable.
    if len(args_str) <= 80 and "\n" not in args_str:
        return f"- `{name}({args_str})`"
    try:
        pretty = json.dumps(args, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        pretty = args_str
    return f"- `{name}`:\n\n```json\n{pretty}\n```"


def _content_as_str(content: object) -> str:
    """Flatten LangChain's union-typed ``content`` to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
                else:
                    chunks.append(json.dumps(part, ensure_ascii=False))
            else:
                chunks.append(str(part))
        return "".join(chunks)
    return str(content)


def _guess_lang(tool_name: str, body: str) -> str:
    """Best-effort language hint for fenced tool-result blocks."""
    name = tool_name.lower()
    if name in {"bash", "bash_background", "shell"}:
        return "bash"
    if name in {"read_file", "write_file", "edit_file"}:
        return ""
    stripped = body.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            json.loads(body)
        except (ValueError, TypeError):
            return ""
        return "json"
    return ""
