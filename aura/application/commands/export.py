"""``/export`` dump transcript (``md``/``json``); reads ``storage.load``; IO errors -> result."""

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

from aura.application.commands.types import CommandResult, CommandSource
from aura.infrastructure.persistence import journal

if TYPE_CHECKING:
    from aura.core.agent import Agent


Format = Literal["md", "json"]

_DEFAULT_DIR = Path("~/.aura/exports")
_MD_EXTS = frozenset({".md", ".markdown"})
_JSON_EXTS = frozenset({".json"})


class ExportCommand:
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

        messages = agent.storage.load(agent.session_id)
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


def _parse_args(arg: str) -> tuple[str | None, Format | None]:
    """Split ``arg`` into (path, format); ``--format`` may appear before or after the path."""
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
            fmt = value  # type: ignore[assignment]  # narrowing branch mypy doesn't track
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
    """Resolve ``(path, format, note)``; ``note`` is an extension-fallback hint or ``""``."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    note = ""

    if path_arg is None:
        fmt: Format = fmt_arg or "md"
        ext = ".json" if fmt == "json" else ".md"
        path = _DEFAULT_DIR.expanduser() / f"aura-session-{timestamp}{ext}"
        return path, fmt, note

    path = Path(path_arg).expanduser()
    if path.is_dir() or path_arg.endswith(("/", "\\")):
        fmt = fmt_arg or "md"
        ext = ".json" if fmt == "json" else ".md"
        return path / f"aura-session-{timestamp}{ext}", fmt, note

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
            note = f"note: unknown extension {suffix!r}, writing markdown"
    return path, fmt, note


def _envelope(agent: Agent, messages: list[BaseMessage]) -> dict[str, object]:
    turns = sum(1 for m in messages if isinstance(m, HumanMessage))
    return {
        "session_id": agent.session_id,
        "exported_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": agent.current_model,
        "cwd": str(agent.cwd),
        "turns": turns,
        "total_tokens": agent.state.total_tokens_used,
    }


def _render_json(agent: Agent, messages: list[BaseMessage]) -> str:
    msg_dicts: list[dict[str, object]] = []
    for m in messages:
        entry: dict[str, object] = {
            "role": m.type, "content": _content_as_str(m.content),
        }
        if isinstance(m, AIMessage) and m.tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("name", ""), "args": tc.get("args", {}), "id": tc.get("id", "")}
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
        "# Aura session export", "",
        f"- session_id: {env['session_id']}",
        f"- timestamp: {env['exported_at']}",
        f"- model: {env['model']}",
        f"- cwd: {env['cwd']}",
        f"- turns: {env['turns']}",
        f"- total tokens: {env['total_tokens']:,}",
        "", "---", "",
    ]

    turn_no = 0
    for m in messages:
        if isinstance(m, HumanMessage):
            turn_no += 1
            lines += [f"## Turn {turn_no} (user)", "", _content_as_str(m.content), ""]
        elif isinstance(m, AIMessage):
            # Post-compact assistant-before-first-human groups under turn 0 instead of raising.
            lines += [f"## Turn {turn_no} (assistant)", ""]
            content = _content_as_str(m.content).strip()
            if content:
                lines += [content, ""]
            if m.tool_calls:
                lines.append("### Tool calls")
                lines += [_format_tool_call(tc) for tc in m.tool_calls]
                lines.append("")
        elif isinstance(m, ToolMessage):
            tool_name = m.name or "tool"
            body = _content_as_str(m.content)
            lines += [
                f"## Turn {turn_no} (tool: {tool_name})", "",
                "```" + _guess_lang(tool_name, body),
                body, "```", "",
            ]
        else:
            # SystemMessage or future subclass — verbatim so the export is lossless.
            lines += [f"## Turn {turn_no} ({m.type})", "", _content_as_str(m.content), ""]

    return "\n".join(lines).rstrip() + "\n"


def _format_tool_call(tc: ToolCall) -> str:
    """Inline short args, fenced block for long ones."""
    name = tc.get("name") or "?"
    args = tc.get("args") or {}
    try:
        args_str = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        args_str = str(args)
    if len(args_str) <= 80 and "\n" not in args_str:
        return f"- `{name}({args_str})`"
    try:
        pretty = json.dumps(args, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        pretty = args_str
    return f"- `{name}`:\n\n```json\n{pretty}\n```"


def _content_as_str(content: object) -> str:
    """Flatten LangChain's union-typed ``content`` to a string."""
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
    if body.lstrip().startswith(("{", "[")):
        try:
            json.loads(body)
        except (ValueError, TypeError):
            return ""
        return "json"
    return ""
