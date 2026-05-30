"""Rich renderer for AgentEvent instances."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.text import Text

from aura.domain.events import (
    AgentEvent,
    AssistantDelta,
    PermissionAudit,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)
from aura.tools.errors import hint_for_error

_FOLD_THRESHOLD = 20
_FOLD_HEAD_LINES = 10
_FOLD_TAIL_LINES = 5


def _hint_for_error(tool_name: str, error: str) -> str | None:
    return hint_for_error(tool_name, error)


_MD_PATTERNS = (
    re.compile(r"^#{1,6}\s", re.MULTILINE),
    re.compile(r"^\s*[-*+]\s", re.MULTILINE),
    re.compile(r"^\s*\d+\.\s", re.MULTILINE),
    re.compile(r"^\s*>\s", re.MULTILINE),
    re.compile(r"```"),
    re.compile(r"`[^`\n]+`"),
    re.compile(r"\*\*[^*\n]+\*\*"),
    re.compile(r"(?<![*\w])\*[^*\n]+\*(?!\w)"),
    re.compile(r"\[[^\]\n]+\]\([^)\n]+\)"),
)


def _looks_like_markdown(text: str) -> bool:
    return any(pat.search(text) for pat in _MD_PATTERNS)


class Renderer:
    def __init__(self, console: Console, *, markdown: bool = True) -> None:
        self._console = console
        self._markdown_enabled = markdown
        self._pending_text = ""
        # Defer the ToolCallStarted line so the leading glyph can be set to
        # the final ✓/✗ when completion arrives before any progress chunk.
        self._pending_tool: tuple[str, dict[str, Any]] | None = None

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, AssistantDelta):
            self._pending_text += event.text
            return
        self._flush_pending()
        if isinstance(event, ToolCallStarted):
            if self._pending_tool is not None:
                self._flush_pending_tool_as_running()
            self._pending_tool = (event.name, event.input)
            return
        if isinstance(event, PermissionAudit):
            if self._pending_tool is not None:
                self._flush_pending_tool_as_running()
            self._console.print(f"    [dim]{rich_escape(event.text)}[/dim]")
            return
        if isinstance(event, ToolCallProgress):
            if self._pending_tool is not None:
                self._flush_pending_tool_as_running()
            text = event.chunk.rstrip("\n")
            if not text:
                return
            for line in text.split("\n"):
                self._console.print(
                    f"[dim]│ {rich_escape(line)}[/dim]",
                    highlight=False,
                )
            return
        if isinstance(event, ToolCallCompleted):
            pending = self._pending_tool
            self._pending_tool = None
            if event.error:
                if pending is not None:
                    name, args = pending
                    self._console.print(
                        f"[red]✗[/red] [dim]{rich_escape(name)}"
                        f"({rich_escape(compact_args(args))})[/dim]",
                    )
                self._console.print(_render_tool_error(event.name, event.error))
                return
            formatter = _TOOL_RESULT_FORMATTERS.get(event.name)
            summary = (
                rich_escape(formatter(event.output))
                if formatter is not None else None
            )
            if pending is not None:
                name, args = pending
                head = (
                    f"[green]✓[/green] [dim]{rich_escape(name)}"
                    f"({rich_escape(compact_args(args))})[/dim]"
                )
                if summary is not None:
                    self._console.print(f"{head} [dim]— {summary}[/dim]")
                else:
                    self._console.print(head)
            elif summary is not None:
                self._console.print(f"[green]✓[/green] [dim]{summary}[/dim]")
            else:
                self._console.print("[green]✓[/green]")
            if event.name in _SEARCH_COMMAND_TOOLS:
                fold_text = _extract_text(event.output)
                if fold_text is not None:
                    total_lines = len(fold_text.splitlines())
                    if total_lines > _FOLD_THRESHOLD:
                        _render_folded(self._console, fold_text)
            return
        if self._pending_tool is not None:
            self._flush_pending_tool_as_running()
        reason = event.reason
        if reason == "aborted":
            self._console.print(Text(" cancelled by user", style="dim"))
        elif reason == "max_turns":
            self._console.print(Text(" max turns reached", style="dim"))
        elif reason == "length_recovery_exhausted":
            self._console.print(Text(
                " ⚠ output truncated by max_output_tokens after 3 retries"
                " — try /retry",
                style="yellow",
            ))

    def finish(self) -> None:
        self._flush_pending()
        if self._pending_tool is not None:
            self._flush_pending_tool_as_running()
        self._console.print()

    def _flush_pending_tool_as_running(self) -> None:
        assert self._pending_tool is not None
        name, args = self._pending_tool
        self._pending_tool = None
        self._console.print(
            f"[dim]◆ {rich_escape(name)}"
            f"({rich_escape(compact_args(args))})[/dim]",
        )

    def _flush_pending(self) -> None:
        text = self._pending_text
        self._pending_text = ""
        if not text or not text.strip():
            return
        if self._markdown_enabled and _looks_like_markdown(text):
            self._console.print(
                Markdown(text, code_theme="monokai", inline_code_lexer="python"),
            )
        else:
            self._console.print(text)


def compact_args(args: dict[str, Any], *, max_len: int = 80) -> str:
    rendered = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) <= max_len:
        return rendered
    return rendered[:max_len] + "…"


def _format_read_file_result(output: Any) -> str:
    if not isinstance(output, dict):
        return str(output)
    lines = output.get("lines", 0)
    total = output.get("total_lines", 0)
    if output.get("partial"):
        return f"{lines} of {total} lines (partial)"
    content = output.get("content", "")
    size_kb = len(content) / 1024 if isinstance(content, str) else 0.0
    return f"{lines} lines, {size_kb:.1f} KB"


def _format_write_file_result(output: Any) -> str:
    # Accept both {"bytes": N} (contract) and {"written": N} (current impl).
    if isinstance(output, dict):
        if "bytes" in output:
            return f"{output['bytes']} bytes written"
        if "written" in output:
            return f"{output['written']} bytes written"
    return "written"


def _format_edit_file_result(output: Any) -> str:
    if isinstance(output, dict):
        reps = output.get("replacements", 0)
        if output.get("created"):
            suffix = "" if reps == 1 else "s"
            return f"created ({reps} line{suffix})"
        suffix = "" if reps == 1 else "s"
        return f"{reps} replacement{suffix}"
    return "edited"


def _format_grep_result(output: Any) -> str:
    if isinstance(output, dict):
        mode = output.get("mode")
        trunc = " (truncated)" if output.get("truncated") else ""
        if mode == "files_with_matches":
            n = len(output.get("files", []))
            suffix = "" if n == 1 else "s"
            return f"{n} file{suffix}{trunc}"
        if mode == "content":
            n = len(output.get("matches", []))
            suffix = "" if n == 1 else "es"
            return f"{n} match{suffix}{trunc}"
        if mode == "count":
            total = output.get("total", 0)
            suffix = "" if total == 1 else "es"
            return f"{total} match{suffix}{trunc}"
    return "searched"


def _format_glob_result(output: Any) -> str:
    if isinstance(output, dict):
        n = output.get("count", len(output.get("files", [])))
        suffix = "" if n == 1 else "s"
        trunc = " (truncated)" if output.get("truncated") else ""
        return f"{n} file{suffix}{trunc}"
    return "globbed"


def _format_bash_result(output: Any) -> str:
    if isinstance(output, dict):
        code = output.get("exit_code", 0)
        marker = "" if code == 0 else f" (exit {code})"
        if output.get("killed_at_hard_ceiling"):
            return f"killed at 100 MB ceiling{marker}"
        if output.get("truncated"):
            return f"output truncated{marker}"
        return f"ok{marker}"
    return "executed"


def _format_task_create_result(output: Any) -> str:
    if isinstance(output, dict):
        tid = str(output.get("task_id", "?"))
        desc = output.get("description", "")
        return f"task {tid[:8]} — {desc}"
    return "spawned"


_TOOL_RESULT_FORMATTERS: dict[str, Callable[[Any], str]] = {
    "read_file": _format_read_file_result,
    "write_file": _format_write_file_result,
    "edit_file": _format_edit_file_result,
    "grep": _format_grep_result,
    "glob": _format_glob_result,
    "bash": _format_bash_result,
    "task_create": _format_task_create_result,
}


_SEARCH_COMMAND_TOOLS: frozenset[str] = frozenset({
    "grep", "glob", "read_file", "web_search",
})


def _extract_text(output: Any) -> str | None:
    if isinstance(output, str):
        return output
    if not isinstance(output, dict):
        return None
    matches = output.get("matches")
    if isinstance(matches, list) and matches:
        lines: list[str] = []
        for m in matches:
            if isinstance(m, dict):
                path = m.get("path", "")
                line_no = m.get("line", "")
                text = m.get("text", "")
                lines.append(f"{path}:{line_no}:{text}")
            else:
                lines.append(str(m))
        return "\n".join(lines)
    files = output.get("files")
    if isinstance(files, list) and files:
        return "\n".join(str(f) for f in files)
    content = output.get("content")
    if isinstance(content, str) and content:
        return content
    results = output.get("results")
    if isinstance(results, list) and results:
        lines = []
        for r in results:
            if isinstance(r, dict):
                title = r.get("title", "")
                url = r.get("url", "")
                lines.append(f"{title} — {url}")
            else:
                lines.append(str(r))
        return "\n".join(lines)
    return None


def _render_folded(console: Console, text: str) -> None:
    lines = text.splitlines()
    total = len(lines)
    head = lines[:_FOLD_HEAD_LINES]
    tail = lines[-_FOLD_TAIL_LINES:]
    for line in head:
        console.print(f"[dim]│ {rich_escape(line)}[/dim]", highlight=False)
    console.print("[dim]…[/dim]", highlight=False)
    for line in tail:
        console.print(f"[dim]│ {rich_escape(line)}[/dim]", highlight=False)
    shown = len(head) + len(tail)
    console.print(
        f"[dim][{total} lines total, {shown} shown][/dim]",
        highlight=False,
    )


def _render_tool_error(tool_name: str, error: str) -> Panel:
    body = Text(error, style="red")
    hint = _hint_for_error(tool_name, error)
    if hint is not None:
        body.append("\n\n")
        body.append(hint, style="dim")
    return Panel(body, title=f"[red]{tool_name} failed[/red]", border_style="red")
