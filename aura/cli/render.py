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

from aura.schemas.events import (
    AgentEvent,
    AssistantDelta,
    Final,
    PermissionAudit,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)
from aura.tools.errors import hint_for_error

# Fold threshold for search/read tool output. Below this, render as usual
# (no fold). At/above, the renderer shows head + "…" + tail + footer.
# 20 lines is a sweet spot: a typical grep hit for a targeted pattern fits
# below the fold; a broad codebase sweep (hundreds of matches) gets
# collapsed so the REPL doesn't flood.
_FOLD_THRESHOLD = 20
_FOLD_HEAD_LINES = 10
_FOLD_TAIL_LINES = 5


# Re-exported so existing callers / tests can continue to do
# ``from aura.cli.render import _hint_for_error``; the real table
# lives in ``aura.tools.errors`` to avoid a cli → tool-error layering
# inversion (loop.py needs it too, and must not import from cli).
def _hint_for_error(tool_name: str, error: str) -> str | None:
    return hint_for_error(tool_name, error)


# Markdown detection — pragmatic heuristic, not a full CommonMark parse.
# Matches the markers rich.Markdown actually styles differently from plain
# text: ATX headings, fenced code, list items, inline code, bold/italic,
# links, blockquotes. Multi-line content gets the benefit of the doubt
# (rich's paragraph reflow is harmless for prose; the win on code fences
# alone is worth it). One-liners with zero markers short-circuit to plain
# ``console.print`` to skip rich.Markdown's AST pass on trivial answers.
_MD_PATTERNS = (
    re.compile(r"^#{1,6}\s", re.MULTILINE),       # # heading
    re.compile(r"^\s*[-*+]\s", re.MULTILINE),     # - list / * list / + list
    re.compile(r"^\s*\d+\.\s", re.MULTILINE),     # 1. ordered list
    re.compile(r"^\s*>\s", re.MULTILINE),         # > blockquote
    re.compile(r"```"),                           # fenced code
    re.compile(r"`[^`\n]+`"),                     # inline code
    re.compile(r"\*\*[^*\n]+\*\*"),               # **bold**
    re.compile(r"(?<![*\w])\*[^*\n]+\*(?!\w)"),   # *italic*
    re.compile(r"\[[^\]\n]+\]\([^)\n]+\)"),       # [text](url) link
)


def _looks_like_markdown(text: str) -> bool:
    """Heuristic — does ``text`` carry any markdown markers worth rendering?

    Short-circuits on the first hit. A string with no markers at all goes
    through plain ``console.print``; anything with even one marker is
    routed to ``rich.markdown.Markdown`` where the full parser takes over.
    """
    return any(pat.search(text) for pat in _MD_PATTERNS)


class Renderer:
    def __init__(self, console: Console, *, markdown: bool = True) -> None:
        self._console = console
        self._markdown_enabled = markdown
        # Buffer for AssistantDelta text; flushed on the first non-delta
        # event (so tool-call output lands after prose) and on ``finish()``.
        # Streaming markdown chunk-by-chunk would break mid-fence rendering,
        # so we defer until the turn's text is complete before handing it
        # to rich.Markdown.
        self._pending_text = ""
        # Buffer for a ToolCallStarted whose line we HAVEN'T flushed yet.
        # Parity with claude-code's AssistantToolUseMessage / ToolUseLoader:
        # the leading glyph transitions ◆-while-running → ✓/✗-when-done on
        # the SAME line. Rich prints to scrollback (can't rewrite lines),
        # so the equivalent outcome is: defer the started line and, when
        # the completion event arrives *before* any progress chunk, emit a
        # single line with the final ✓/✗ glyph inline. If progress arrives
        # first we must flush the header so chunks nest under it; in that
        # case completion prints just the ✓/✗ summary under the header.
        # Stored as (name, input) — the args string is (re)rendered on emit.
        self._pending_tool: tuple[str, dict[str, Any]] | None = None

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, AssistantDelta):
            self._pending_text += event.text
            return
        # Any non-delta event means "the assistant paragraph is done, flush
        # it before rendering what comes next" — this preserves prose →
        # tool-call → prose ordering within a turn.
        self._flush_pending()
        if isinstance(event, ToolCallStarted):
            # Defer the header line (see `_pending_tool` docstring). The
            # glyph choice (◆ vs ✓ vs ✗) is decided on ToolCallCompleted.
            # A new ToolCallStarted before the previous one completed is
            # unexpected but we flush the old one with ◆ first (running
            # placeholder) to keep ordering honest.
            if self._pending_tool is not None:
                self._flush_pending_tool_as_running()
            self._pending_tool = (event.name, event.input)
            return
        if isinstance(event, PermissionAudit):
            # Audit line renders UNDER the tool-call header. Flush the
            # pending header first (as running ◆) so audit text lands
            # below the line it annotates rather than above a header
            # that's still buffered.
            if self._pending_tool is not None:
                self._flush_pending_tool_as_running()
            # Spec §8.4: dim one-liner, 4-space indent, directly after the
            # started line. Audit text carries rule strings that may contain
            # literal ``[...]`` — escape before wrapping in [dim].
            self._console.print(f"    [dim]{rich_escape(event.text)}[/dim]")
            return
        if isinstance(event, ToolCallProgress):
            # Progress chunks need the header printed first so they nest
            # visibly under it. Flush the buffered header (as running ◆)
            # once, then stream chunks below.
            if self._pending_tool is not None:
                self._flush_pending_tool_as_running()
            # Stream each chunk dim, with a ``│`` prefix so bash output
            # visually nests under its ToolCallStarted line rather than
            # masquerading as assistant prose. The spinner is already
            # stopped by the first tool event in this turn — progress
            # events ride on the same "some tool event fired" trigger
            # so nothing flickers.
            #
            # Trailing newlines stripped: rich's ``print`` adds its own,
            # so a newline-terminated chunk would otherwise emit a blank
            # row per line.
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
            # Consume a pending header (if any) INLINE with the final glyph.
            # Claude-code parity: the leading character of the tool-use line
            # carries the status (gray ● while running, green ● on success,
            # red ● on error) — the equivalent in scrollback is "one line,
            # final glyph". If progress already flushed the header, we
            # print just the completion summary under it.
            pending = self._pending_tool
            self._pending_tool = None
            if event.error:
                # Still need the tool-call header so the user sees WHICH
                # invocation failed. If the header was already flushed by
                # progress, don't reprint it; otherwise emit it inline with
                # the failure glyph.
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
                # Single inline line: glyph + tool(args) [+ summary].
                name, args = pending
                head = (
                    f"[green]✓[/green] [dim]{rich_escape(name)}"
                    f"({rich_escape(compact_args(args))})[/dim]"
                )
                if summary is not None:
                    self._console.print(f"{head} [dim]— {summary}[/dim]")
                else:
                    self._console.print(head)
            else:
                # Header was already flushed by a progress chunk — emit
                # only the completion glyph + summary to avoid duplicating
                # the tool name on-screen.
                if summary is not None:
                    self._console.print(f"[green]✓[/green] [dim]{summary}[/dim]")
                else:
                    self._console.print("[green]✓[/green]")
            # Fold step: only for tools whose metadata declares
            # ``is_search_command=True`` AND output line count >
            # _FOLD_THRESHOLD. Fold follows the ✓ summary so the user
            # still sees the one-line result signal; the folded body is
            # a dim aid below it.
            if event.name in _SEARCH_COMMAND_TOOLS:
                fold_text = _extract_text(event.output)
                if fold_text is not None:
                    total_lines = len(fold_text.splitlines())
                    if total_lines > _FOLD_THRESHOLD:
                        _render_folded(self._console, fold_text)
            return
        if isinstance(event, Final):
            # Final carries no new text under normal (natural) stops — the
            # body was already streamed via AssistantDelta and flushed
            # above.
            if self._pending_tool is not None:
                # Defensive: a Final arriving while a tool header is still
                # buffered (tool started but never completed — e.g. the
                # stream broke mid-call) — flush the header as running so
                # the transcript still shows the invocation.
                self._flush_pending_tool_as_running()
            # Round 3A — surface non-natural stop reasons inline. ``natural``
            # stops stay quiet (the body was the answer); ``max_turns`` and
            # ``aborted`` get a dim one-liner so the operator sees WHY the
            # turn ended without having to read the journal.
            reason = getattr(event, "reason", "natural")
            if reason == "aborted":
                self._console.print(Text(" cancelled by user", style="dim"))
            elif reason == "max_turns":
                self._console.print(Text(" max turns reached", style="dim"))
            return

    def finish(self) -> None:
        # Flush any trailing assistant text (e.g. turn ended without a
        # Final event in tests) before the terminal newline. A pending
        # tool header with no completion (cancelled stream, etc.) is
        # emitted as running ◆ so the user sees "something started here"
        # in the transcript.
        self._flush_pending()
        if self._pending_tool is not None:
            self._flush_pending_tool_as_running()
        self._console.print()

    def _flush_pending_tool_as_running(self) -> None:
        """Emit the buffered ToolCallStarted line with the running ◆ glyph.

        Called when a progress chunk / audit / early Final forces us to
        commit to the header BEFORE the completion event arrives. The
        glyph choice mirrors claude-code's ToolUseLoader "unresolved"
        state (gray BLACK_CIRCLE); ◆ is the closest Unicode diamond we
        already ship.
        """
        assert self._pending_tool is not None
        name, args = self._pending_tool
        self._pending_tool = None
        self._console.print(
            f"[dim]◆ {rich_escape(name)}"
            f"({rich_escape(compact_args(args))})[/dim]",
        )

    def _flush_pending(self) -> None:
        """Emit buffered assistant text as markdown-or-plain, then clear."""
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
    """Compact JSON preview of a params dict, truncated with ellipsis."""
    rendered = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) <= max_len:
        return rendered
    return rendered[:max_len] + "…"


# ---------------------------------------------------------------------------
# Per-tool result formatters. Each produces a short one-line summary attached
# after the ✓ on a successful ToolCallCompleted. Mirrors claude-code's
# ``renderToolResultMessage`` per-tool customisation. Formatters MUST be total
# over their documented input shape AND degrade gracefully on anything else —
# the renderer is on the hot path and cannot raise on a surprising dict.
# ---------------------------------------------------------------------------
def _format_read_file_result(output: Any) -> str:
    """read_file → "N lines, X.Y KB" or "N of M lines (partial)"."""
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
    """write_file → "N bytes written"."""
    # The in-tree WriteFile currently returns {"written": N}, but the
    # published contract documents {"bytes": N} — accept both so the
    # formatter survives either without flagging a false-negative.
    if isinstance(output, dict):
        if "bytes" in output:
            return f"{output['bytes']} bytes written"
        if "written" in output:
            return f"{output['written']} bytes written"
    return "written"


def _format_edit_file_result(output: Any) -> str:
    """edit_file → "N replacements" or "created (N line[s])"."""
    if isinstance(output, dict):
        reps = output.get("replacements", 0)
        if output.get("created"):
            suffix = "" if reps == 1 else "s"
            return f"created ({reps} line{suffix})"
        suffix = "" if reps == 1 else "s"
        return f"{reps} replacement{suffix}"
    return "edited"


def _format_grep_result(output: Any) -> str:
    """grep → mode-specific count: files / matches / total."""
    if isinstance(output, dict):
        mode = output.get("mode")
        trunc = " (truncated)" if output.get("truncated") else ""
        if mode == "files_with_matches":
            n = len(output.get("files", []))
            suffix = "" if n == 1 else "s"
            return f"{n} file{suffix}{trunc}"
        if mode == "content":
            n = len(output.get("matches", []))
            # "match" → "matches" is irregular; hand-plural.
            suffix = "" if n == 1 else "es"
            return f"{n} match{suffix}{trunc}"
        if mode == "count":
            total = output.get("total", 0)
            suffix = "" if total == 1 else "es"
            return f"{total} match{suffix}{trunc}"
    return "searched"


def _format_glob_result(output: Any) -> str:
    """glob → "N file[s]" [ (truncated)]."""
    if isinstance(output, dict):
        n = output.get("count", len(output.get("files", [])))
        suffix = "" if n == 1 else "s"
        trunc = " (truncated)" if output.get("truncated") else ""
        return f"{n} file{suffix}{trunc}"
    return "globbed"


def _format_bash_result(output: Any) -> str:
    """bash → "ok" / "output truncated" / "killed at 100 MB ceiling" + optional exit marker."""
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
    """task_create → "task <id8> — <description>"."""
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


# Tools whose output is search/read-like — the renderer folds long output
# (head + "…" + tail) so a 1000-line grep doesn't flood the REPL. Mirrors
# ``tool_metadata(is_search_command=True)`` on the tool object. Kept as a
# static set to avoid a render → registry dependency (same pattern as
# ``_TOOL_RESULT_FORMATTERS`` above); tests assert the set matches the
# actual metadata on each tool so drift is caught.
_SEARCH_COMMAND_TOOLS: frozenset[str] = frozenset({
    "grep", "glob", "read_file", "web_search",
})


def _extract_text(output: Any) -> str | None:
    """Pull a renderable string out of a tool's output, if any.

    Search tools return dicts with the interesting text under known keys
    (``matches`` for grep content mode, ``files`` for files_with_matches /
    glob, ``content`` for read_file, ``results`` for web_search). Returns
    ``None`` when there's nothing text-shaped to fold — the caller falls
    back to the standard ✓-summary path.
    """
    if isinstance(output, str):
        return output
    if not isinstance(output, dict):
        return None
    # grep content mode: matches is a list of {path, line, text, ...}
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
    # grep files_with_matches / glob: list of paths.
    files = output.get("files")
    if isinstance(files, list) and files:
        return "\n".join(str(f) for f in files)
    # read_file: ``content`` string.
    content = output.get("content")
    if isinstance(content, str) and content:
        return content
    # web_search: ``results`` is a list of {title, url, snippet}.
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
    """Render ``text`` in fold mode: head + ``…`` + tail + footer.

    Only called when the caller has already verified fold applies (search
    tool + line count above threshold). Keeps the fold purely a rendering
    concern — no truncation of the underlying tool output or model-facing
    history.
    """
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


# ---------------------------------------------------------------------------
# Error rendering — boxed panel + actionable hints. Replaces the previous
# `✗ message` one-liner so operators see WHAT failed and a concrete next
# step without hunting the transcript for context.
#
# Hint lookup lives in ``aura.tools.errors`` (imported as ``_hint_for_error``
# above) — shared with the loop's ToolMessage builder so the MODEL sees the
# same guidance the user does. Don't inline the table here.
# ---------------------------------------------------------------------------


def _render_tool_error(tool_name: str, error: str) -> Panel:
    """Build a red-bordered panel with the error + optional hint.

    Panel is chosen over a single dim line because tool errors are the
    LLM's actionable feedback; giving them visual weight encourages the
    model to read + correct rather than retry blindly."""
    body = Text(error, style="red")
    hint = _hint_for_error(tool_name, error)
    if hint is not None:
        body.append("\n\n")
        body.append(hint, style="dim")
    return Panel(body, title=f"[red]{tool_name} failed[/red]", border_style="red")
