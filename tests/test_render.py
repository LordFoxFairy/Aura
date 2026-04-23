"""Tests for aura.cli.render.Renderer."""

from __future__ import annotations

import io

from rich.console import Console

from aura.cli.render import Renderer
from aura.schemas.events import (
    AssistantDelta,
    Final,
    PermissionAudit,
    ToolCallCompleted,
    ToolCallStarted,
)


def _capture() -> tuple[Renderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200, highlight=False)
    return Renderer(console), buf


def test_assistant_delta_renders_markdown_in_final_buffer() -> None:
    r, buf = _capture()
    r.on_event(AssistantDelta(text="**hello**"))
    r.finish()
    out = buf.getvalue()
    assert "hello" in out
    assert "**" not in out


def test_assistant_delta_plain_text_appears() -> None:
    r, buf = _capture()
    r.on_event(AssistantDelta(text="hello world"))
    r.finish()
    assert "hello world" in buf.getvalue()


def test_tool_call_started_alone_does_not_emit_until_completion() -> None:
    # U3 parity with claude-code: the tool-use header line carries the
    # FINAL glyph (✓/✗) inline, so the started line is buffered until
    # completion (or a progress chunk / Final / new started / audit /
    # finish forces a flush). A bare ToolCallStarted should produce no
    # scrollback yet.
    r, buf = _capture()
    r.on_event(ToolCallStarted(name="read_file", input={"path": "/tmp/x"}))
    assert buf.getvalue() == ""


def test_tool_call_started_plus_completed_emits_single_inline_line() -> None:
    # The common path: a tool runs to completion without progress chunks.
    # Renderer emits ONE line — ``✓ name(args) — summary`` — not two.
    r, buf = _capture()
    r.on_event(ToolCallStarted(name="read_file", input={"path": "/tmp/x"}))
    r.on_event(ToolCallCompleted(
        name="read_file",
        output={
            "content": "x", "lines": 1, "total_lines": 1,
            "offset": 0, "limit": None, "partial": False,
        },
    ))
    out = buf.getvalue()
    assert "✓" in out
    assert "read_file" in out
    assert "path" in out
    # Exactly one non-empty line of output — no stray ``✓`` on its own.
    non_empty = [ln for ln in out.splitlines() if ln.strip()]
    assert len(non_empty) == 1
    # And the ◆ running marker shouldn't appear — completion replaced it.
    assert "◆" not in out


def test_tool_call_completed_success_shows_checkmark() -> None:
    # Without a preceding Started (e.g. fake/partial stream), a bare
    # completed event still prints a ✓ line — preserves the old contract
    # for callers that don't emit a matched pair.
    r, buf = _capture()
    r.on_event(ToolCallCompleted(name="read_file", output={"content": "x"}))
    assert "✓" in buf.getvalue()


def test_tool_call_error_inline_glyph_precedes_error_panel() -> None:
    # U3 parity: on failure, the ✗ glyph + tool(args) go inline on a
    # header line and the error panel renders UNDER it. No orphan header
    # with a ◆, no separate ✓.
    r, buf = _capture()
    r.on_event(ToolCallStarted(name="bash", input={"command": "oops"}))
    r.on_event(ToolCallCompleted(name="bash", output=None, error="permission denied"))
    out = buf.getvalue()
    assert "✗" in out
    assert "bash" in out
    assert "oops" in out
    # Error panel borders show up below.
    assert "╭" in out or "┌" in out
    assert "◆" not in out
    assert "✓" not in out


def test_tool_call_completed_after_progress_chunk_omits_duplicated_header() -> None:
    # If a progress chunk flushed the ◆ header, completion should NOT
    # reprint the tool name — that would produce a visual duplicate.
    # Only the ✓ summary prints under the progress stream.
    from aura.schemas.events import ToolCallProgress

    r, buf = _capture()
    r.on_event(ToolCallStarted(name="bash", input={"command": "ls /"}))
    r.on_event(ToolCallProgress(name="bash", stream="stdout", chunk="bin\netc\n"))
    r.on_event(ToolCallCompleted(
        name="bash",
        output={
            "stdout": "bin\netc\n", "stderr": "", "exit_code": 0,
            "truncated": False, "killed_at_hard_ceiling": False,
        },
    ))
    out = buf.getvalue()
    # Header printed once — exactly one ◆ line for "bash(ls /)".
    assert out.count("◆") == 1
    # Progress lines nested under it.
    assert "bin" in out
    assert "etc" in out
    # Completion summary visible but NOT re-naming the tool.
    assert "✓" in out
    assert out.count("bash") == 1  # only on the ◆ header — not re-named
    assert "ok" in out             # bash-formatter summary


def test_tool_call_lone_checkmark_line_never_emitted() -> None:
    # Guard the specific user-reported pattern: "◆ tool(args)" followed
    # by a lone line with just "✓". Renderer must NEVER produce that.
    r, buf = _capture()
    r.on_event(ToolCallStarted(name="todo_write", input={"todos": []}))
    r.on_event(ToolCallCompleted(name="todo_write", output={"ok": True}))
    out = buf.getvalue()
    # Split and strip ANSI — look for any line that is JUST the check
    # with nothing else. Rich may wrap in escape codes; compare raw runes.
    import re
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    lines = [ansi_re.sub("", ln).strip() for ln in out.splitlines() if ln.strip()]
    assert not any(ln == "✓" for ln in lines), (
        f"lone '✓' line regression: {lines!r}"
    )


def test_tool_call_completed_error_shows_panel_with_tool_name_and_message() -> None:
    # As of CLI polish, the renderer uses a boxed rich Panel for tool errors
    # (see aura/cli/render.py::_render_tool_error) — no ✗ anymore. The panel
    # includes the tool name in its title + the error message in the body.
    r, buf = _capture()
    r.on_event(ToolCallCompleted(name="bash", output=None, error="permission denied"))
    out = buf.getvalue()
    assert "bash" in out
    assert "permission denied" in out
    # A rich Panel draws corner glyphs; one of them shows up.
    assert "╭" in out or "┌" in out


def test_final_closes_live_silently() -> None:
    r, buf = _capture()
    r.on_event(AssistantDelta(text="some text"))
    before = buf.getvalue()
    r.on_event(Final(message="done"))
    after = buf.getvalue()
    assert len(after) >= len(before)


def test_finish_emits_trailing_newline() -> None:
    r, buf = _capture()
    r.finish()
    assert buf.getvalue().endswith("\n")


def test_long_tool_args_are_truncated() -> None:
    # Args truncation applies wherever the tool line is flushed — whether
    # via a matched Completed inline emit or via finish() forcing the
    # pending header to print with ◆. Exercise the Completed path since
    # that's the common dogfood case.
    r, buf = _capture()
    long_value = "x" * 500
    r.on_event(ToolCallStarted(name="t", input={"data": long_value}))
    r.on_event(ToolCallCompleted(name="t", output={"ok": True}))
    out = buf.getvalue()
    assert "…" in out
    assert len(out) < 300


def test_renderer_handles_multi_event_sequence() -> None:
    r, buf = _capture()
    r.on_event(AssistantDelta(text="let me check "))
    r.on_event(ToolCallStarted(name="read_file", input={"path": "/tmp/x"}))
    r.on_event(ToolCallCompleted(
        name="read_file",
        output={
            "content": "...", "lines": 1, "total_lines": 1,
            "offset": 0, "limit": None, "partial": False,
        },
    ))
    r.on_event(AssistantDelta(text="done"))
    r.on_event(Final(message="done"))
    r.finish()
    out = buf.getvalue()
    assert "let me check" in out
    # Post-U3: no ◆ when the matched Started/Completed pair is emitted
    # inline with the final ✓ glyph.
    assert "◆" not in out
    assert "✓" in out
    assert "read_file" in out
    assert "done" in out


def test_permission_audit_renders_with_indent_and_text() -> None:
    r, buf = _capture()
    r.on_event(PermissionAudit(tool="read_file", text="auto-allowed: rule `read_file`"))
    out = buf.getvalue()
    assert "auto-allowed: rule `read_file`" in out
    # 4-space indent per spec §8.4
    assert out.startswith("    ")


def test_permission_audit_rule_allow_text_flows_through() -> None:
    r, buf = _capture()
    r.on_event(PermissionAudit(tool="bash", text="auto-allowed: rule `bash(npm test)`"))
    out = buf.getvalue()
    assert "bash(npm test)" in out
    assert "auto-allowed: rule" in out


def test_permission_audit_appears_between_started_and_completed() -> None:
    # Audit arriving mid-flight forces the Started header to flush as a
    # running ◆ — otherwise the audit text would land above a header
    # that's still buffered. Then the Completed line prints the ✓
    # summary line under the audit. Three lines in order: ◆ → audit → ✓.
    r, buf = _capture()
    r.on_event(ToolCallStarted(name="read_file", input={"path": "/tmp/x"}))
    r.on_event(PermissionAudit(tool="read_file", text="auto-allowed: rule `read_file`"))
    r.on_event(ToolCallCompleted(
        name="read_file",
        output={
            "content": "x", "lines": 1, "total_lines": 1,
            "offset": 0, "limit": None, "partial": False,
        },
    ))
    out = buf.getvalue()
    started_pos = out.index("◆")
    audit_pos = out.index("auto-allowed")
    completed_pos = out.index("✓")
    assert started_pos < audit_pos < completed_pos


def test_renderer_escapes_bracket_markup_in_tool_input() -> None:
    # LLM-chosen input might contain ``[...]`` which rich would otherwise
    # parse as inline markup. Renderer must escape before wrapping — both
    # on the buffered-then-flushed header path AND the inline success path.
    # Exercise the header path via Finish (flushes pending as ◆).
    r, buf = _capture()
    r.on_event(ToolCallStarted(name="bash", input={"command": "echo [red]HI[/red]"}))
    r.finish()
    out = buf.getvalue()
    # Literal ``[red]`` must appear in output as text (escaped).
    assert "[red]" in out


def test_renderer_escapes_bracket_markup_in_completion_error() -> None:
    r, buf = _capture()
    r.on_event(ToolCallCompleted(name="x", output=None, error="oops [bold]trap"))
    out = buf.getvalue()
    assert "[bold]" in out


def test_renderer_escapes_bracket_markup_in_audit_text() -> None:
    r, buf = _capture()
    r.on_event(PermissionAudit(tool="x", text="auto-allowed: rule `x([weird])`"))
    out = buf.getvalue()
    assert "[weird]" in out


# --------------------------------------------------------------------------
# Per-tool formatted summaries on ToolCallCompleted (spec §render polish).
# --------------------------------------------------------------------------
def test_tool_call_completed_read_file_renders_formatted_summary() -> None:
    r, buf = _capture()
    r.on_event(ToolCallCompleted(
        name="read_file",
        output={
            "content": "x" * 4300,
            "lines": 142,
            "total_lines": 142,
            "offset": 0,
            "limit": None,
            "partial": False,
        },
    ))
    out = buf.getvalue()
    assert "✓" in out
    assert "142 lines" in out
    assert "KB" in out


def test_tool_call_completed_error_still_uses_error_path() -> None:
    # Error path should NOT consult the formatter registry, even for a known tool.
    r, buf = _capture()
    r.on_event(ToolCallCompleted(
        name="read_file", output=None, error="ENOENT: no such file",
    ))
    out = buf.getvalue()
    # Panel renders tool name in title + error in body; formatter summary
    # ("42 lines, 1.5 KB") must NOT leak into the error output.
    assert "read_file" in out
    assert "ENOENT" in out
    assert "lines" not in out
    assert "KB" not in out


def test_tool_call_completed_unknown_tool_uses_generic_fallback() -> None:
    r, buf = _capture()
    r.on_event(ToolCallCompleted(name="not_a_real_tool", output={"x": 1}))
    out = buf.getvalue()
    assert "✓" in out
    # No per-tool formatter ran, so no structured summary text appears.
    assert "lines" not in out


def test_tool_call_completed_bash_shows_exit_marker_on_failure() -> None:
    r, buf = _capture()
    r.on_event(ToolCallCompleted(
        name="bash",
        output={
            "stdout": "", "stderr": "bad", "exit_code": 2,
            "truncated": False, "killed_at_hard_ceiling": False,
        },
    ))
    out = buf.getvalue()
    assert "✓" in out
    assert "exit 2" in out


# --------------------------------------------------------------------------
# Markdown rendering path (buffer-on-delta, flush-on-boundary).
#
# Streaming chunk-by-chunk through ``rich.Markdown`` would break mid-fence
# rendering, so the renderer buffers AssistantDelta text and only flushes
# at turn boundaries (any non-delta event) or ``finish()``. Tests below
# cover the markdown/plain branch, empty/whitespace short-circuit, the
# UIConfig toggle, and the ordering guarantee (prose flushes BEFORE the
# tool line that triggered the flush).
# --------------------------------------------------------------------------
def test_assistant_delta_not_emitted_incrementally() -> None:
    r, buf = _capture()
    r.on_event(AssistantDelta(text="chunk one "))
    # Before flush, nothing should be on the wire — avoids duplicate
    # output when the final markdown block renders.
    assert buf.getvalue() == ""
    r.on_event(AssistantDelta(text="chunk two"))
    assert buf.getvalue() == ""
    r.finish()
    out = buf.getvalue()
    assert "chunk one" in out
    assert "chunk two" in out


def test_assistant_delta_with_heading_renders_as_markdown() -> None:
    r, buf = _capture()
    r.on_event(AssistantDelta(text="# Title\n\nBody text."))
    r.finish()
    out = buf.getvalue()
    # Markdown strips the raw ``#`` marker (renders as a styled heading);
    # the literal ``# Title`` text should NOT appear.
    assert "# Title" not in out
    assert "Title" in out
    assert "Body text" in out


def test_assistant_delta_with_code_fence_renders_as_markdown() -> None:
    r, buf = _capture()
    r.on_event(AssistantDelta(text="```python\nprint('hi')\n```"))
    r.finish()
    out = buf.getvalue()
    # Code-fence content survives; fence markers themselves are consumed.
    assert "print" in out
    assert "```" not in out


def test_looks_like_markdown_heuristic_matrix() -> None:
    # Direct unit test for the marker heuristic. One positive case per
    # supported syntax + confirmation that marker-free prose does NOT
    # route through rich.Markdown.
    from aura.cli.render import _looks_like_markdown

    assert _looks_like_markdown("# heading") is True
    assert _looks_like_markdown("## sub heading") is True
    assert _looks_like_markdown("- bullet") is True
    assert _looks_like_markdown("1. item") is True
    assert _looks_like_markdown("> quoted") is True
    assert _looks_like_markdown("```python\nx\n```") is True
    assert _looks_like_markdown("use `foo` here") is True
    assert _looks_like_markdown("**bold**") is True
    assert _looks_like_markdown("see [here](http://x)") is True
    # Negative cases — plain prose must short-circuit.
    assert _looks_like_markdown("hello world") is False
    assert _looks_like_markdown("42") is False
    assert _looks_like_markdown("a sentence with no markup at all.") is False


def test_assistant_delta_plain_text_skips_markdown_wrapping() -> None:
    # Marker-free content is routed through plain ``console.print``,
    # which emits the string verbatim with a trailing newline. The output
    # width is NOT reflowed by Markdown's block layout — so a short
    # string stays short (no trailing padding from rich.Markdown's
    # paragraph renderer).
    r, buf = _capture()
    r.on_event(AssistantDelta(text="hello world"))
    r.finish()
    out = buf.getvalue()
    assert "hello world" in out
    # Plain-print of a short string is <= a handful of bytes + one \n.
    # rich.Markdown would emit a paragraph block (multiple spaces of
    # right-padding, an extra blank line). Guarding on byte count is
    # the cleanest way to prove the plain branch ran.
    assert len(out) < 20


def test_assistant_delta_markdown_content_renders_via_markdown() -> None:
    # Symmetric check: content WITH markers routes through
    # rich.Markdown — the ``**bold**`` markers get consumed by the
    # parser rather than appearing verbatim in the output.
    r, buf = _capture()
    r.on_event(AssistantDelta(text="**bold** statement"))
    r.finish()
    out = buf.getvalue()
    assert "bold" in out
    assert "statement" in out
    # rich.Markdown strips the asterisks (renders as styled text).
    assert "**bold**" not in out


def test_assistant_delta_empty_content_emits_no_output() -> None:
    r, buf = _capture()
    r.on_event(AssistantDelta(text=""))
    # finish() adds one trailing newline — that's the ONLY output.
    r.finish()
    assert buf.getvalue() == "\n"


def test_assistant_delta_whitespace_only_content_emits_no_output() -> None:
    r, buf = _capture()
    r.on_event(AssistantDelta(text="   \n\t\n  "))
    r.finish()
    assert buf.getvalue() == "\n"


def test_assistant_delta_flushed_before_tool_call_started() -> None:
    # Ordering guarantee: prose must land on screen BEFORE the tool-call
    # line. Post-U3 the ToolCallStarted header is buffered until
    # completion; force it out with finish() and check ordering with the
    # ◆-running glyph (no matched Completed in this test).
    r, buf = _capture()
    r.on_event(AssistantDelta(text="let me check"))
    r.on_event(ToolCallStarted(name="read_file", input={"path": "/x"}))
    r.finish()
    out = buf.getvalue()
    prose_pos = out.index("let me check")
    tool_pos = out.index("◆")
    assert prose_pos < tool_pos


def test_renderer_markdown_toggle_disables_markdown_path() -> None:
    # UIConfig.ui.markdown=False → Renderer(markdown=False) → even
    # marker-rich content goes through plain ``console.print``, so raw
    # markdown markers survive in the output.
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200, highlight=False)
    r = Renderer(console, markdown=False)
    r.on_event(AssistantDelta(text="# Heading\n\n**bold**"))
    r.finish()
    out = buf.getvalue()
    assert "# Heading" in out
    assert "**bold**" in out


def test_finish_flushes_pending_assistant_text() -> None:
    # A turn that ends without a Final event (test harness, or an
    # astream that yields only AssistantDelta) must still surface the
    # buffered text on finish() — otherwise it would be silently dropped.
    r, buf = _capture()
    r.on_event(AssistantDelta(text="trailing text"))
    r.finish()
    assert "trailing text" in buf.getvalue()


def test_final_event_flushes_pending_text() -> None:
    r, buf = _capture()
    r.on_event(AssistantDelta(text="pending"))
    r.on_event(Final(message="pending"))
    # Flush happens on Final, before finish() adds its trailing newline.
    assert "pending" in buf.getvalue()
