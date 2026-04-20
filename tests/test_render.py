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


def test_tool_call_started_shows_marker_and_args() -> None:
    r, buf = _capture()
    r.on_event(ToolCallStarted(name="read_file", input={"path": "/tmp/x"}))
    out = buf.getvalue()
    assert "◆" in out
    assert "read_file" in out
    assert "path" in out


def test_tool_call_completed_success_shows_checkmark() -> None:
    r, buf = _capture()
    r.on_event(ToolCallCompleted(name="read_file", output={"content": "x"}))
    assert "✓" in buf.getvalue()


def test_tool_call_completed_error_shows_cross_and_message() -> None:
    r, buf = _capture()
    r.on_event(ToolCallCompleted(name="bash", output=None, error="permission denied"))
    out = buf.getvalue()
    assert "✗" in out
    assert "permission denied" in out


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
    r, buf = _capture()
    long_value = "x" * 500
    r.on_event(ToolCallStarted(name="t", input={"data": long_value}))
    out = buf.getvalue()
    assert "…" in out
    assert len(out) < 300


def test_renderer_handles_multi_event_sequence() -> None:
    r, buf = _capture()
    r.on_event(AssistantDelta(text="let me check "))
    r.on_event(ToolCallStarted(name="read_file", input={"path": "/tmp/x"}))
    r.on_event(ToolCallCompleted(name="read_file", output={"content": "..."}))
    r.on_event(AssistantDelta(text="done"))
    r.on_event(Final(message="done"))
    r.finish()
    out = buf.getvalue()
    assert "let me check" in out
    assert "◆" in out
    assert "✓" in out
    assert "done" in out


def test_permission_audit_renders_with_indent_and_text() -> None:
    r, buf = _capture()
    r.on_event(PermissionAudit(tool="read_file", text="auto-allowed: read_only"))
    out = buf.getvalue()
    assert "auto-allowed: read_only" in out
    # 4-space indent per spec §8.4
    assert out.startswith("    ")


def test_permission_audit_rule_allow_text_flows_through() -> None:
    r, buf = _capture()
    r.on_event(PermissionAudit(tool="bash", text="auto-allowed: rule `bash(npm test)`"))
    out = buf.getvalue()
    assert "bash(npm test)" in out
    assert "auto-allowed: rule" in out


def test_permission_audit_appears_between_started_and_completed() -> None:
    r, buf = _capture()
    r.on_event(ToolCallStarted(name="read_file", input={"path": "/tmp/x"}))
    r.on_event(PermissionAudit(tool="read_file", text="auto-allowed: read_only"))
    r.on_event(ToolCallCompleted(name="read_file", output={"content": "x"}))
    out = buf.getvalue()
    started_pos = out.index("◆")
    audit_pos = out.index("auto-allowed")
    completed_pos = out.index("✓")
    assert started_pos < audit_pos < completed_pos
