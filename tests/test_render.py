"""Tests for aura.cli.render.Renderer."""

from __future__ import annotations

import io

from rich.console import Console

from aura.cli.render import Renderer
from aura.core.events import (
    AssistantDelta,
    Final,
    ToolCallCompleted,
    ToolCallStarted,
)


def _capture() -> tuple[Renderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200, highlight=False)
    return Renderer(console), buf


def test_assistant_delta_writes_text_without_newline() -> None:
    r, buf = _capture()
    r.on_event(AssistantDelta(text="hello"))
    assert "hello" in buf.getvalue()


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


def test_final_is_silent() -> None:
    r, buf = _capture()
    r.on_event(Final(message="done"))
    assert buf.getvalue() == ""


def test_finish_emits_trailing_newline() -> None:
    r, buf = _capture()
    r.finish()
    assert buf.getvalue() == "\n"


def test_long_tool_args_are_truncated() -> None:
    r, buf = _capture()
    long_value = "x" * 500
    r.on_event(ToolCallStarted(name="t", input={"data": long_value}))
    out = buf.getvalue()
    assert "…" in out
    assert len(out) < 200


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
