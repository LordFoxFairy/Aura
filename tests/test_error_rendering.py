"""Tests for tool-error panel rendering + permission-audit crispness."""

from __future__ import annotations

import io

from rich.console import Console

from aura.cli.render import Renderer, _hint_for_error, _render_tool_error
from aura.schemas.events import PermissionAudit, ToolCallCompleted, ToolCallStarted


def _capture() -> tuple[Renderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200, highlight=False)
    return Renderer(console), buf


def _render_string(panel: object) -> str:
    """Render a renderable to plain text for substring checks."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200, highlight=False)
    console.print(panel)
    return buf.getvalue()


def test_error_panel_includes_tool_name() -> None:
    panel = _render_tool_error("read_file", "not found: missing.py")
    out = _render_string(panel)
    assert "read_file" in out
    assert "not found: missing.py" in out
    # A boxed panel draws corner glyphs — one of the border chars should appear.
    assert "╭" in out or "┌" in out


def test_hint_matched_for_not_found() -> None:
    hint = _hint_for_error("read_file", "not found: missing.py")
    assert hint is not None
    assert "path" in hint.lower()


def test_hint_matched_for_must_read_first() -> None:
    hint = _hint_for_error("edit_file", "file has not been read yet")
    assert hint is not None
    assert "read_file" in hint


def test_hint_matched_for_ripgrep_missing() -> None:
    hint = _hint_for_error("grep", "ripgrep (rg) on PATH not found")
    assert hint is not None
    assert "brew install ripgrep" in hint or "install" in hint.lower()


def test_no_hint_for_unknown_error() -> None:
    hint = _hint_for_error("bash", "something entirely unexpected happened xyz123")
    assert hint is None


def test_permission_audit_line_renders_crisply() -> None:
    r, buf = _capture()
    r.on_event(ToolCallStarted(name="bash", input={"command": "npm test"}))
    r.on_event(
        PermissionAudit(tool="bash", text="auto-allowed: rule `bash(npm test)`"),
    )
    out = buf.getvalue()
    # The rule content itself is in the audit text — still visible after
    # rich strips markup.
    assert "bash(npm test)" in out
    assert "auto-allowed: rule" in out
    # Keep the 4-space indent contract from spec §8.4.
    audit_line = next(line for line in out.splitlines() if "auto-allowed" in line)
    assert audit_line.startswith("    ")


def test_tool_completed_error_uses_panel_layout() -> None:
    r, buf = _capture()
    r.on_event(
        ToolCallCompleted(
            name="read_file", output=None, error="not found: missing.py",
        ),
    )
    out = buf.getvalue()
    # Panel rendering uses box-drawing borders, not the old inline ``✗`` line.
    assert "read_file" in out
    assert "not found: missing.py" in out
    assert "╭" in out or "┌" in out
