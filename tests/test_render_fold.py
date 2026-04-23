"""Tests for the render-fold path (Finding B).

Search/read tools whose metadata declares ``is_search_command=True`` get
their output folded in the REPL when it exceeds ``_FOLD_THRESHOLD`` (20)
lines — head (10) + ``…`` + tail (5) + a ``[N lines total, 15 shown]``
footer. Other tools render as usual even for large outputs.
"""

from __future__ import annotations

import io

from rich.console import Console

from aura.cli.render import (
    _FOLD_HEAD_LINES,
    _FOLD_TAIL_LINES,
    _FOLD_THRESHOLD,
    _SEARCH_COMMAND_TOOLS,
    Renderer,
)
from aura.schemas.events import ToolCallCompleted
from aura.tools.glob import glob
from aura.tools.grep import grep
from aura.tools.read_file import read_file
from aura.tools.web_fetch import web_fetch
from aura.tools.web_search import WebSearch


def _capture() -> tuple[Renderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200, highlight=False)
    return Renderer(console), buf


def _build_grep_content_output(n_lines: int) -> dict[str, object]:
    """Build a realistic grep content-mode dict with N match entries."""
    matches = [
        {"path": f"src/file_{i}.py", "line": i + 1, "text": f"match_{i}"}
        for i in range(n_lines)
    ]
    return {"mode": "content", "matches": matches, "truncated": False}


def test_fold_triggers_for_grep_output_over_threshold() -> None:
    r, buf = _capture()
    r.on_event(ToolCallCompleted(
        name="grep",
        output=_build_grep_content_output(30),
    ))
    out = buf.getvalue()
    # Footer with total + shown line counts (10 head + 5 tail = 15 shown).
    assert "30 lines total" in out
    assert "15 shown" in out
    # "…" separator between head and tail.
    assert "…" in out
    # Head should include early matches; tail should include late matches.
    # match_0 is in the head; match_29 is in the tail; match_15 is hidden.
    assert "match_0" in out
    assert "match_29" in out
    assert "match_15" not in out


def test_no_fold_when_output_under_threshold() -> None:
    # 20 lines exactly = threshold; fold fires only for > threshold.
    # Pick 15 to sit comfortably below.
    r, buf = _capture()
    r.on_event(ToolCallCompleted(
        name="grep",
        output=_build_grep_content_output(15),
    ))
    out = buf.getvalue()
    # No fold footer, no "…" separator.
    assert "lines total" not in out
    assert "…" not in out
    # The standard ✓-summary path still runs.
    assert "✓" in out


def test_no_fold_for_non_search_tool_even_if_output_long() -> None:
    # ``bash`` is not in _SEARCH_COMMAND_TOOLS (its output is user-
    # requested command output, not a listing to collapse). A long bash
    # result must render through its normal formatter without any fold.
    r, buf = _capture()
    long_stdout = "\n".join(f"line {i}" for i in range(50))
    r.on_event(ToolCallCompleted(
        name="bash",
        output={
            "stdout": long_stdout,
            "stderr": "",
            "exit_code": 0,
            "truncated": False,
            "killed_at_hard_ceiling": False,
        },
    ))
    out = buf.getvalue()
    # No fold footer.
    assert "lines total" not in out
    # Standard bash formatter still fires.
    assert "✓" in out


def test_fold_footer_reflects_correct_line_counts() -> None:
    r, buf = _capture()
    r.on_event(ToolCallCompleted(
        name="grep",
        output=_build_grep_content_output(100),
    ))
    out = buf.getvalue()
    # 100 total, 10 head + 5 tail = 15 shown.
    assert "100 lines total" in out
    assert "15 shown" in out


def test_fold_applies_to_glob_files_output() -> None:
    # glob returns {"files": [...]} — the fold path must recognise that
    # shape as well as grep's {"matches": [...]}.
    r, buf = _capture()
    r.on_event(ToolCallCompleted(
        name="glob",
        output={
            "files": [f"src/file_{i}.py" for i in range(25)],
            "count": 25,
            "truncated": False,
        },
    ))
    out = buf.getvalue()
    assert "25 lines total" in out
    assert "15 shown" in out


def test_fold_applies_to_read_file_content() -> None:
    # read_file returns {"content": "..."} — lines come from splitting
    # the content string, not a list.
    r, buf = _capture()
    content = "\n".join(f"line_{i}" for i in range(30))
    r.on_event(ToolCallCompleted(
        name="read_file",
        output={
            "content": content,
            "lines": 30,
            "total_lines": 30,
            "offset": 0,
            "limit": None,
            "partial": False,
        },
    ))
    out = buf.getvalue()
    assert "30 lines total" in out
    assert "15 shown" in out


def test_search_command_set_matches_tool_metadata() -> None:
    # Drift guard: the static _SEARCH_COMMAND_TOOLS set in render.py must
    # stay in sync with the ``is_search_command=True`` metadata on each
    # tool. If a tool flips the flag, the renderer set has to flip too —
    # this test forces the two updates to land together.
    ws = WebSearch()
    tool_flag: dict[str, bool] = {
        "grep": bool((grep.metadata or {}).get("is_search_command")),
        "glob": bool((glob.metadata or {}).get("is_search_command")),
        "read_file": bool((read_file.metadata or {}).get("is_search_command")),
        "web_fetch": bool((web_fetch.metadata or {}).get("is_search_command")),
        "web_search": bool((ws.metadata or {}).get("is_search_command")),
    }
    expected = {name for name, flag in tool_flag.items() if flag}
    assert frozenset(expected) == _SEARCH_COMMAND_TOOLS


def test_web_fetch_is_not_search_command() -> None:
    # Explicit: fetched documents are user-requested content; folding
    # would hide exactly what the user asked for.
    assert bool((web_fetch.metadata or {}).get("is_search_command")) is False
    assert "web_fetch" not in _SEARCH_COMMAND_TOOLS


def test_fold_constants_are_consistent() -> None:
    # Sanity: head + tail must be strictly less than the threshold, else
    # the fold hides zero lines (or negative lines) at threshold+1 and
    # the UI reports a nonsense ratio.
    assert _FOLD_HEAD_LINES + _FOLD_TAIL_LINES < _FOLD_THRESHOLD


def test_fold_output_structure_is_dim_bar_head_ellipsis_bar_tail() -> None:
    # Visual structure check — the renderer emits:
    #   │ <head line 1>
    #   ...
    #   │ <head line 10>
    #   …
    #   │ <tail line 1>
    #   ...
    #   │ <tail line 5>
    #   [N lines total, 15 shown]
    # Each body line is prefixed with ``│`` (same glyph as bash progress).
    r, buf = _capture()
    r.on_event(ToolCallCompleted(
        name="grep",
        output=_build_grep_content_output(30),
    ))
    out = buf.getvalue()
    # At least 15 ``│`` prefixes (head + tail).
    assert out.count("│") >= 15
    # Exactly one ``…`` separator between head and tail.
    assert out.count("…") == 1
