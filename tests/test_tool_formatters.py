"""Tests for per-tool result formatters in aura.cli.render."""

from __future__ import annotations

from aura.cli.render import (
    _TOOL_RESULT_FORMATTERS,
    _format_bash_result,
    _format_edit_file_result,
    _format_glob_result,
    _format_grep_result,
    _format_read_file_result,
    _format_task_create_result,
    _format_write_file_result,
)


# --------------------------------------------------------------------------
# read_file
# --------------------------------------------------------------------------
def test_read_file_full_shows_lines_and_size() -> None:
    out = _format_read_file_result({
        "content": "x" * 4300,  # ~4.2 KB
        "lines": 142,
        "total_lines": 142,
        "offset": 0,
        "limit": None,
        "partial": False,
    })
    assert "142 lines" in out
    assert "KB" in out


def test_read_file_partial_shows_out_of_total() -> None:
    out = _format_read_file_result({
        "content": "abc",
        "lines": 50,
        "total_lines": 142,
        "offset": 0,
        "limit": 50,
        "partial": True,
    })
    assert "50" in out
    assert "142" in out
    assert "partial" in out


def test_read_file_non_dict_falls_through_to_str() -> None:
    assert _format_read_file_result("weird") == "weird"


# --------------------------------------------------------------------------
# write_file
# --------------------------------------------------------------------------
def test_write_file_bytes_rendering() -> None:
    out = _format_write_file_result({"bytes": 1234})
    assert "1234" in out
    assert "bytes" in out


def test_write_file_missing_bytes_fallback() -> None:
    assert _format_write_file_result({}) == "written"


# --------------------------------------------------------------------------
# edit_file
# --------------------------------------------------------------------------
def test_edit_file_created_pluralises() -> None:
    out = _format_edit_file_result({"replacements": 1, "created": True})
    assert "created" in out
    assert "1" in out
    # singular
    assert "line)" in out


def test_edit_file_replacements_singular_vs_plural() -> None:
    one = _format_edit_file_result({"replacements": 1})
    many = _format_edit_file_result({"replacements": 3})
    assert "1 replacement" in one
    assert "replacements" not in one.replace("replacement", "", 1)  # only singular form present
    assert "3 replacements" in many


def test_edit_file_non_dict_fallback() -> None:
    assert _format_edit_file_result(None) == "edited"


# --------------------------------------------------------------------------
# grep
# --------------------------------------------------------------------------
def test_grep_files_with_matches_mode() -> None:
    out = _format_grep_result({
        "mode": "files_with_matches",
        "files": ["a.py", "b.py", "c.py"],
        "truncated": False,
    })
    assert "3 files" in out
    assert "truncated" not in out


def test_grep_files_with_matches_singular_and_truncated() -> None:
    out = _format_grep_result({
        "mode": "files_with_matches",
        "files": ["a.py"],
        "truncated": True,
    })
    assert "1 file" in out
    assert "truncated" in out


def test_grep_content_mode() -> None:
    out = _format_grep_result({
        "mode": "content",
        "matches": [{"path": "a.py"}, {"path": "b.py"}],
        "truncated": False,
    })
    assert "2 matches" in out


def test_grep_count_mode_reports_total() -> None:
    out = _format_grep_result({
        "mode": "count",
        "counts": {"a.py": 3, "b.py": 4},
        "total": 7,
        "truncated": False,
    })
    assert "7 matches" in out


def test_grep_unknown_mode_fallback() -> None:
    assert _format_grep_result({"mode": "weird"}) == "searched"


# --------------------------------------------------------------------------
# glob
# --------------------------------------------------------------------------
def test_glob_reports_count() -> None:
    out = _format_glob_result({"files": ["a", "b"], "count": 2, "truncated": False})
    assert "2 files" in out


def test_glob_singular_and_truncated() -> None:
    out = _format_glob_result({"files": ["a"], "count": 1, "truncated": True})
    assert "1 file" in out
    assert "truncated" in out


# --------------------------------------------------------------------------
# bash
# --------------------------------------------------------------------------
def test_bash_exit_zero_shows_ok() -> None:
    out = _format_bash_result({
        "stdout": "x", "stderr": "", "exit_code": 0,
        "truncated": False, "killed_at_hard_ceiling": False,
    })
    assert out == "ok"


def test_bash_non_zero_exit_shows_code() -> None:
    out = _format_bash_result({
        "stdout": "", "stderr": "oops", "exit_code": 127,
        "truncated": False, "killed_at_hard_ceiling": False,
    })
    assert "exit 127" in out


def test_bash_hard_ceiling_marker() -> None:
    out = _format_bash_result({
        "stdout": "", "stderr": "", "exit_code": 0,
        "truncated": True, "killed_at_hard_ceiling": True,
    })
    assert "ceiling" in out or "100 MB" in out


def test_bash_truncated_marker() -> None:
    out = _format_bash_result({
        "stdout": "x", "stderr": "", "exit_code": 0,
        "truncated": True, "killed_at_hard_ceiling": False,
    })
    assert "truncated" in out


# --------------------------------------------------------------------------
# task_create
# --------------------------------------------------------------------------
def test_task_create_shows_short_id_and_description() -> None:
    out = _format_task_create_result({
        "task_id": "abcdef1234567890",
        "description": "write tests",
        "status": "running",
    })
    assert "abcdef12" in out
    assert "write tests" in out
    # Short id is only first 8 chars
    assert "abcdef1234" not in out


def test_task_create_non_dict_fallback() -> None:
    assert _format_task_create_result(None) == "spawned"


# --------------------------------------------------------------------------
# registry wiring
# --------------------------------------------------------------------------
def test_registry_maps_all_expected_tools() -> None:
    for name in [
        "read_file", "write_file", "edit_file", "grep", "glob",
        "bash", "task_create",
    ]:
        assert name in _TOOL_RESULT_FORMATTERS


def test_unknown_tool_not_in_registry() -> None:
    assert "does_not_exist_tool" not in _TOOL_RESULT_FORMATTERS
