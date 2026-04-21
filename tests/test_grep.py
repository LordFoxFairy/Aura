"""Tests for aura.tools.grep (ripgrep-backed)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from aura.schemas.tool import ToolError
from aura.tools.grep import grep


async def test_default_mode_is_files_with_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("foo bar\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("foo baz\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("nothing here\n", encoding="utf-8")
    out = await grep.ainvoke({"pattern": "foo", "path": str(tmp_path)})
    assert out["mode"] == "files_with_matches"
    assert len(out["files"]) == 2
    assert all(p.endswith((".txt",)) for p in out["files"])
    assert out["truncated"] is False


async def test_content_mode_returns_match_objects(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello world\nno match\nhello again\n", encoding="utf-8")
    out = await grep.ainvoke(
        {"pattern": "hello", "path": str(tmp_path), "output_mode": "content"}
    )
    assert out["mode"] == "content"
    assert len(out["matches"]) == 2
    for m in out["matches"]:
        assert "path" in m
        assert "line" in m
        assert "text" in m
        assert isinstance(m["line"], int)
    lines = {m["line"] for m in out["matches"]}
    assert lines == {1, 3}


async def test_count_mode_returns_per_file_counts(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("foo\nfoo\nfoo\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("foo\n", encoding="utf-8")
    out = await grep.ainvoke(
        {"pattern": "foo", "path": str(tmp_path), "output_mode": "count"}
    )
    assert out["mode"] == "count"
    assert isinstance(out["counts"], dict)
    assert sum(out["counts"].values()) == 4
    assert out["total"] == 4


async def test_case_insensitive_flag(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("foo bar\n", encoding="utf-8")
    out = await grep.ainvoke(
        {"pattern": "FOO", "path": str(tmp_path), "case_insensitive": True}
    )
    assert len(out["files"]) == 1


async def test_multiline_mode_spans_newlines(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    out = await grep.ainvoke(
        {
            "pattern": r"def foo\(\):\s+pass",
            "path": str(tmp_path),
            "multiline": True,
            "output_mode": "content",
        }
    )
    assert len(out["matches"]) >= 1


async def test_context_before_after_in_content_mode(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text(
        "line1\nline2\nline3\nMATCH\nline5\nline6\n", encoding="utf-8",
    )
    out = await grep.ainvoke(
        {
            "pattern": "MATCH",
            "path": str(tmp_path),
            "output_mode": "content",
            "context_before": 2,
            "context_after": 1,
        }
    )
    entries = out["matches"]
    context_entries = [e for e in entries if e.get("is_context")]
    match_entries = [e for e in entries if not e.get("is_context")]
    assert len(match_entries) == 1
    assert len(context_entries) == 3


def test_context_rejected_in_non_content_mode() -> None:
    from aura.tools.grep import GrepParams

    with pytest.raises(ValidationError):
        GrepParams(pattern="x", context_before=2)
    with pytest.raises(ValidationError):
        GrepParams(pattern="x", context_after=2)
    with pytest.raises(ValidationError):
        GrepParams(pattern="x", output_mode="count", context_before=2)


async def test_glob_filter(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("import os\n", encoding="utf-8")
    (tmp_path / "notes.js").write_text("import something\n", encoding="utf-8")
    out = await grep.ainvoke(
        {"pattern": "import", "path": str(tmp_path), "glob": "*.py"}
    )
    assert len(out["files"]) == 1
    assert out["files"][0].endswith("code.py")


async def test_type_filter(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("import os\n", encoding="utf-8")
    (tmp_path / "notes.js").write_text("import something\n", encoding="utf-8")
    out = await grep.ainvoke(
        {"pattern": "import", "path": str(tmp_path), "type": "py"}
    )
    assert len(out["files"]) == 1
    assert out["files"][0].endswith("code.py")


async def test_head_limit_truncates(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("match\n", encoding="utf-8")
    out = await grep.ainvoke(
        {"pattern": "match", "path": str(tmp_path), "head_limit": 2}
    )
    assert len(out["files"]) == 2
    assert out["truncated"] is True


async def test_no_matches_returns_empty_not_error(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("nothing to see\n", encoding="utf-8")
    out = await grep.ainvoke({"pattern": "zzznomatchzzz", "path": str(tmp_path)})
    assert out["mode"] == "files_with_matches"
    assert out["files"] == []
    assert out["truncated"] is False


async def test_missing_rg_raises_clear_error(tmp_path: Path) -> None:
    with (
        patch("aura.tools.grep.shutil.which", return_value=None),
        pytest.raises(ToolError, match="ripgrep"),
    ):
        await grep.ainvoke({"pattern": "x", "path": str(tmp_path)})


async def test_real_error_raises_tool_error(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    with pytest.raises(ToolError):
        await grep.ainvoke({"pattern": "[", "path": str(tmp_path)})


def test_grep_capability_flags() -> None:
    meta = grep.metadata or {}
    assert meta.get("is_read_only") is True
    assert meta.get("is_concurrency_safe") is True
    assert meta.get("is_destructive") is False


def test_grep_metadata_includes_matcher_and_preview() -> None:
    meta = grep.metadata or {}
    assert meta.get("rule_matcher") is not None
    preview = meta.get("args_preview")
    assert callable(preview)
    assert preview({"pattern": "foo", "path": "src"}) == "pattern: foo  @ src"


async def test_content_mode_without_context(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    out = await grep.ainvoke(
        {"pattern": "hello", "path": str(tmp_path), "output_mode": "content"}
    )
    assert out["mode"] == "content"
    assert len(out["matches"]) == 1
    assert out["matches"][0]["line"] == 1


async def test_content_mode_parses_path_with_hyphen_digits(tmp_path: Path) -> None:
    # Regression for the `-<digits>-` path-parse bug: rg's context-line
    # output uses a '-' separator, and the naive parser scanned for the
    # first `-N-` pair — which tripped inside paths like
    # src/v-42-release/foo.rs (the literal "-42-" ate the parse). Fixed by
    # passing sentinel separators to rg that never appear in paths.
    nested = tmp_path / "src" / "v-42-release"
    nested.mkdir(parents=True)
    target = nested / "foo.rs"
    target.write_text("line1\nmatch_here\nline3\n", encoding="utf-8")

    out = await grep.ainvoke(
        {
            "pattern": "match_here",
            "path": str(tmp_path),
            "output_mode": "content",
            "context_before": 1,
            "context_after": 1,
        }
    )
    assert out["mode"] == "content"
    # Match + 2 context lines = 3 entries.
    assert len(out["matches"]) >= 1

    # Full path preserved (no truncation at "src/v" or similar).
    match_entry = next(
        m for m in out["matches"] if not m.get("is_context", False)
    )
    assert match_entry["path"] == str(target)
    assert match_entry["line"] == 2
    assert match_entry["text"] == "match_here"


async def test_content_mode_distinguishes_match_vs_context_on_hyphen_path(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "python-3-lib"
    nested.mkdir(parents=True)
    target = nested / "mod.py"
    target.write_text("pre\nmatch_here\npost\n", encoding="utf-8")

    out = await grep.ainvoke(
        {
            "pattern": "match_here",
            "path": str(tmp_path),
            "output_mode": "content",
            "context_before": 1,
            "context_after": 1,
        }
    )
    match_entries = [m for m in out["matches"] if not m.get("is_context", False)]
    ctx_entries = [m for m in out["matches"] if m.get("is_context", False)]
    # Exactly one match (on line 2); two context lines (lines 1 and 3).
    assert len(match_entries) == 1
    assert len(ctx_entries) == 2
    assert match_entries[0]["line"] == 2
    assert {e["line"] for e in ctx_entries} == {1, 3}


async def test_content_mode_single_hyphen_path_still_works(tmp_path: Path) -> None:
    # Regression guard: a simple hyphenated dir (no digits in between) must
    # also round-trip cleanly — confirms the fix isn't narrowly tied to
    # "digits between hyphens".
    nested = tmp_path / "foo-bar"
    nested.mkdir(parents=True)
    target = nested / "baz.txt"
    target.write_text("first\nmatch_here\nlast\n", encoding="utf-8")

    out = await grep.ainvoke(
        {
            "pattern": "match_here",
            "path": str(tmp_path),
            "output_mode": "content",
            "context_before": 1,
        }
    )
    match_entry = next(
        m for m in out["matches"] if not m.get("is_context", False)
    )
    assert match_entry["path"] == str(target)
