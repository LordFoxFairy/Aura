"""Tests for aura.tools.grep (ripgrep-backed)."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from aura.domain.tool import ToolError
from aura.domain.tool_meta_access import meta_dict
from aura.tools.grep import _CTX_SEP, _MATCH_SEP, grep

# `import aura.tools.grep` binds the re-exported Grep instance, so reach the
# real module object through sys.modules to patch its `subprocess` reference.
_grep_mod: ModuleType = sys.modules["aura.tools.grep"]


class _FakeSubprocess:
    """Stand-in for grep's `subprocess` reference; swaps only `run` at the seam."""

    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, run: Callable[..., subprocess.CompletedProcess[str]]) -> None:
        self.run = run


def _patch_run(
    monkeypatch: pytest.MonkeyPatch,
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    """Replace grep's module-level subprocess so the global stdlib one stays intact."""
    monkeypatch.setattr(_grep_mod, "subprocess", _FakeSubprocess(run))


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
    meta = meta_dict(grep)
    assert meta.get("is_read_only") is True
    assert meta.get("is_concurrency_safe") is True
    assert meta.get("is_destructive") is False


def test_grep_metadata_includes_matcher_and_preview() -> None:
    meta = meta_dict(grep)
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


async def test_grep_excludes_vcs_directories(tmp_path: Path) -> None:
    # Simulate a checkout: the magic word lives inside .git/HEAD AND in a
    # tracked source file. Auto-exclude must drop the .git/ hit.
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/SECRET_TOKEN\n", encoding="utf-8")
    svn_dir = tmp_path / ".svn"
    svn_dir.mkdir()
    (svn_dir / "entries").write_text("SECRET_TOKEN\n", encoding="utf-8")
    (tmp_path / "src.py").write_text("# SECRET_TOKEN here\n", encoding="utf-8")

    out = await grep.ainvoke({"pattern": "SECRET_TOKEN", "path": str(tmp_path)})
    assert any(p.endswith("src.py") for p in out["files"])
    assert not any(".git" in p for p in out["files"])
    assert not any(".svn" in p for p in out["files"])


async def test_grep_max_columns_truncates_long_lines(tmp_path: Path) -> None:
    long_line = "x" * 600 + "MARKER" + "y" * 600
    (tmp_path / "long.txt").write_text(long_line + "\n", encoding="utf-8")
    out = await grep.ainvoke(
        {"pattern": "MARKER", "path": str(tmp_path), "output_mode": "content"}
    )
    assert out["mode"] == "content"
    assert len(out["matches"]) == 1
    text = out["matches"][0]["text"]
    assert len(text) < 600
    # rg signals truncation with a bracketed notice (`[Omitted ...]` or `[... omitted]`).
    assert "[" in text and "]" in text


async def test_grep_max_columns_param_override(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x" * 50 + "MARK" + "y" * 50 + "\n", encoding="utf-8")
    out = await grep.ainvoke(
        {
            "pattern": "MARK",
            "path": str(tmp_path),
            "output_mode": "content",
            "max_columns": 20,
        }
    )
    assert len(out["matches"]) == 1
    assert "[" in out["matches"][0]["text"]


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


def _fake_proc(
    stdout: str, *, returncode: int = 0, stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    """Build a fake ripgrep result for seam-mocking subprocess.run."""
    return subprocess.CompletedProcess(
        args=["rg"], returncode=returncode, stdout=stdout, stderr=stderr,
    )


async def test_timeout_raises_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ripgrep that hangs must surface as a bounded ToolError, never block forever."""
    def _boom(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="rg", timeout=30)

    _patch_run(monkeypatch, _boom)
    with pytest.raises(ToolError, match="timed out"):
        await grep.ainvoke({"pattern": "x", "path": str(tmp_path)})


async def test_count_mode_drops_unparseable_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed rg count rows must be skipped, not corrupt the per-file totals."""
    stdout = "good.py:3\nno_colon_line\nbad.py:notanumber\nother.py:2\n"

    def _fake(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _fake_proc(stdout)

    _patch_run(monkeypatch, _fake)
    out = await grep.ainvoke(
        {"pattern": "x", "path": str(tmp_path), "output_mode": "count"}
    )
    assert out["mode"] == "count"
    assert out["counts"] == {"good.py": 3, "other.py": 2}
    assert out["total"] == 5
    assert out["truncated"] is False


async def test_content_mode_drops_unparseable_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garbage stdout rows from rg must not become phantom matches."""
    valid = f"file.py{_MATCH_SEP}7{_MATCH_SEP}hit"
    stdout = f"{valid}\nrow with no separators\nfile.py{_MATCH_SEP}NaN{_MATCH_SEP}x\n"

    def _fake(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _fake_proc(stdout)

    _patch_run(monkeypatch, _fake)
    out = await grep.ainvoke(
        {"pattern": "x", "path": str(tmp_path), "output_mode": "content"}
    )
    assert out["mode"] == "content"
    assert len(out["matches"]) == 1
    assert out["matches"][0] == {"path": "file.py", "line": 7, "text": "hit"}


async def test_content_mode_ignores_context_shaped_line_when_no_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without context flags, a context-separated row is not a match and is dropped."""
    ctx_shaped = f"file.py{_CTX_SEP}4{_CTX_SEP}around"
    match_line = f"file.py{_MATCH_SEP}5{_MATCH_SEP}real"
    stdout = f"{match_line}\n{ctx_shaped}\n"

    def _fake(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _fake_proc(stdout)

    _patch_run(monkeypatch, _fake)
    out = await grep.ainvoke(
        {"pattern": "x", "path": str(tmp_path), "output_mode": "content"}
    )
    assert len(out["matches"]) == 1
    assert out["matches"][0]["line"] == 5


async def test_content_mode_skips_group_separator(tmp_path: Path) -> None:
    """Multiple match groups separated by rg's '--' line must yield only real matches."""
    f = tmp_path / "a.txt"
    f.write_text(
        "a\nMARK\nb\n\n\n\nc\nMARK\nd\n", encoding="utf-8",
    )
    out = await grep.ainvoke(
        {
            "pattern": "MARK",
            "path": str(tmp_path),
            "output_mode": "content",
            "context_before": 1,
            "context_after": 1,
        }
    )
    match_entries = [m for m in out["matches"] if not m.get("is_context", False)]
    assert len(match_entries) == 2
    assert {m["line"] for m in match_entries} == {2, 8}
    assert all(m["text"] != "--" for m in out["matches"])


async def test_head_limit_zero_keeps_nothing_but_flags_truncation(
    tmp_path: Path,
) -> None:
    """head_limit=0 is the zero-boundary: drop every row yet still signal truncation."""
    (tmp_path / "a.txt").write_text("match\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("match\n", encoding="utf-8")
    out = await grep.ainvoke(
        {"pattern": "match", "path": str(tmp_path), "head_limit": 0}
    )
    assert out["files"] == []
    assert out["truncated"] is True


async def test_count_mode_head_limit_caps_files_and_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count cap must bound BOTH the file list and the summed total to the kept rows."""
    stdout = "a.py:5\nb.py:7\nc.py:9\n"

    def _fake(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _fake_proc(stdout)

    _patch_run(monkeypatch, _fake)
    out = await grep.ainvoke(
        {
            "pattern": "x",
            "path": str(tmp_path),
            "output_mode": "count",
            "head_limit": 2,
        }
    )
    assert out["counts"] == {"a.py": 5, "b.py": 7}
    assert out["total"] == 12
    assert out["truncated"] is True


async def test_rg_internal_error_propagates_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rg exit code >=2 is a hard failure; its stderr must reach the caller, trimmed."""
    def _fake(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _fake_proc("", returncode=2, stderr="  regex parse error\n")

    _patch_run(monkeypatch, _fake)
    with pytest.raises(ToolError, match="regex parse error"):
        await grep.ainvoke({"pattern": "x", "path": str(tmp_path)})


async def test_search_is_idempotent(tmp_path: Path) -> None:
    """Read-only search must return byte-identical results across repeated calls."""
    (tmp_path / "a.txt").write_text("foo\nfoo\n", encoding="utf-8")
    args = {"pattern": "foo", "path": str(tmp_path), "output_mode": "count"}
    first = await grep.ainvoke(dict(args))
    second = await grep.ainvoke(dict(args))
    assert first == second
    assert first["total"] == 2
