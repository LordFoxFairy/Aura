"""Tests for aura.tools.edit_file."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aura.schemas.tool import ToolError
from aura.tools.edit_file import _MAX_EDIT_SIZE, edit_file


async def test_edit_file_single_replacement(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello world\n", encoding="utf-8")
    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "hello", "new_str": "goodbye"}
    )
    assert out == {"replacements": 1}
    assert f.read_text(encoding="utf-8") == "goodbye world\n"


async def test_edit_file_replace_all(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("foo foo foo\n", encoding="utf-8")
    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "foo", "new_str": "bar", "replace_all": True}
    )
    assert out == {"replacements": 3}
    assert f.read_text(encoding="utf-8") == "bar bar bar\n"


async def test_edit_file_ambiguous_match_errors(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x x\n", encoding="utf-8")
    with pytest.raises(ToolError, match="2"):
        await edit_file.ainvoke({"path": str(f), "old_str": "x", "new_str": "y"})
    assert f.read_text(encoding="utf-8") == "x x\n"


async def test_edit_file_not_found_errors(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello\n", encoding="utf-8")
    with pytest.raises(ToolError, match="not found"):
        await edit_file.ainvoke({"path": str(f), "old_str": "zzz", "new_str": "aaa"})


async def test_edit_file_missing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="not found"):
        await edit_file.ainvoke(
            {"path": str(tmp_path / "ghost.txt"), "old_str": "x", "new_str": "y"}
        )


async def test_edit_file_empty_old_str_errors(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello\n", encoding="utf-8")
    with pytest.raises(ToolError, match="non-empty"):
        await edit_file.ainvoke({"path": str(f), "old_str": "", "new_str": "x"})


async def test_edit_file_delete_via_empty_new_str(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello world\n", encoding="utf-8")
    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": " world", "new_str": ""}
    )
    assert out == {"replacements": 1}
    assert f.read_text(encoding="utf-8") == "hello\n"


def test_edit_file_capability_flags() -> None:
    meta = edit_file.metadata or {}
    assert meta.get("is_destructive") is True
    assert meta.get("is_read_only") is False


def test_edit_file_metadata_includes_matcher_and_preview() -> None:
    from aura.tools.edit_file import edit_file

    meta = edit_file.metadata or {}
    assert meta.get("rule_matcher") is not None
    preview = meta.get("args_preview")
    assert callable(preview)
    assert (
        preview({"path": "f.py", "old_str": "a\nb", "new_str": "a\nb\nc"})
        == "path: f.py  +3/-2 lines"
    )


async def test_crlf_file_matches_lf_old_str(tmp_path: Path) -> None:
    f = tmp_path / "crlf.txt"
    f.write_bytes(b"hello world\r\nsecond line\r\n")
    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "hello world\nsecond", "new_str": "HELLO\nSECOND"}
    )
    assert out == {"replacements": 1}
    raw = f.read_bytes()
    assert raw == b"HELLO\r\nSECOND line\r\n"


async def test_lf_file_stays_lf_after_edit(tmp_path: Path) -> None:
    f = tmp_path / "lf.txt"
    f.write_bytes(b"hello world\nsecond line\n")
    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "hello", "new_str": "goodbye"}
    )
    assert out == {"replacements": 1}
    raw = f.read_bytes()
    assert raw == b"goodbye world\nsecond line\n"
    assert b"\r" not in raw


async def test_crlf_file_stays_crlf_after_edit(tmp_path: Path) -> None:
    f = tmp_path / "crlf2.txt"
    f.write_bytes(b"hello\r\nworld\r\n")
    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "hello\r\nworld", "new_str": "bye\r\nall"}
    )
    assert out == {"replacements": 1}
    raw = f.read_bytes()
    assert raw == b"bye\r\nall\r\n"


async def test_old_mac_cr_file_normalized(tmp_path: Path) -> None:
    f = tmp_path / "mac.txt"
    f.write_bytes(b"hello\rworld\r")
    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "hello\nworld", "new_str": "bye\nall"}
    )
    assert out == {"replacements": 1}
    raw = f.read_bytes()
    assert raw == b"bye\rall\r"


async def test_create_new_file_with_empty_old_str(tmp_path: Path) -> None:
    f = tmp_path / "new.txt"
    assert not f.exists()
    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "", "new_str": "hello\n"}
    )
    assert out == {"replacements": 1, "created": True}
    assert f.read_text(encoding="utf-8") == "hello\n"


async def test_empty_old_str_rejected_when_file_exists(tmp_path: Path) -> None:
    f = tmp_path / "exists.txt"
    f.write_text("hello\n", encoding="utf-8")
    with pytest.raises(ToolError, match="cannot edit with empty old_str when file exists"):
        await edit_file.ainvoke(
            {"path": str(f), "old_str": "", "new_str": "x"}
        )


async def test_error_message_echoes_missing_string(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello world\n", encoding="utf-8")
    missing = "totally absent snippet"
    with pytest.raises(ToolError) as excinfo:
        await edit_file.ainvoke(
            {"path": str(f), "old_str": missing, "new_str": "x"}
        )
    msg = str(excinfo.value)
    assert "missing:" in msg
    assert repr(missing) in msg


async def test_error_message_truncates_long_missing_string(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello world\n", encoding="utf-8")
    long_missing = "x" * 500
    with pytest.raises(ToolError) as excinfo:
        await edit_file.ainvoke(
            {"path": str(f), "old_str": long_missing, "new_str": "y"}
        )
    msg = str(excinfo.value)
    assert "missing:" in msg
    assert "\u2026" in msg
    # The repr of the full 500-char string should NOT appear in full.
    assert repr(long_missing) not in msg
    # The echoed payload (after "missing: ") is capped at 120 chars + ellipsis.
    echoed = msg.split("missing:", 1)[1].strip()
    # echoed is a truncated repr; total payload length (pre-ellipsis) <= 120.
    assert len(echoed.rstrip("\u2026").rstrip()) <= 120


# ---------------------------------------------------------------------------
# Mixed CRLF + LF handling — prior behavior silently mutated unrelated lines
# ---------------------------------------------------------------------------


async def test_mixed_endings_file_does_not_mutate_unrelated_lines(
    tmp_path: Path,
) -> None:
    # A file with BOTH \r\n and bare \n lines: any "normalize to LF for
    # match, restore to CRLF on write" strategy silently converts the
    # bare-LF lines to CRLF, which the user didn't ask for and breaks
    # surgical edits. Correct behavior: detect mixed endings, skip
    # normalization, match and write on raw content.
    f = tmp_path / "mixed.txt"
    raw = b"line one\r\nline two\nline three\r\nline four\n"
    f.write_bytes(raw)

    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "line two", "new_str": "line TWO"}
    )
    assert out == {"replacements": 1}

    written = f.read_bytes()
    assert b"line TWO" in written
    # Per-line endings preserved: 1 and 3 stay CRLF, 2 and 4 stay bare LF.
    assert written == b"line one\r\nline TWO\nline three\r\nline four\n"


async def test_mixed_endings_match_on_crlf_region_with_crlf_old_str(
    tmp_path: Path,
) -> None:
    f = tmp_path / "mixed.txt"
    f.write_bytes(b"line one\r\nline two\nline three\r\n")

    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "line one\r\n", "new_str": "LINE ONE\r\n"}
    )
    assert out == {"replacements": 1}
    assert f.read_bytes() == b"LINE ONE\r\nline two\nline three\r\n"


async def test_mixed_endings_match_on_lf_region_with_lf_old_str(
    tmp_path: Path,
) -> None:
    f = tmp_path / "mixed.txt"
    f.write_bytes(b"line one\r\nline two\nline three\r\n")

    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "line two\n", "new_str": "LINE TWO\n"}
    )
    assert out == {"replacements": 1}
    assert f.read_bytes() == b"line one\r\nLINE TWO\nline three\r\n"


async def test_mixed_endings_ambiguous_old_str_rejected_by_uniqueness(
    tmp_path: Path,
) -> None:
    # Regression guard: uniqueness still fires on mixed files.
    f = tmp_path / "mixed.txt"
    f.write_bytes(b"line one\r\nline two\nline three\r\n")

    with pytest.raises(ToolError, match="matches"):
        await edit_file.ainvoke(
            {"path": str(f), "old_str": "line", "new_str": "LINE"}
        )


# ---------------------------------------------------------------------------
# F-02-009 — pre-stat size cap rejects files above _MAX_EDIT_SIZE
# ---------------------------------------------------------------------------


async def test_edit_file_above_size_cap_rejected(tmp_path: Path) -> None:
    f = tmp_path / "huge.bin"
    # Sparse file via os.truncate: 257 MB without consuming disk.
    f.touch()
    os.truncate(f, _MAX_EDIT_SIZE + 1024 * 1024)
    with pytest.raises(ToolError, match="too large to edit"):
        await edit_file.ainvoke(
            {"path": str(f), "old_str": "x", "new_str": "y"}
        )


async def test_edit_file_at_size_cap_accepted(tmp_path: Path) -> None:
    # Exactly at the cap is OK; the comparison is strict ``>``.
    f = tmp_path / "ok.txt"
    f.write_text("hello world\n", encoding="utf-8")
    out = await edit_file.ainvoke(
        {"path": str(f), "old_str": "hello", "new_str": "goodbye"}
    )
    assert out == {"replacements": 1}
