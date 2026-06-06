"""Tests for aura.tools.read_file."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from aura.domain.tool import ToolError, ToolMetadata, ValidationResult
from aura.domain.tool_meta_access import meta_dict
from aura.tools.read_file import ReadFileParams, read_file


async def test_read_file_utf8_success(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hello\nworld\n", encoding="utf-8")
    out = await read_file.ainvoke({"path": str(f)})
    assert out["content"] == "hello\nworld\n"
    assert out["lines"] == 2


async def test_read_file_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    out = await read_file.ainvoke({"path": str(f)})
    assert out["content"] == ""
    assert out["lines"] == 0


async def test_read_file_single_line_no_newline(tmp_path: Path) -> None:
    f = tmp_path / "single.txt"
    f.write_text("hello", encoding="utf-8")
    out = await read_file.ainvoke({"path": str(f)})
    assert out["lines"] == 1


async def test_read_file_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="not found"):
        await read_file.ainvoke({"path": str(tmp_path / "nope.txt")})


async def test_read_file_invalid_utf8(tmp_path: Path) -> None:
    f = tmp_path / "bad.bin"
    # No BOM prefix — random high bytes that are not valid UTF-8.
    f.write_bytes(b"\xc3\x28\xa0\xa1")
    with pytest.raises(ToolError, match="UTF-8"):
        await read_file.ainvoke({"path": str(f)})


async def test_read_file_oversize_head_truncated(tmp_path: Path) -> None:
    f = tmp_path / "big.bin"
    # 2 MB of single-byte ASCII lines: each "a\n" is 2 bytes ⇒ 1,048,576
    # lines. Head-truncated to 1 MB ⇒ 524,288 lines visible.
    f.write_bytes(b"a\n" * (1024 * 1024))
    out = await read_file.ainvoke({"path": str(f), "limit": 10})
    assert out["partial"] is True
    assert out["truncated_at_bytes"] == 1024 * 1024
    # First 10 lines come from the head of the file.
    assert out["content"] == "a\n" * 10
    assert out["lines"] == 10


async def test_read_file_within_cap_no_truncation_field(tmp_path: Path) -> None:
    f = tmp_path / "small.txt"
    f.write_text("hello\n", encoding="utf-8")
    out = await read_file.ainvoke({"path": str(f)})
    # Field present, but None when no head-truncation occurred.
    assert out["truncated_at_bytes"] is None
    assert out["partial"] is False


async def test_read_file_utf16_le_bom(tmp_path: Path) -> None:
    f = tmp_path / "u16le.txt"
    text = "hello\nworld\n"
    # codecs writer adds the LE BOM automatically when encoding via
    # ``utf-16``; here we write the BOM + LE bytes directly.
    f.write_bytes(b"\xff\xfe" + text.encode("utf-16-le"))
    out = await read_file.ainvoke({"path": str(f)})
    assert out["content"] == text


async def test_read_file_utf16_be_bom(tmp_path: Path) -> None:
    f = tmp_path / "u16be.txt"
    text = "hello\nworld\n"
    f.write_bytes(b"\xfe\xff" + text.encode("utf-16-be"))
    out = await read_file.ainvoke({"path": str(f)})
    assert out["content"] == text


async def test_read_file_utf8_bom_stripped(tmp_path: Path) -> None:
    f = tmp_path / "u8bom.txt"
    f.write_bytes(b"\xef\xbb\xbfhello\n")
    out = await read_file.ainvoke({"path": str(f)})
    # BOM must NOT appear in the decoded content.
    assert out["content"] == "hello\n"
    assert "﻿" not in out["content"]


def test_read_file_aura_metadata_is_typed() -> None:
    """Phase 2 Task 2 pilot — ``read_file`` ships the typed
    ``ToolMetadata`` (the legacy ``metadata`` dict has been retired
    on this tool; consumers reach the values via
    ``aura.domain.tool_meta_access.meta_dict``). Asserting the
    dataclass type AND each capability flag pins both halves of the
    migration: a future tool that flips off ``aura_metadata`` would
    fail this test before silently degrading to legacy semantics.
    """
    assert isinstance(read_file.aura_metadata, ToolMetadata)
    assert read_file.aura_metadata.is_read_only is True
    assert read_file.aura_metadata.is_destructive is False
    assert read_file.aura_metadata.is_concurrency_safe is True
    assert read_file.aura_metadata.timeout_sec == 10.0
    # ``is_search_command`` lived on the legacy dict; on the typed
    # surface it's promoted to the open-ended capability_flags set
    # (spec §3 keeps the named-field surface narrow). The reader
    # bridge in ``meta_dict`` projects it back to the legacy key.
    assert "search_command" in read_file.aura_metadata.capability_flags


def test_read_file_meta_dict_bridge_exposes_legacy_keys() -> None:
    """The reader bridge must produce the same dict shape consumers
    have always read from, so the loop / hook / CLI sites that go
    through ``meta_dict(read_file)`` see no behavioural change after
    the typed-metadata migration.
    """
    meta = meta_dict(read_file)
    assert meta["is_read_only"] is True
    assert meta["is_destructive"] is False
    assert meta["is_concurrency_safe"] is True
    assert meta["timeout_sec"] == 10.0
    assert meta["is_search_command"] is True
    assert meta["max_result_size_chars"] is None


def test_read_file_metadata_includes_matcher_and_preview() -> None:
    meta = meta_dict(read_file)
    matcher = meta.get("rule_matcher")
    assert callable(matcher)
    # Path-prefix matcher: /tmp covers /tmp/foo but not /tmpfoo.
    assert matcher({"path": "/tmp/foo"}, "/tmp") is True
    assert matcher({"path": "/tmpfoo"}, "/tmp") is False

    preview = meta.get("args_preview")
    assert callable(preview)
    assert preview({"path": "/tmp/a"}) == "path: /tmp/a"


def _five_line_file(tmp_path: Path, name: str = "lines.txt") -> Path:
    f = tmp_path / name
    f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    return f


async def test_read_with_offset_skips_early_lines(tmp_path: Path) -> None:
    f = _five_line_file(tmp_path)
    out = await read_file.ainvoke({"path": str(f), "offset": 2})
    assert out["content"] == "c\nd\ne\n"
    assert out["lines"] == 3
    assert out["total_lines"] == 5
    assert out["offset"] == 2
    assert out["limit"] is None
    assert out["partial"] is True


async def test_read_with_limit_caps_lines(tmp_path: Path) -> None:
    f = _five_line_file(tmp_path)
    out = await read_file.ainvoke({"path": str(f), "limit": 2})
    assert out["content"] == "a\nb\n"
    assert out["lines"] == 2
    assert out["total_lines"] == 5
    assert out["offset"] == 0
    assert out["limit"] == 2
    assert out["partial"] is True


async def test_read_with_offset_and_limit_returns_middle_slice(tmp_path: Path) -> None:
    f = _five_line_file(tmp_path)
    out = await read_file.ainvoke({"path": str(f), "offset": 1, "limit": 2})
    assert out["content"] == "b\nc\n"
    assert out["lines"] == 2
    assert out["total_lines"] == 5
    assert out["offset"] == 1
    assert out["limit"] == 2
    assert out["partial"] is True


async def test_read_full_file_reports_partial_false(tmp_path: Path) -> None:
    f = _five_line_file(tmp_path)
    # Explicit offset=0, limit=None — full read, partial must be False.
    out = await read_file.ainvoke({"path": str(f)})
    assert out["total_lines"] == 5
    assert out["lines"] == 5
    assert out["partial"] is False

    # And: limit large enough to cover the file also counts as non-partial.
    out2 = await read_file.ainvoke({"path": str(f), "limit": 99})
    assert out2["partial"] is False
    assert out2["lines"] == 5


async def test_read_offset_beyond_file_returns_empty(tmp_path: Path) -> None:
    f = _five_line_file(tmp_path)
    out = await read_file.ainvoke({"path": str(f), "offset": 99})
    assert out["content"] == ""
    assert out["lines"] == 0
    assert out["total_lines"] == 5
    assert out["offset"] == 99
    # Honest about over-shooting.
    assert out["partial"] is True


def test_offset_negative_rejected_by_schema() -> None:
    with pytest.raises(ValidationError):
        ReadFileParams.model_validate({"path": "/tmp/x", "offset": -1})


def test_limit_zero_rejected_by_schema() -> None:
    with pytest.raises(ValidationError):
        ReadFileParams.model_validate({"path": "/tmp/x", "limit": 0})


async def test_total_lines_reflects_file_not_slice(tmp_path: Path) -> None:
    f = _five_line_file(tmp_path)
    out = await read_file.ainvoke({"path": str(f), "offset": 3, "limit": 1})
    # Slice has 1 line, but total_lines is the whole file.
    assert out["lines"] == 1
    assert out["total_lines"] == 5


def test_validate_input_rejects_blocked_device() -> None:
    """A blocked-device path resolves to the closed set ⇒ invalid."""
    result = read_file.validate_input({"path": "/dev/stdin"})
    assert isinstance(result, ValidationResult)
    assert result.invalid is True
    assert "device" in result.reason


def test_validate_input_accepts_regular_path(tmp_path: Path) -> None:
    """A normal path (whether or not it exists) is structurally valid;
    existence / decode failures stay on the runtime path.
    """
    result = read_file.validate_input({"path": str(tmp_path / "any.txt")})
    assert result.invalid is False
    assert result.reason == ""


class _UnresolvablePath:
    """Stand-in whose ``resolve`` raises, modelling an OS that refuses to
    canonicalise a hostile path (e.g. ENAMETOOLONG / symlink ELOOP).
    """

    def __init__(self, raised: type[OSError | RuntimeError]) -> None:
        self._raised = raised

    def resolve(self, strict: bool = False) -> Path:
        raise self._raised("cannot resolve")


@pytest.mark.parametrize("raised", [OSError, RuntimeError])
def test_validate_input_failsafe_when_resolve_raises(
    monkeypatch: pytest.MonkeyPatch,
    raised: type[OSError | RuntimeError],
) -> None:
    """If the OS cannot canonicalise the path, the device guard must not
    crash the whole validation pass; it degrades to 'structurally valid'
    so the real not-found/decode error surfaces on the runtime path.
    """
    # The package re-exports the tool instance under the dotted name, so the
    # real module (which owns ``Path``) must be reached via sys.modules.
    module: ModuleType = sys.modules["aura.tools.read_file"]
    monkeypatch.setattr(module, "Path", lambda _path: _UnresolvablePath(raised))
    result = read_file.validate_input({"path": "/dev/stdin"})
    assert result.invalid is False
    assert result.reason == ""


async def test_run_rejects_blocked_device_via_public_invoke() -> None:
    """ainvoke bypasses the validate_input gate, so the runtime guard
    itself must refuse kernel/interactive device endpoints; without it a
    /dev/stdin read would block the agent forever.
    """
    with pytest.raises(ToolError, match="refusing to read"):
        await read_file.ainvoke({"path": "/dev/stdin"})


async def test_run_blocked_device_message_names_device(tmp_path: Path) -> None:
    """The refusal must explain WHY (device endpoint) so the agent can
    self-correct rather than blindly retrying the same blocking path.
    """
    with pytest.raises(ToolError, match="device"):
        await read_file.ainvoke({"path": "/dev/zero"})


def test_validate_input_non_str_path_is_structurally_valid() -> None:
    """A non-string path can't be a known blocked device, so the schema
    layer (not the device guard) owns rejecting it; validate_input must
    not choke on the wrong type and instead defer downstream.
    """
    result = read_file.validate_input({"path": 123})
    assert isinstance(result, ValidationResult)
    assert result.invalid is False
    assert result.reason == ""


async def test_token_budget_overflow_rejected(tmp_path: Path) -> None:
    """A file under the 1 MB byte cap can still blow the ~25k-token
    budget once decoded; the tool must refuse rather than flood the
    model context, steering the caller toward offset+limit slicing.
    """
    f = tmp_path / "verbose.txt"
    # 100,004 two-byte lines ⇒ 200,008 bytes (< 1 MB, no head-truncation)
    # ⇒ 200,008 chars ⇒ ~50,002 tokens, double the 25k budget.
    f.write_bytes(b"a\n" * 100_004)
    with pytest.raises(ToolError, match="too large"):
        await read_file.ainvoke({"path": str(f)})


async def test_token_budget_overflow_is_idempotent(tmp_path: Path) -> None:
    """Refusing an over-budget read is read-only and must stay
    deterministic: a retry of the identical request raises the same
    error, never partially succeeding on the second call.
    """
    f = tmp_path / "verbose2.txt"
    f.write_bytes(b"a\n" * 100_004)
    for _ in range(2):
        with pytest.raises(ToolError, match="too large"):
            await read_file.ainvoke({"path": str(f)})


async def test_token_budget_slicing_escapes_overflow(tmp_path: Path) -> None:
    """The very file that overflows in full must read fine once sliced —
    proving offset+limit is the documented escape hatch, not just that
    the cap fires.
    """
    f = tmp_path / "verbose3.txt"
    f.write_bytes(b"a\n" * 100_004)
    out = await read_file.ainvoke({"path": str(f), "limit": 5})
    assert out["content"] == "a\n" * 5
    assert out["lines"] == 5
    assert out["total_lines"] == 100_004
    assert out["partial"] is True
