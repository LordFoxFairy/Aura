"""Tests for the must-read-first invariant on edit_file.

Mirrors claude-code's FileEditTool.ts:275–287: before an edit, the file MUST
have been read in the same session, else the hook short-circuits with a
ToolResult(ok=False). The enforcement is a PreToolHook closure over a
``Context`` reference; ``Context`` records successful read_file paths via
``record_read`` (called from AgentLoop._maybe_trigger_path).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from aura.core.hooks.must_read_first import make_must_read_first_hook
from aura.core.memory.context import Context
from aura.core.memory.rules import RulesBundle
from aura.core.persistence import journal as journal_module
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool


class _PathOnly(BaseModel):
    path: str


class _PathOldNew(BaseModel):
    path: str
    old_str: str
    new_str: str


class _BashArgs(BaseModel):
    command: str


class _GrepArgs(BaseModel):
    pattern: str
    path: str


def _edit_tool() -> Any:
    return build_tool(
        name="edit_file",
        description="edit",
        args_schema=_PathOldNew,
        func=lambda path, old_str, new_str: {"replacements": 1},
        is_destructive=True,
    )


def _read_tool() -> Any:
    return build_tool(
        name="read_file",
        description="read",
        args_schema=_PathOnly,
        func=lambda path: "content",
        is_read_only=True,
    )


def _write_tool() -> Any:
    return build_tool(
        name="write_file",
        description="write",
        args_schema=_PathOnly,
        func=lambda path: {"bytes": 0},
        is_destructive=True,
    )


def _bash_tool() -> Any:
    return build_tool(
        name="bash",
        description="bash",
        args_schema=_BashArgs,
        func=lambda command: "",
    )


def _grep_tool() -> Any:
    return build_tool(
        name="grep",
        description="grep",
        args_schema=_GrepArgs,
        func=lambda pattern, path: [],
        is_read_only=True,
    )


def _ctx(tmp_path: Path) -> Context:
    return Context(
        cwd=tmp_path,
        system_prompt="",
        primary_memory="",
        rules=RulesBundle(),
    )


@pytest.mark.asyncio
async def test_edit_file_rejected_without_prior_read(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")

    result = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "hello", "new_str": "bye"},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "has not been read" in result.error


@pytest.mark.asyncio
async def test_edit_file_allowed_after_record_read(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    ctx.record_read(target)

    hook = make_must_read_first_hook(ctx)
    result = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "hello", "new_str": "bye"},
        state=LoopState(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_edit_file_rejected_when_different_path_read(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A\n")
    b.write_text("B\n")
    ctx.record_read(a)

    hook = make_must_read_first_hook(ctx)
    result = await hook(
        tool=_edit_tool(),
        args={"path": str(b), "old_str": "B", "new_str": "C"},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False


@pytest.mark.asyncio
async def test_other_tools_pass_through_unaffected(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)  # nothing recorded
    hook = make_must_read_first_hook(ctx)

    target = tmp_path / "f.txt"
    target.write_text("x")

    r1 = await hook(
        tool=_bash_tool(), args={"command": "ls"}, state=LoopState(),
    )
    r3 = await hook(
        tool=_grep_tool(),
        args={"pattern": "x", "path": str(target)},
        state=LoopState(),
    )
    r4 = await hook(
        tool=_read_tool(), args={"path": str(target)}, state=LoopState(),
    )
    assert r1 is None
    assert r3 is None
    assert r4 is None


@pytest.mark.asyncio
async def test_relative_and_absolute_paths_normalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ctx = _ctx(tmp_path)
    target = tmp_path / "foo.py"
    target.write_text("pass\n")

    ctx.record_read(Path("foo.py"))

    hook = make_must_read_first_hook(ctx)
    result = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "pass", "new_str": "return"},
        state=LoopState(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_non_existent_path_blocks_as_never_read(tmp_path: Path) -> None:
    # record_read on a non-existent path is silent (fail-soft stat); nothing
    # gets recorded, so read_status stays "never_read" and the edit is blocked
    # — mirrors claude-code's guarantee: never edit what you haven't read.
    ctx = _ctx(tmp_path)
    ghost = tmp_path / "ghost.txt"
    ctx.record_read(ghost)

    hook = make_must_read_first_hook(ctx)
    result = await hook(
        tool=_edit_tool(),
        args={"path": str(ghost), "old_str": "x", "new_str": "y"},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "has not been read" in result.error


@pytest.mark.asyncio
async def test_journal_event_on_block(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        ctx = _ctx(tmp_path)
        target = tmp_path / "f.txt"
        target.write_text("hello\n")
        hook = make_must_read_first_hook(ctx)
        await hook(
            tool=_edit_tool(),
            args={"path": str(target), "old_str": "hello", "new_str": "bye"},
            state=LoopState(),
        )
        events = [json.loads(line) for line in log.read_text().splitlines()]
        blocked = [e for e in events if e["event"] == "must_read_first_blocked"]
        assert len(blocked) == 1
        assert blocked[0]["tool"] == "edit_file"
        assert blocked[0]["path"] == str(target.resolve())
        assert blocked[0]["reason"] == "never_read"
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_edit_file_rejected_when_file_changed_since_read(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    ctx.record_read(target)

    # Mutate contents AND bump mtime to make sure at least one of the two
    # signals (mtime, size) differs regardless of FS mtime resolution.
    target.write_text("hello world! longer now\n")
    st = target.stat()
    os.utime(target, (st.st_mtime + 10, st.st_mtime + 10))

    hook = make_must_read_first_hook(ctx)
    result = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "hello", "new_str": "bye"},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "has changed since last read" in result.error


@pytest.mark.asyncio
async def test_stale_and_never_read_errors_are_distinct(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)

    never = tmp_path / "never.txt"
    never.write_text("x\n")
    r_never = await hook(
        tool=_edit_tool(),
        args={"path": str(never), "old_str": "x", "new_str": "y"},
        state=LoopState(),
    )
    assert isinstance(r_never, ToolResult)
    assert r_never.error is not None
    assert "has not been read" in r_never.error

    stale = tmp_path / "stale.txt"
    stale.write_text("x\n")
    ctx.record_read(stale)
    stale.write_text("xy\n")
    st = stale.stat()
    os.utime(stale, (st.st_mtime + 10, st.st_mtime + 10))

    r_stale = await hook(
        tool=_edit_tool(),
        args={"path": str(stale), "old_str": "x", "new_str": "y"},
        state=LoopState(),
    )
    assert isinstance(r_stale, ToolResult)
    assert r_stale.error is not None
    assert "has changed since last read" in r_stale.error

    assert r_never.error != r_stale.error


def test_fresh_after_record_returns_fresh_status(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    p = tmp_path / "f.txt"
    p.write_text("x\n")
    ctx.record_read(p)
    assert ctx.read_status(p) == "fresh"


def test_read_status_stale_after_size_change(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    p = tmp_path / "f.txt"
    p.write_text("x")
    ctx.record_read(p)

    # Preserve mtime while changing size — forces the size branch of staleness.
    st_before = p.stat()
    p.write_text("xyz")
    os.utime(p, (st_before.st_mtime, st_before.st_mtime))

    assert ctx.read_status(p) == "stale"


def test_read_status_stale_after_mtime_bump(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    p = tmp_path / "f.txt"
    p.write_text("xyz")
    ctx.record_read(p)

    st = p.stat()
    os.utime(p, (st.st_mtime + 10, st.st_mtime + 10))

    assert ctx.read_status(p) == "stale"


def test_read_status_never_read_for_untracked_path(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    p = tmp_path / "nope.txt"
    p.write_text("x\n")
    assert ctx.read_status(p) == "never_read"


def test_record_read_swallows_missing_path(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ghost = tmp_path / "ghost.txt"

    ctx.record_read(ghost)  # must not raise
    assert ctx.read_status(ghost) == "never_read"


def test_recorded_then_deleted_path_returns_stale(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    p = tmp_path / "f.txt"
    p.write_text("x\n")
    ctx.record_read(p)

    p.unlink()
    assert ctx.read_status(p) == "stale"


@pytest.mark.asyncio
async def test_journal_event_reason_is_stale(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        ctx = _ctx(tmp_path)
        target = tmp_path / "f.txt"
        target.write_text("hello\n")
        ctx.record_read(target)
        target.write_text("hello world!\n")
        st = target.stat()
        os.utime(target, (st.st_mtime + 10, st.st_mtime + 10))

        hook = make_must_read_first_hook(ctx)
        await hook(
            tool=_edit_tool(),
            args={"path": str(target), "old_str": "hello", "new_str": "bye"},
            state=LoopState(),
        )
        events = [json.loads(line) for line in log.read_text().splitlines()]
        blocked = [e for e in events if e["event"] == "must_read_first_blocked"]
        assert len(blocked) == 1
        assert blocked[0]["reason"] == "stale"
    finally:
        journal_module.reset()


def test_context_record_read_is_idempotent(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    p = tmp_path / "f.txt"
    p.write_text("x")

    ctx.record_read(p)
    ctx.record_read(p)
    ctx.record_read(p)

    assert ctx.read_status(p) == "fresh"
    assert len(ctx._read_records) == 1


@pytest.mark.asyncio
async def test_hook_allows_new_file_creation_via_empty_old_str(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    ghost = tmp_path / "brand_new.txt"  # does NOT exist
    result = await hook(
        tool=_edit_tool(),
        args={"path": str(ghost), "old_str": "", "new_str": "hello\n"},
        state=LoopState(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_hook_still_blocks_edit_with_old_str_on_never_read_file(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")

    result = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "hello", "new_str": "bye"},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "has not been read" in result.error


@pytest.mark.asyncio
async def test_partial_read_blocks_edit_with_partial_reason(
    tmp_path: Path,
) -> None:
    # A partial read (offset>0 or limit truncated below total_lines) must
    # block edit_file — the model hasn't seen the whole file, so any edit
    # is based on an incomplete view.
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        ctx = _ctx(tmp_path)
        target = tmp_path / "f.txt"
        target.write_text("a\nb\nc\n")
        ctx.record_read(target, partial=True)

        assert ctx.read_status(target) == "partial"

        hook = make_must_read_first_hook(ctx)
        result = await hook(
            tool=_edit_tool(),
            args={"path": str(target), "old_str": "a", "new_str": "A"},
            state=LoopState(),
        )
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert result.error is not None
        assert "partially read" in result.error

        events = [json.loads(line) for line in log.read_text().splitlines()]
        blocked = [e for e in events if e["event"] == "must_read_first_blocked"]
        assert len(blocked) == 1
        assert blocked[0]["reason"] == "partial"
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_full_read_after_partial_read_recovers_fresh(
    tmp_path: Path,
) -> None:
    # A subsequent FULL read (partial=False) must overwrite the partial
    # record, flipping status back to "fresh" and unblocking edits.
    ctx = _ctx(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("a\nb\nc\n")
    ctx.record_read(target, partial=True)
    assert ctx.read_status(target) == "partial"

    ctx.record_read(target, partial=False)
    assert ctx.read_status(target) == "fresh"

    hook = make_must_read_first_hook(ctx)
    result = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "a", "new_str": "A"},
        state=LoopState(),
    )
    assert result is None


# write_file — file-unchanged guard mirroring claude-code's FileWriteTool.
# Only applies when the target already exists on disk; pure creation is free.


@pytest.mark.asyncio
async def test_write_file_to_new_path_passes_through_hook(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    ghost = tmp_path / "brand_new.txt"  # does NOT exist

    result = await hook(
        tool=_write_tool(),
        args={"path": str(ghost)},
        state=LoopState(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_write_file_overwrite_rejected_without_prior_read(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")

    result = await hook(
        tool=_write_tool(),
        args={"path": str(target)},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "has not been read" in result.error
    assert "overwriting" in result.error


@pytest.mark.asyncio
async def test_write_file_overwrite_allowed_after_record_read(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    ctx.record_read(target)

    hook = make_must_read_first_hook(ctx)
    result = await hook(
        tool=_write_tool(),
        args={"path": str(target)},
        state=LoopState(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_write_file_overwrite_rejected_when_stale(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    ctx.record_read(target)

    target.write_text("hello world! longer now\n")
    st = target.stat()
    os.utime(target, (st.st_mtime + 10, st.st_mtime + 10))

    hook = make_must_read_first_hook(ctx)
    result = await hook(
        tool=_write_tool(),
        args={"path": str(target)},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "has changed since last read" in result.error


@pytest.mark.asyncio
async def test_write_file_overwrite_rejected_when_partial(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("a\nb\nc\n")
    ctx.record_read(target, partial=True)

    hook = make_must_read_first_hook(ctx)
    result = await hook(
        tool=_write_tool(),
        args={"path": str(target)},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "partially read" in result.error


@pytest.mark.asyncio
async def test_write_file_error_messages_say_overwriting_not_editing(
    tmp_path: Path,
) -> None:
    # Regression guard: the per-tool error message must mention "overwriting",
    # not the edit-flavored "before edit".
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")

    result = await hook(
        tool=_write_tool(),
        args={"path": str(target)},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.error is not None
    assert "overwriting" in result.error
    assert "before edit" not in result.error


@pytest.mark.asyncio
async def test_never_read_message_no_duplicated_path(tmp_path: Path) -> None:
    # Regression guard: the never_read error message must cite the path
    # exactly once. Prior version duplicated it ("read_file({p}) ... (path={p})").
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)

    for tool_factory in (_edit_tool, _write_tool):
        target = tmp_path / f"f_{tool_factory.__name__}.txt"
        target.write_text("hello\n")
        result = await hook(
            tool=tool_factory(),
            args={"path": str(target), "old_str": "hello", "new_str": "bye"}
            if tool_factory is _edit_tool
            else {"path": str(target)},
            state=LoopState(),
        )
        assert isinstance(result, ToolResult)
        assert result.error is not None
        assert result.error.count(str(target.resolve())) == 1, (
            f"path appeared {result.error.count(str(target.resolve()))}x in: "
            f"{result.error!r}"
        )


@pytest.mark.asyncio
async def test_journal_event_shows_write_file_as_tool(
    tmp_path: Path,
) -> None:
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        ctx = _ctx(tmp_path)
        target = tmp_path / "f.txt"
        target.write_text("hello\n")
        hook = make_must_read_first_hook(ctx)
        await hook(
            tool=_write_tool(),
            args={"path": str(target)},
            state=LoopState(),
        )
        events = [json.loads(line) for line in log.read_text().splitlines()]
        blocked = [e for e in events if e["event"] == "must_read_first_blocked"]
        assert len(blocked) == 1
        assert blocked[0]["tool"] == "write_file"
        assert blocked[0]["path"] == str(target.resolve())
        assert blocked[0]["reason"] == "never_read"
    finally:
        journal_module.reset()
