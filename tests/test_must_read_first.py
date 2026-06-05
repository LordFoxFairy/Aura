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

from aura.application.hooks.must_read_first import make_must_read_first_hook
from aura.application.loop_state import LoopState
from aura.application.memory.context import Context
from aura.application.memory.rules_types import RulesBundle
from aura.domain.permission.outcome import Replace
from aura.domain.tool import ToolResult
from aura.infrastructure.persistence import journal as journal_module
from aura.tools.base import build_tool


def _sc(outcome: object) -> ToolResult | None:
    """Extract the short-circuit ToolResult from either Replace (Task 9+)
    Keeps existing test assertions concise while
    supporting both shapes during the migration window."""
    if isinstance(outcome, Replace):
        return outcome.result
    sc = getattr(outcome, "short_circuit", None)
    return sc


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

    outcome = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "hello", "new_str": "bye"},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False
    assert sc.error is not None
    assert "has not been read" in sc.error


@pytest.mark.asyncio
async def test_edit_file_allowed_after_record_read(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    ctx.record_read(target)

    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "hello", "new_str": "bye"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_edit_file_rejected_when_different_path_read(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A\n")
    b.write_text("B\n")
    ctx.record_read(a)

    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_edit_tool(),
        args={"path": str(b), "old_str": "B", "new_str": "C"},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False


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
    assert _sc(r1) is None
    assert _sc(r3) is None
    assert _sc(r4) is None


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
    outcome = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "pass", "new_str": "return"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_non_existent_path_blocks_as_never_read(tmp_path: Path) -> None:
    # record_read on a non-existent path is silent (fail-soft stat); nothing
    # gets recorded, so read_status stays "never_read" and the edit is blocked
    # — mirrors claude-code's guarantee: never edit what you haven't read.
    ctx = _ctx(tmp_path)
    ghost = tmp_path / "ghost.txt"
    ctx.record_read(ghost)

    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_edit_tool(),
        args={"path": str(ghost), "old_str": "x", "new_str": "y"},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False
    assert sc.error is not None
    assert "has not been read" in sc.error


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
    outcome = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "hello", "new_str": "bye"},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False
    assert sc.error is not None
    assert "has changed since last read" in sc.error


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
    sc_never = _sc(r_never)
    assert sc_never is not None
    assert sc_never.error is not None
    assert "has not been read" in sc_never.error

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
    sc_stale = _sc(r_stale)
    assert sc_stale is not None
    assert sc_stale.error is not None
    assert "has changed since last read" in sc_stale.error

    assert sc_never.error != sc_stale.error


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
    outcome = await hook(
        tool=_edit_tool(),
        args={"path": str(ghost), "old_str": "", "new_str": "hello\n"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_hook_still_blocks_edit_with_old_str_on_never_read_file(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")

    outcome = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "hello", "new_str": "bye"},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False
    assert sc.error is not None
    assert "has not been read" in sc.error


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
        outcome = await hook(
            tool=_edit_tool(),
            args={"path": str(target), "old_str": "a", "new_str": "A"},
            state=LoopState(),
        )
        sc = _sc(outcome)
        assert sc is not None
        assert sc.ok is False
        assert sc.error is not None
        assert "partially read" in sc.error

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
    outcome = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "a", "new_str": "A"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


# write_file — file-unchanged guard mirroring claude-code's FileWriteTool.
# Only applies when the target already exists on disk; pure creation is free.


@pytest.mark.asyncio
async def test_write_file_to_new_path_passes_through_hook(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    ghost = tmp_path / "brand_new.txt"  # does NOT exist

    outcome = await hook(
        tool=_write_tool(),
        args={"path": str(ghost)},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_write_file_overwrite_rejected_without_prior_read(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")

    outcome = await hook(
        tool=_write_tool(),
        args={"path": str(target)},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False
    assert sc.error is not None
    assert "has not been read" in sc.error
    assert "overwriting" in sc.error


@pytest.mark.asyncio
async def test_write_file_overwrite_allowed_after_record_read(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    ctx.record_read(target)

    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_write_tool(),
        args={"path": str(target)},
        state=LoopState(),
    )
    assert _sc(outcome) is None


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
    outcome = await hook(
        tool=_write_tool(),
        args={"path": str(target)},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False
    assert sc.error is not None
    assert "has changed since last read" in sc.error


@pytest.mark.asyncio
async def test_write_file_overwrite_rejected_when_partial(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("a\nb\nc\n")
    ctx.record_read(target, partial=True)

    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_write_tool(),
        args={"path": str(target)},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False
    assert sc.error is not None
    assert "partially read" in sc.error


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

    outcome = await hook(
        tool=_write_tool(),
        args={"path": str(target)},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.error is not None
    assert "overwriting" in sc.error
    assert "before edit" not in sc.error


@pytest.mark.asyncio
async def test_never_read_message_no_duplicated_path(tmp_path: Path) -> None:
    # Regression guard: the never_read error message must cite the path
    # exactly once. Prior version duplicated it ("read_file({p}) ... (path={p})").
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)

    for tool_factory in (_edit_tool, _write_tool):
        target = tmp_path / f"f_{tool_factory.__name__}.txt"
        target.write_text("hello\n")
        outcome = await hook(
            tool=tool_factory(),
            args={"path": str(target), "old_str": "hello", "new_str": "bye"}
            if tool_factory is _edit_tool
            else {"path": str(target)},
            state=LoopState(),
        )
        sc = _sc(outcome)
        assert sc is not None
        assert sc.error is not None
        assert sc.error.count(str(target.resolve())) == 1, (
            f"path appeared {sc.error.count(str(target.resolve()))}x in: "
            f"{sc.error!r}"
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


@pytest.mark.asyncio
async def test_bash_sed_in_place_blocked_when_target_unread(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    target = tmp_path / "config.toml"
    target.write_text("k = 1\n")
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": f"sed -i 's/1/2/' {target}"},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False
    assert "would mutate" in (sc.error or "")
    assert str(target.resolve()) in (sc.error or "")


@pytest.mark.asyncio
async def test_bash_sed_in_place_allowed_after_read(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = tmp_path / "config.toml"
    target.write_text("k = 1\n")
    ctx.record_read(target)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": f"sed -i 's/1/2/' {target}"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_redirect_overwrite_blocked_when_target_unread(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    target = tmp_path / "out.txt"
    target.write_text("old\n")
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": f"echo new > {target}"},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False


@pytest.mark.parametrize(
    "template",
    ["echo new >{path}", "echo new 2>{path}", "echo new 1>>{path}"],
)
@pytest.mark.asyncio
async def test_bash_compact_redirect_blocked_when_target_unread(
    tmp_path: Path,
    template: str,
) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    target = tmp_path / "compact.txt"
    target.write_text("old\n")
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": template.format(path=target)},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False


@pytest.mark.asyncio
async def test_bash_redirect_to_new_file_passes(tmp_path: Path) -> None:
    """``> path`` to a non-existent target is pure creation — no prior read needed."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    target = tmp_path / "fresh.txt"  # does not exist
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": f"echo hi > {target}"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_append_redirect_blocked_when_target_unread(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    target = tmp_path / "log.txt"
    target.write_text("entry-1\n")
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": f"echo entry-2 >> {target}"},
        state=LoopState(),
    )
    assert _sc(outcome) is not None


@pytest.mark.asyncio
async def test_bash_tee_blocked_when_target_unread(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    target = tmp_path / "shared.txt"
    target.write_text("v0\n")
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": f"echo v1 | tee {target}"},
        state=LoopState(),
    )
    assert _sc(outcome) is not None


@pytest.mark.asyncio
async def test_bash_redirect_to_dev_null_passes(tmp_path: Path) -> None:
    """``> /dev/null`` should not flag — /dev/* is excluded from the regex."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "ls /tmp > /dev/null 2>&1"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_pure_read_command_passes(tmp_path: Path) -> None:
    """No mutation idiom in command → hook passthrough."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "ls -la /tmp"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_journal_emits_blocked_event_with_command(
    tmp_path: Path,
) -> None:
    log = tmp_path / "journal.jsonl"
    journal_module.configure(log)
    try:
        ctx = _ctx(tmp_path)
        hook = make_must_read_first_hook(ctx)
        target = tmp_path / "cfg.toml"
        target.write_text("k=1\n")
        cmd = f"sed -i 's/1/2/' {target}"
        await hook(
            tool=_bash_tool(),
            args={"command": cmd},
            state=LoopState(),
        )
        events = [json.loads(line) for line in log.read_text().splitlines()]
        blocked = [e for e in events if e["event"] == "must_read_first_blocked"]
        assert len(blocked) == 1
        assert blocked[0]["tool"] == "bash"
        assert blocked[0]["path"] == str(target.resolve())
        assert blocked[0]["command"] == cmd
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_edit_file_unread_returns_replace_outcome(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("content")
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    tool = _edit_tool()
    outcome = await hook(
        tool=tool,
        args={"path": str(target), "old_str": "content", "new_str": "new"},
        state=LoopState(),
    )
    assert isinstance(outcome, Replace), f"expected Replace, got {type(outcome).__name__}"
    assert outcome.result.ok is False
    assert "has not been read" in (outcome.result.error or "")
    assert outcome.decision.allow is False
    assert outcome.decision.reason == "safety_blocked"


@pytest.mark.asyncio
async def test_write_file_unread_existing_returns_replace_outcome(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("content")
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    tool = _write_tool()
    outcome = await hook(
        tool=tool,
        args={"path": str(target), "content": "new content"},
        state=LoopState(),
    )
    assert isinstance(outcome, Replace)
    assert outcome.result.ok is False
    assert outcome.decision.allow is False
    assert outcome.decision.reason == "safety_blocked"


@pytest.mark.asyncio
async def test_subagent_inherited_fresh_read_allows_edit(tmp_path: Path) -> None:
    """Parent reads f.py → spawn subagent → file unchanged → subagent edit allowed.

    Pins the happy-path baseline for Task 5: an inherited fresh record
    behaves exactly like a live fresh record in the child's hook.
    """
    from aura.domain.state_values import ReadCarryover, ReadRecord

    target = tmp_path / "f.py"
    target.write_text("hello\n")
    st = target.stat()

    parent_record = ReadRecord(
        path=target.resolve(),
        mtime_at_read=st.st_mtime,
        size_at_read=st.st_size,
        read_at_turn=1,
    )
    carryover = ReadCarryover(
        records={target.resolve(): parent_record},
        source_session_id="parent-1",
        generated_at_turn=5,
    )

    sub_ctx = Context(
        cwd=tmp_path,
        system_prompt="",
        primary_memory="",
        rules=RulesBundle(),
        carryover=carryover,
    )
    hook = make_must_read_first_hook(sub_ctx)

    outcome = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "hello", "new_str": "bye"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_subagent_inherited_stale_read_is_blocked(tmp_path: Path) -> None:
    """Parent reads f.py at turn 1 → subagent spawned at turn 5 → external
    process modifies f.py mtime/size → subagent edit is BLOCKED.

    Phase 3 Task 5 acceptance: inherited reads honour ``ReadCarryover``-style
    staleness. Concretely, the subagent's seeded ``_ReadRecord`` carries the
    parent's recorded ``(mtime, size)`` fingerprint; ``Context.read_status``
    re-stats at hook-fire time and surfaces ``"stale"`` when the file
    drifted on disk. Functionally equivalent to ``carryover.is_fresh(path)``
    being False.
    """
    from aura.domain.state_values import ReadCarryover, ReadRecord

    target = tmp_path / "f.py"
    target.write_text("hello\n")
    st = target.stat()

    parent_record = ReadRecord(
        path=target.resolve(),
        mtime_at_read=st.st_mtime,
        size_at_read=st.st_size,
        read_at_turn=1,
    )
    carryover = ReadCarryover(
        records={target.resolve(): parent_record},
        source_session_id="parent-1",
        generated_at_turn=5,
    )

    # Build the subagent's Context from the carryover BEFORE the external
    # mutation so the seeding happens against the parent's pristine view.
    sub_ctx = Context(
        cwd=tmp_path,
        system_prompt="",
        primary_memory="",
        rules=RulesBundle(),
        carryover=carryover,
    )

    # External process modifies the file AFTER spawn but BEFORE the
    # subagent attempts an edit — this is the staleness scenario we
    # need the hook to catch.
    target.write_text("hello world! longer now\n")
    st2 = target.stat()
    os.utime(target, (st2.st_mtime + 10, st2.st_mtime + 10))

    hook = make_must_read_first_hook(sub_ctx)
    outcome = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "hello", "new_str": "bye"},
        state=LoopState(),
    )
    assert isinstance(outcome, Replace)
    assert outcome.result.ok is False
    # Staleness manifests as the "has changed since last read" branch
    # (not the never_read branch) — same wording as a live stale read,
    # which is the contract: inherited and live reads share one gate.
    assert "has changed since last read" in (outcome.result.error or "")
    assert outcome.decision.allow is False
    assert outcome.decision.reason == "safety_blocked"


@pytest.mark.asyncio
async def test_subagent_inherited_read_blocked_when_file_deleted(
    tmp_path: Path,
) -> None:
    """Parent read f.py → carryover seeds subagent → file deleted out-of-band
    → subagent edit blocked. ``read_status`` returns ``"stale"`` on missing
    file (matching ``ReadCarryover.is_fresh`` behavior on missing path).
    """
    from aura.domain.state_values import ReadCarryover, ReadRecord

    target = tmp_path / "f.py"
    target.write_text("hello\n")
    st = target.stat()
    parent_record = ReadRecord(
        path=target.resolve(),
        mtime_at_read=st.st_mtime,
        size_at_read=st.st_size,
        read_at_turn=1,
    )
    carryover = ReadCarryover(
        records={target.resolve(): parent_record},
        source_session_id="parent-1",
        generated_at_turn=5,
    )
    sub_ctx = Context(
        cwd=tmp_path,
        system_prompt="",
        primary_memory="",
        rules=RulesBundle(),
        carryover=carryover,
    )

    target.unlink()

    hook = make_must_read_first_hook(sub_ctx)
    outcome = await hook(
        tool=_edit_tool(),
        args={"path": str(target), "old_str": "hello", "new_str": "bye"},
        state=LoopState(),
    )
    assert isinstance(outcome, Replace)
    assert outcome.result.ok is False
    assert outcome.decision.allow is False


# ---------------------------------------------------------------------------
# Appended boundary tests: uncovered branches in the read-gating decision,
# bash-target extraction, and path normalization. No real TTY/subprocess —
# all seams are exercised via the in-process hook and a monkeypatched
# ``Path.resolve`` where an OSError must be simulated.
# ---------------------------------------------------------------------------


def _raise_oserror_resolve(_self: Path, strict: bool = False) -> Path:
    """Stand-in for ``Path.resolve`` that always fails — feeds the OSError seam."""
    raise OSError("simulated resolve failure")


@pytest.mark.parametrize("bad_command", [None, "", 0, [], {}])
@pytest.mark.asyncio
async def test_bash_non_string_or_empty_command_bypasses(
    tmp_path: Path,
    bad_command: object,
) -> None:
    """A bash call whose ``command`` arg is missing/empty/non-str cannot be
    parsed for mutation targets, so the gate must fail-open (bypass) rather
    than crash — schema-crash resilience on the tool-arg boundary."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": bad_command},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_missing_command_key_bypasses(tmp_path: Path) -> None:
    """A bash args dict with no ``command`` key at all (args.get → None) must
    bypass cleanly — the gate never assumes the key is present."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_target_resolve_oserror_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If resolving a detected bash mutation target raises OSError (e.g. a
    path the OS refuses to canonicalise), that target is silently skipped so
    one un-resolvable token cannot wedge the whole command — fail-open at the
    resolve seam, not a hard crash."""
    ctx = _ctx(tmp_path)
    target = tmp_path / "cfg.toml"
    target.write_text("k = 1\n")
    hook = make_must_read_first_hook(ctx)
    monkeypatch.setattr(Path, "resolve", _raise_oserror_resolve)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": f"sed -i 's/1/2/' {target}"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_fd_dup_redirect_is_not_a_mutation_target(
    tmp_path: Path,
) -> None:
    """``2>&1`` duplicates a file descriptor — it is NOT a file write, so it
    must never be treated as a mutation target (the ``&`` suffix guard). A
    pure ``cmd 2>&1`` with no real redirect target stays a passthrough."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "grep x file 2>&1"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_unbalanced_quote_segment_is_skipped(tmp_path: Path) -> None:
    """A command segment with an unbalanced quote makes ``shlex.split`` raise
    ValueError; that segment must be skipped (not crash the gate) so a
    syntactically broken command degrades gracefully to passthrough."""
    ctx = _ctx(tmp_path)
    target = tmp_path / "out.txt"
    target.write_text("old\n")
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "echo 'unterminated"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_empty_segments_between_separators_are_skipped(
    tmp_path: Path,
) -> None:
    """Consecutive separators (``;;``) yield empty segments that tokenize to
    [] and must be skipped without error — the splitter is robust to malformed
    chaining."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "ls ;; pwd"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_leading_separator_yields_empty_segment_skipped(
    tmp_path: Path,
) -> None:
    """A command that starts with a separator (``; ls``) splits into an empty
    leading segment that tokenizes to []; that segment must be skipped without
    error — leading/dangling separators degrade to passthrough."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "; ls"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_fd_prefixed_dangling_redirect_at_end_is_ignored(
    tmp_path: Path,
) -> None:
    """A trailing fd-prefixed redirect token (``2>``) with no following file
    (compact form, empty suffix, last token) has no real target; the gate must
    not index past the token list — malformed fd redirect is treated as no
    mutation, never an IndexError."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "echo hi 2>"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_fd_prefixed_spaced_redirect_to_unread_file_blocks(
    tmp_path: Path,
) -> None:
    """A fd-prefixed redirect with a spaced filename (``cmd 2> file``) writes
    that file; when the target was never read it must be blocked exactly like a
    plain ``> file`` redirect — fd-numbered stderr/stdout redirects are real
    writes, not exempt."""
    ctx = _ctx(tmp_path)
    target = tmp_path / "err.log"
    target.write_text("old\n")
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": f"run_thing 2> {target}"},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False
    assert "would mutate" in (sc.error or "")


@pytest.mark.asyncio
async def test_bash_dangling_redirect_operator_at_end_is_ignored(
    tmp_path: Path,
) -> None:
    """A bare ``>`` as the final token has no following filename; the gate must
    not index past the token list — a malformed redirect is treated as no
    mutation, never an IndexError."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "echo hi >"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_compact_redirect_to_dev_is_ignored(tmp_path: Path) -> None:
    """A compact ``>/dev/null`` token (operator+suffix fused) targets a device
    node, which must be excluded just like the spaced ``> /dev/null`` form —
    device sinks are never gated."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "ls >/dev/null"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_sed_long_inplace_flag_is_blocked(tmp_path: Path) -> None:
    """The GNU long form ``sed --in-place`` must be recognised as an in-place
    mutation exactly like ``-i`` — otherwise the gate is trivially bypassed by
    spelling the flag out."""
    ctx = _ctx(tmp_path)
    target = tmp_path / "cfg.toml"
    target.write_text("k = 1\n")
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": f"sed --in-place 's/1/2/' {target}"},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False
    assert "would mutate" in (sc.error or "")


@pytest.mark.asyncio
async def test_bash_sed_inplace_with_suffix_value_is_blocked(
    tmp_path: Path,
) -> None:
    """``sed --in-place=.bak`` (backup-suffix form) is still an in-place edit
    of the target and must be blocked when the file was never read."""
    ctx = _ctx(tmp_path)
    target = tmp_path / "cfg.toml"
    target.write_text("k = 1\n")
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": f"sed --in-place=.bak 's/1/2/' {target}"},
        state=LoopState(),
    )
    sc = _sc(outcome)
    assert sc is not None
    assert sc.ok is False
    assert "would mutate" in (sc.error or "")


@pytest.mark.asyncio
async def test_bash_sed_long_non_inplace_flag_does_not_block(
    tmp_path: Path,
) -> None:
    """A non-mutating GNU long flag such as ``--quiet`` must NOT be mistaken
    for in-place editing; without ``-i`` the sed command reads-only and the
    gate stays a passthrough."""
    ctx = _ctx(tmp_path)
    target = tmp_path / "cfg.toml"
    target.write_text("k = 1\n")
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": f"sed --quiet 's/1/2/' {target}"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_sed_inplace_with_no_file_argument_is_ignored(
    tmp_path: Path,
) -> None:
    """``sed -i`` whose only remaining tokens are options (no filename) yields
    no extractable target; the gate must treat it as no mutation rather than
    crash when the last-non-option lookup returns None."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "sed -i -n"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.parametrize("bad_path", [None, "", 0, [], {}])
@pytest.mark.asyncio
async def test_edit_file_non_string_or_empty_path_bypasses(
    tmp_path: Path,
    bad_path: object,
) -> None:
    """An edit_file call whose ``path`` arg is missing/empty/non-str cannot be
    gated, so the hook fails open instead of raising — schema-crash resilience
    on the path boundary."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_edit_tool(),
        args={"path": bad_path, "old_str": "a", "new_str": "b"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_write_file_missing_path_key_bypasses(tmp_path: Path) -> None:
    """A write_file args dict without a ``path`` key (args.get → None) must
    bypass cleanly rather than crash — the gate never assumes the key exists."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    outcome = await hook(
        tool=_write_tool(),
        args={},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_edit_file_path_resolve_oserror_bypasses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If canonicalising the edit target raises OSError, the gate cannot decide
    safely and must fail open (bypass) instead of propagating — mirrors the
    bash-target resolve seam for the file tools."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    monkeypatch.setattr(Path, "resolve", _raise_oserror_resolve)
    outcome = await hook(
        tool=_edit_tool(),
        args={"path": "/some/path", "old_str": "a", "new_str": "b"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_write_file_path_resolve_oserror_bypasses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """write_file shares the same resolve seam as edit_file: an OSError while
    canonicalising the target must bypass, never crash the hook."""
    ctx = _ctx(tmp_path)
    hook = make_must_read_first_hook(ctx)
    monkeypatch.setattr(Path, "resolve", _raise_oserror_resolve)
    outcome = await hook(
        tool=_write_tool(),
        args={"path": "/some/path"},
        state=LoopState(),
    )
    assert _sc(outcome) is None


@pytest.mark.asyncio
async def test_bash_block_is_idempotent_across_repeated_calls(
    tmp_path: Path,
) -> None:
    """Firing the same unread-target bash mutation twice must yield the same
    block both times — the gate holds no per-call state that could let a
    second attempt slip through."""
    ctx = _ctx(tmp_path)
    target = tmp_path / "cfg.toml"
    target.write_text("k = 1\n")
    hook = make_must_read_first_hook(ctx)
    cmd = f"sed -i 's/1/2/' {target}"
    first = await hook(
        tool=_bash_tool(), args={"command": cmd}, state=LoopState(),
    )
    second = await hook(
        tool=_bash_tool(), args={"command": cmd}, state=LoopState(),
    )
    sc1 = _sc(first)
    sc2 = _sc(second)
    assert sc1 is not None and sc2 is not None
    assert sc1.ok is False and sc2.ok is False
    assert sc1.error == sc2.error
