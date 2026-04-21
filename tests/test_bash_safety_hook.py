"""Hook-level tests for aura.core.hooks.bash_safety.make_bash_safety_hook.

Pre-tool closure that short-circuits bash commands hitting the Tier A
hard-floor safety rules. Factory takes no context; hook is stateless.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from aura.core.hooks.bash_safety import make_bash_safety_hook
from aura.core.persistence import journal as journal_module
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool


class _BashArgs(BaseModel):
    command: str


class _PathOnly(BaseModel):
    path: str


def _bash_tool() -> Any:
    return build_tool(
        name="bash",
        description="bash",
        args_schema=_BashArgs,
        func=lambda command: "",
    )


def _read_tool() -> Any:
    return build_tool(
        name="read_file",
        description="read",
        args_schema=_PathOnly,
        func=lambda path: "",
        is_read_only=True,
    )


@pytest.mark.asyncio
async def test_non_bash_tool_passes_through() -> None:
    hook = make_bash_safety_hook()
    result = await hook(
        tool=_read_tool(),
        args={"path": "/tmp/x"},
        state=LoopState(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_safe_bash_passes_through() -> None:
    hook = make_bash_safety_hook()
    result = await hook(
        tool=_bash_tool(),
        args={"command": "ls -la"},
        state=LoopState(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_dangerous_bash_short_circuits() -> None:
    hook = make_bash_safety_hook()
    result = await hook(
        tool=_bash_tool(),
        args={"command": "zmodload zsh/system"},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "bash safety blocked" in result.error
    assert "zsh_dangerous_command" in result.error


@pytest.mark.asyncio
async def test_cd_git_compound_short_circuits() -> None:
    hook = make_bash_safety_hook()
    result = await hook(
        tool=_bash_tool(),
        args={"command": "cd /x && git status"},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "cd_git_compound" in result.error


@pytest.mark.asyncio
async def test_journal_event_on_block(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        hook = make_bash_safety_hook()
        await hook(
            tool=_bash_tool(),
            args={"command": "zmodload zsh/system"},
            state=LoopState(),
        )
        events = [json.loads(line) for line in log.read_text().splitlines()]
        blocked = [e for e in events if e["event"] == "bash_safety_blocked"]
        assert len(blocked) == 1
        assert blocked[0]["reason"] == "zsh_dangerous_command"
        assert "detail" in blocked[0]
        assert blocked[0]["command"] == "zmodload zsh/system"
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_empty_command_arg_passes_through() -> None:
    hook = make_bash_safety_hook()
    result = await hook(
        tool=_bash_tool(),
        args={"command": ""},
        state=LoopState(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_missing_command_arg_passes_through() -> None:
    hook = make_bash_safety_hook()
    result = await hook(
        tool=_bash_tool(),
        args={},
        state=LoopState(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_non_string_command_arg_passes_through() -> None:
    hook = make_bash_safety_hook()
    result = await hook(
        tool=_bash_tool(),
        args={"command": 42},
        state=LoopState(),
    )
    assert result is None
