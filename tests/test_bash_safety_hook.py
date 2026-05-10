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
from aura.schemas.permissions import Allow, Replace
from aura.schemas.state import LoopState
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
    outcome = await hook(
        tool=_read_tool(),
        args={"path": "/tmp/x"},
        state=LoopState(),
    )
    assert isinstance(outcome, Allow)


@pytest.mark.asyncio
async def test_safe_bash_passes_through() -> None:
    hook = make_bash_safety_hook()
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "ls -la"},
        state=LoopState(),
    )
    assert isinstance(outcome, Allow)


@pytest.mark.asyncio
async def test_dangerous_bash_short_circuits() -> None:
    hook = make_bash_safety_hook()
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "zmodload zsh/system"},
        state=LoopState(),
    )
    # Hook returns Replace.
    assert isinstance(outcome, Replace)
    assert outcome.result.ok is False
    assert outcome.result.error is not None
    assert "bash safety blocked" in outcome.result.error
    assert "zsh_dangerous_command" in outcome.result.error


@pytest.mark.asyncio
async def test_cd_git_compound_short_circuits() -> None:
    hook = make_bash_safety_hook()
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "cd /x && git status"},
        state=LoopState(),
    )
    # Hook returns Replace.
    assert isinstance(outcome, Replace)
    assert outcome.result.ok is False
    assert outcome.result.error is not None
    assert "cd_git_compound" in outcome.result.error


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
        blocked = [
            e for e in events
            if e["event"] == "permission_decision"
            and e.get("reason") == "safety_blocked"
        ]
        assert len(blocked) == 1
        assert blocked[0]["tool"] == "bash"
        assert "detail" in blocked[0]
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_empty_command_arg_passes_through() -> None:
    hook = make_bash_safety_hook()
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": ""},
        state=LoopState(),
    )
    assert isinstance(outcome, Allow)


@pytest.mark.asyncio
async def test_missing_command_arg_passes_through() -> None:
    hook = make_bash_safety_hook()
    outcome = await hook(
        tool=_bash_tool(),
        args={},
        state=LoopState(),
    )
    assert isinstance(outcome, Allow)


@pytest.mark.asyncio
async def test_non_string_command_arg_passes_through() -> None:
    hook = make_bash_safety_hook()
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": 42},
        state=LoopState(),
    )
    assert isinstance(outcome, Allow)


# ---------------------------------------------------------------------------
# Phase 1 Task 9 — Outcome variant assertions.
# Blocked paths return Replace; passthrough paths return Allow.
# (legacy, resolved in Task 10).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dangerous_bash_returns_replace_outcome() -> None:
    hook = make_bash_safety_hook()
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "zmodload zsh/system"},
        state=LoopState(),
    )
    assert isinstance(outcome, Replace), f"expected Replace, got {type(outcome).__name__}"
    assert outcome.result.ok is False
    assert "bash safety blocked" in (outcome.result.error or "")
    assert outcome.decision.allow is False
    assert outcome.decision.reason == "safety_blocked"


@pytest.mark.asyncio
async def test_cd_git_compound_returns_replace_outcome() -> None:
    hook = make_bash_safety_hook()
    outcome = await hook(
        tool=_bash_tool(),
        args={"command": "cd /x && git status"},
        state=LoopState(),
    )
    assert isinstance(outcome, Replace)
    assert outcome.result.ok is False
    assert outcome.decision.allow is False
    assert outcome.decision.reason == "safety_blocked"
