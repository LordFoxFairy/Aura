"""Hook-level tests for aura.application.hooks.bash_safety.make_bash_safety_hook.

Pre-tool closure that short-circuits bash commands hitting the Tier A
hard-floor safety rules. Factory takes no context; hook is stateless.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from aura.application.hooks.bash_safety import make_bash_safety_hook
from aura.application.loop_state import LoopState
from aura.domain.permission.outcome import Allow, Outcome, Replace
from aura.infrastructure.persistence import journal as journal_module
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


async def _run(command: str) -> Outcome:
    hook = make_bash_safety_hook()
    return await hook(tool=_bash_tool(), args={"command": command}, state=LoopState())


def _assert_blocked(outcome: Outcome, reason: str) -> None:
    assert isinstance(outcome, Replace)
    assert outcome.result.ok is False
    assert outcome.decision.reason == "safety_blocked"
    assert reason in (outcome.result.error or "")


# --- destructive_removal: long-form flags + non-recursive escape hatch ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "rm --recursive /etc",
        "rm --force /etc",
        "rm --verbose --force /etc",
    ],
)
async def test_rm_long_form_recursive_force_on_system_path_blocks(command: str) -> None:
    """GNU long flags (--recursive/--force) must arm the floor exactly like -rf."""
    _assert_blocked(await _run(command), "destructive_removal")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "rm /etc/passwd",
        "rm --verbose /etc/hosts",
        "rm /usr/bin/python",
    ],
)
async def test_rm_without_recursive_or_force_is_allowed(command: str) -> None:
    """A single non-recursive rm of one system file is not the Tier A floor target."""
    assert isinstance(await _run(command), Allow)


# --- root_chown: leading flags skipped before the owner spec ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "chown -R root /tmp/x",
        "chown --recursive root /tmp/x",
        "chown -h 0:0 /tmp/x",
    ],
)
async def test_chown_root_after_leading_flags_blocks(command: str) -> None:
    """Flags before the owner spec must be skipped so chown root is still caught."""
    _assert_blocked(await _run(command), "root_chown")


@pytest.mark.asyncio
async def test_chown_non_root_owner_is_allowed() -> None:
    """Reparenting to a normal user is routine and must not trip the root-chown floor."""
    assert isinstance(await _run("chown -R alice /tmp/x"), Allow)


# --- sed_inplace_system_path: long-form in-place + unrelated long flags ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "sed --in-place s/a/b/ /etc/hosts",
        "sed --in-place=.bak s/a/b/ /etc/passwd",
        "sed --regexp-extended -i s/a/b/ /etc/hosts",
    ],
)
async def test_sed_long_form_inplace_on_system_path_blocks(command: str) -> None:
    """--in-place and --in-place=SUFFIX rewrite system config just like -i."""
    _assert_blocked(await _run(command), "sed_inplace_system_path")


@pytest.mark.asyncio
async def test_sed_non_inplace_long_flag_is_allowed() -> None:
    """A read-only sed flag on a system path must not be mistaken for in-place editing."""
    assert isinstance(await _run("sed --quiet /etc/hosts"), Allow)


# --- pipe_to_shell: empty pipe segment must continue, not short-circuit ---


@pytest.mark.asyncio
async def test_empty_pipe_segment_then_shell_still_blocks() -> None:
    """An empty pipe segment must be skipped, not stop the scan from reaching 'sh'."""
    _assert_blocked(await _run("echo a | | sh"), "pipe_to_shell")


@pytest.mark.asyncio
async def test_trailing_empty_pipe_segment_is_allowed() -> None:
    """A dangling trailing pipe yields no shell target, so the command is benign."""
    assert isinstance(await _run("cat f | head | "), Allow)


# --- idempotency: stateless hook returns identical verdict on repeated calls ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expect_blocked"),
    [
        ("rm --recursive /etc", True),
        ("chown -R root /tmp/x", True),
        ("sed --in-place=.bak s/a/b/ /etc/passwd", True),
        ("rm /etc/passwd", False),
        ("sed --quiet /etc/hosts", False),
    ],
)
async def test_repeated_invocation_is_idempotent(command: str, expect_blocked: bool) -> None:
    """Two identical calls must yield the same verdict — the floor holds no per-call state."""
    first = await _run(command)
    second = await _run(command)
    assert isinstance(first, Replace) is expect_blocked
    assert isinstance(second, Replace) is expect_blocked
    assert type(first) is type(second)
