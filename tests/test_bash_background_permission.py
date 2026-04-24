"""bash_background — parity with bash through the unified safety hook.

Before the fix, ``bash_background`` called ``check_bash_safety`` inline
and raised a :class:`ToolError`, bypassing the hook chain entirely. That
meant:

1. No ``permission_decision`` journal event was emitted — only a
   ``bash_background_safety_blocked`` one — so audit scrapers that filter
   on ``permission_decision`` missed bash_background denials.
2. The denials sink (``state.custom[DENIALS_SINK_KEY]``) stayed empty —
   :meth:`Agent.last_turn_denials` returned ``()`` even though the call
   was denied.
3. ``mode == "bypass"`` was ignored — a user-opted-in bypass still had
   safety applied, while the blocking ``bash`` tool correctly skipped
   safety under bypass.

These tests assert the post-fix parity: bash_background flows through
the same :func:`make_bash_safety_hook` the blocking ``bash`` tool uses,
so the journal event, denials sink, and bypass semantics are identical.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.hooks.bash_safety import make_bash_safety_hook
from aura.core.permissions.denials import DENIALS_SINK_KEY, PermissionDenial
from aura.core.persistence import journal as journal_module
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.store import TasksStore
from aura.schemas.state import LoopState
from aura.tools.bash_background import BashBackground
from tests.conftest import FakeChatModel, FakeTurn


def _fake_bash_bg_tool() -> BashBackground:
    """A real BashBackground with fresh state — used for hook-level assertions."""
    store = TasksStore()
    return BashBackground(
        store=store,
        running_shells={},
        running_tasks={},
    )


# -----------------------------------------------------------------------
# Hook-level — bash_safety hook matches bash_background
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_safety_hook_matches_bash_background() -> None:
    """``make_bash_safety_hook`` must short-circuit a dangerous command
    routed through a ``bash_background`` BaseTool — not just ``bash``."""
    hook = make_bash_safety_hook()
    outcome = await hook(
        tool=_fake_bash_bg_tool(),
        args={"command": "zmodload zsh/system"},
        state=LoopState(),
    )
    assert outcome.short_circuit is not None
    assert outcome.short_circuit.ok is False
    assert outcome.short_circuit.error is not None
    assert "bash safety blocked" in outcome.short_circuit.error
    assert "zsh_dangerous_command" in outcome.short_circuit.error


@pytest.mark.asyncio
async def test_bash_safety_hook_emits_permission_decision_for_bash_background(
    tmp_path: Path,
) -> None:
    """Safety denials on bash_background must emit a ``permission_decision``
    journal event with ``reason='safety_blocked'`` — same shape the
    permission hook uses for path-based safety blocks, so downstream
    audit consumers see a uniform record."""
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)
    try:
        hook = make_bash_safety_hook()
        await hook(
            tool=_fake_bash_bg_tool(),
            args={"command": "zmodload zsh/system"},
            state=LoopState(),
        )
        events = [json.loads(line) for line in log.read_text().splitlines()]
        decisions = [e for e in events if e["event"] == "permission_decision"]
        assert len(decisions) == 1, events
        assert decisions[0]["tool"] == "bash_background"
        assert decisions[0]["reason"] == "safety_blocked"
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_bash_safety_hook_populates_denials_sink_for_bash_background() -> None:
    """The denials sink (``state.custom[DENIALS_SINK_KEY]``) must carry a
    :class:`PermissionDenial` entry for bash_background safety blocks, so
    :meth:`Agent.last_turn_denials` surfaces them to SDK consumers."""
    sink: list[PermissionDenial] = []
    state = LoopState(custom={DENIALS_SINK_KEY: sink})
    hook = make_bash_safety_hook()
    await hook(
        tool=_fake_bash_bg_tool(),
        args={"command": "rm -rf /etc"},
        state=state,
        tool_call_id="tc_bg_safety_1",
    )
    assert len(sink) == 1
    entry = sink[0]
    assert entry.tool_name == "bash_background"
    assert entry.reason == "safety_blocked"
    assert entry.tool_use_id == "tc_bg_safety_1"
    assert entry.tool_input == {"command": "rm -rf /etc"}


@pytest.mark.asyncio
async def test_bash_safety_hook_sets_decision_on_outcome_for_bash_background() -> None:
    """The hook must return ``PreToolOutcome.decision`` populated so the
    Loop can stamp it on :attr:`ToolStep.permission_decision` and the
    auditor emits a ``PermissionAudit`` event between Started and
    Completed — same channel path the permission hook uses."""
    hook = make_bash_safety_hook()
    outcome = await hook(
        tool=_fake_bash_bg_tool(),
        args={"command": "zmodload zsh/system"},
        state=LoopState(),
    )
    assert outcome.decision is not None
    assert outcome.decision.allow is False
    assert outcome.decision.reason == "safety_blocked"


# -----------------------------------------------------------------------
# Hook-level — bypass mode short-circuits safety for bash_background
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_safety_hook_honors_bypass_mode_for_bash_background() -> None:
    """``mode == 'bypass'`` must skip the safety check for bash_background
    (same behavior the permission layer has for bypass). The user opted
    into bypass; this tool layer must not second-guess that opt-in."""
    hook = make_bash_safety_hook(mode_provider=lambda: "bypass")
    outcome = await hook(
        tool=_fake_bash_bg_tool(),
        args={"command": "zmodload zsh/system"},
        state=LoopState(),
    )
    assert outcome.short_circuit is None


@pytest.mark.asyncio
async def test_bash_safety_hook_honors_bypass_mode_for_bash() -> None:
    """Same bypass semantics for the blocking bash tool — no regression."""
    from aura.tools.base import build_tool

    class _BashArgs(BaseModel):
        command: str

    bash = build_tool(
        name="bash",
        description="bash",
        args_schema=_BashArgs,
        func=lambda command: "",
    )
    hook = make_bash_safety_hook(mode_provider=lambda: "bypass")
    outcome = await hook(
        tool=bash,
        args={"command": "zmodload zsh/system"},
        state=LoopState(),
    )
    assert outcome.short_circuit is None


# -----------------------------------------------------------------------
# Tool-level — inline check_bash_safety is gone; the tool no longer raises
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_background_no_longer_raises_tool_error_inline() -> None:
    """After the fix the tool must NOT raise :class:`ToolError` for a
    dangerous command — safety lives entirely in the hook chain. A
    direct ``ainvoke`` (bypassing the hook chain) therefore succeeds at
    the tool level, letting the subprocess run; that's fine because in
    real turns the hook runs FIRST and short-circuits.

    This test documents the contract flip: bash_background is now a
    dumb executor, and safety is a hook-chain concern. The legitimate
    end-to-end guarantee is exercised by
    ``test_agent_bash_background_safety_end_to_end`` below.
    """
    store = TasksStore()
    running_shells: dict[str, asyncio.subprocess.Process] = {}
    running_tasks: dict[str, asyncio.Task[None]] = {}
    tool = BashBackground(
        store=store, running_shells=running_shells, running_tasks=running_tasks,
    )
    # The command is safe enough to spawn + complete (``exit 0``); the
    # point is that the tool itself does no safety check. Before the
    # fix, inline check_bash_safety would run; after, it's gone.
    out = await tool.ainvoke({"command": "exit 0"})
    assert out["status"] == "running"
    # Drain the watcher so the test doesn't leak a pending subprocess task.
    await asyncio.gather(*running_tasks.values())


# -----------------------------------------------------------------------
# End-to-end — Agent.astream routes bash_background through the hook chain
# -----------------------------------------------------------------------


def _minimal_config(enabled: list[str]) -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": enabled},
    })


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "aura.db")


@pytest.mark.asyncio
async def test_agent_bash_background_safety_end_to_end_populates_denials(
    tmp_path: Path,
) -> None:
    """When an Agent turn calls bash_background with a dangerous command,
    the safety hook must (a) block the call, (b) append a PermissionDenial
    to the per-turn sink, and (c) emit ``permission_decision`` — all
    without raising out of the tool."""
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)

    try:
        cfg = _minimal_config(enabled=["bash_background"])
        model = FakeChatModel(turns=[
            FakeTurn(message=AIMessage(
                content="",
                tool_calls=[{
                    "id": "tc_bg_1",
                    "name": "bash_background",
                    "args": {"command": "zmodload zsh/system"},
                }],
            )),
            FakeTurn(message=AIMessage(content="done")),
        ])
        agent = Agent(
            config=cfg,
            model=model,
            storage=_storage(tmp_path),
        )

        events: list[Any] = []
        async for e in agent.astream("try bg exploit"):
            events.append(e)

        # ToolMessage carries the short-circuit error.
        history = agent._storage.load("default")
        tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
        assert tool_msgs, "expected a ToolMessage for the blocked call"
        blob = " ".join(str(m.content) for m in tool_msgs)
        assert "bash safety blocked" in blob
        assert "zsh_dangerous_command" in blob

        # last_turn_denials surfaces the structured denial.
        denials = agent.last_turn_denials()
        assert len(denials) == 1
        assert denials[0].tool_name == "bash_background"
        assert denials[0].reason == "safety_blocked"
        assert denials[0].tool_use_id == "tc_bg_1"

        # Journal carries a permission_decision with the matching fields.
        events_on_disk = [
            json.loads(line) for line in log.read_text().splitlines()
        ]
        decisions = [
            e for e in events_on_disk
            if e["event"] == "permission_decision"
            and e.get("tool") == "bash_background"
        ]
        assert len(decisions) == 1
        assert decisions[0]["reason"] == "safety_blocked"

        await agent.aclose()
    finally:
        journal_module.reset()


@pytest.mark.asyncio
async def test_agent_bash_background_bypass_mode_skips_safety(
    tmp_path: Path,
) -> None:
    """Under ``mode='bypass'`` the same dangerous command must reach the
    tool (i.e. the safety short-circuit is skipped). We don't want the
    command to actually run (``zmodload`` would no-op on non-zsh shells
    anyway), so we only assert that no safety short-circuit fired: no
    ``permission_decision{reason='safety_blocked'}`` and no denial in the
    sink."""
    log = tmp_path / "events.jsonl"
    journal_module.reset()
    journal_module.configure(log)

    try:
        cfg = _minimal_config(enabled=["bash_background"])
        # We use a harmless command that WOULD normally trip safety under
        # default mode but is fine to actually execute: ``echo $(whoami)``
        # trips command_substitution. Under bypass, it should run.
        model = FakeChatModel(turns=[
            FakeTurn(message=AIMessage(
                content="",
                tool_calls=[{
                    "id": "tc_bg_bypass",
                    "name": "bash_background",
                    "args": {"command": "echo $(whoami)"},
                }],
            )),
            FakeTurn(message=AIMessage(content="done")),
        ])
        agent = Agent(
            config=cfg,
            model=model,
            storage=_storage(tmp_path),
            mode="bypass",
        )

        events: list[Any] = []
        async for e in agent.astream("run in bypass"):
            events.append(e)

        # No safety-block denial in the sink.
        denials = agent.last_turn_denials()
        safety_denials = [d for d in denials if d.reason == "safety_blocked"]
        assert safety_denials == [], safety_denials

        # Let the background watcher drain so we don't leak tasks; cleanup
        # happens inside agent.aclose(), which also cancels running shells.
        await agent.aclose()
        # Give pending subprocess cleanup a moment.
        await asyncio.sleep(0.1)
    finally:
        journal_module.reset()
