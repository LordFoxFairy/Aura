"""Integration: permission asker roundtrip through a real Agent.

Unit tests in ``tests/test_permission_mode.py`` / ``tests/test_cli_permission*.py``
exercise the permission hook or the CLI asker in isolation. This tier
wires both ends to a real Agent (so history gets the ToolMessage, mode
transitions propagate, and the LLM sees the decision) and asserts on
the *observable* downstream effects: did the tool run, did the LLM
receive the right ToolMessage, did ``accept_edits`` auto-allow without
asking.

Each test scripts a full turn sequence through FakeChatModel and
inspects both the ``ToolCallCompleted`` events and the persisted
history.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from aura.core.hooks import HookChain
from aura.core.hooks.permission import AskerResponse, make_permission_hook
from aura.core.permissions import store as perm_store
from aura.core.permissions.session import SessionRuleSet
from aura.schemas.events import ToolCallCompleted
from tests.conftest import FakeChatModel, FakeTurn
from tests.integration.conftest import (
    ScriptedAsker,
    ScriptedPermissionAsker,
    build_integration_agent,
    drain,
)


def _wire_permission_hook(
    *,
    project_root: Path,
    asker: ScriptedPermissionAsker,
    session_rules: SessionRuleSet,
    mode: str = "default",
) -> HookChain:
    """Build a HookChain with a real permission hook installed."""
    ruleset = perm_store.load_ruleset(project_root)
    hook = make_permission_hook(
        asker=asker,
        session=session_rules,
        rules=ruleset,
        project_root=project_root,
        mode=mode,  # type: ignore[arg-type]
    )
    return HookChain(pre_tool=[hook])


# ---------------------------------------------------------------------------
# Test 1 — bash → asker says Yes → tool runs, LLM sees output.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_permission_allow_tool_runs_llm_sees_stdout(
    tmp_path: Path,
) -> None:
    turns = [
        FakeTurn(
            message=AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_1",
                        "name": "bash",
                        "args": {"command": "echo hello-world"},
                    }
                ],
            )
        ),
        FakeTurn(message=AIMessage(content="ok")),
    ]
    perm_asker = ScriptedPermissionAsker()
    perm_asker.queue(AskerResponse(choice="accept"))
    hooks = _wire_permission_hook(
        project_root=tmp_path,
        asker=perm_asker,
        session_rules=SessionRuleSet(),
    )
    agent, _ = build_integration_agent(
        tmp_path,
        turns,
        enabled_tools=["bash"],
        hooks=hooks,
    )
    try:
        events = await drain(agent, "say hello")
        # Read history BEFORE close — SessionStorage.close shuts the DB
        # connection down, so deferred reads blow up with "closed database".
        history = agent._storage.load(agent.session_id)
    finally:
        agent.close()

    # Asker was consulted exactly once.
    assert len(perm_asker.calls) == 1
    assert perm_asker.calls[0]["tool"] == "bash"

    # Tool ran — ToolCallCompleted carries stdout containing "hello-world".
    completed = [e for e in events if isinstance(e, ToolCallCompleted)]
    assert len(completed) == 1
    output = completed[0].output
    assert output is not None
    assert "hello-world" in str(output)

    # LLM's turn 2 history has a ToolMessage with success status carrying
    # the same content.
    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "success"
    assert "hello-world" in str(tool_msgs[0].content)


# ---------------------------------------------------------------------------
# Test 2 — bash → asker says No with feedback → tool refused, LLM sees deny.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_permission_deny_tool_refused_llm_sees_feedback(
    tmp_path: Path,
) -> None:
    turns = [
        FakeTurn(
            message=AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_1",
                        "name": "bash",
                        "args": {"command": "rm -rf /"},
                    }
                ],
            )
        ),
        FakeTurn(message=AIMessage(content="understood, I won't")),
    ]
    perm_asker = ScriptedPermissionAsker()
    perm_asker.queue(
        AskerResponse(choice="deny", feedback="dangerous command")
    )
    hooks = _wire_permission_hook(
        project_root=tmp_path,
        asker=perm_asker,
        session_rules=SessionRuleSet(),
    )
    agent, _ = build_integration_agent(
        tmp_path,
        turns,
        enabled_tools=["bash"],
        hooks=hooks,
    )
    try:
        events = await drain(agent, "rm things")
        history = agent._storage.load(agent.session_id)
    finally:
        agent.close()

    completed = [e for e in events if isinstance(e, ToolCallCompleted)]
    assert len(completed) == 1
    assert completed[0].error is not None
    # The denial message must surface the user's feedback so the LLM can
    # read WHY it was refused.
    err = completed[0].error
    assert "denied" in err.lower()
    assert "dangerous" in err

    # ToolMessage in history is status="error" with same content.
    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "error"
    assert "dangerous" in str(tool_msgs[0].content)


# ---------------------------------------------------------------------------
# Test 3 — plan mode state machine: write blocked → exit_plan_mode approved
#          → mode flips → write now permitted.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_mode_exit_approval_flow_flips_mode_and_user_deny(
    tmp_path: Path,
) -> None:
    """Plan mode state machine — write blocked, exit_plan_mode Yes flips mode,
    exit_plan_mode No stays in plan.

    The test covers the FULL 3-turn state machine BUT stops BEFORE asserting
    "a later write_file tool call now sees default-mode semantics". The
    permission hook currently closes over ``mode`` at construction time
    (see ``aura.core.hooks.permission.make_permission_hook``) — the CLI
    ``/shift+tab`` keybinding and the ``exit_plan_mode`` approval path both
    mutate ``Agent.mode`` via ``set_mode``, but the hook never re-reads it.
    That divergence is reported in the integration-test findings; this
    test exercises everything up to and including the tool's approval
    gate so the gate itself can't silently regress.
    """
    target = tmp_path / "draft.txt"
    # Turn 1: LLM tries write_file (blocked by plan mode).
    turn_1 = FakeTurn(
        message=AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc_w1",
                    "name": "write_file",
                    "args": {"path": str(target), "content": "initial"},
                }
            ],
        )
    )
    # Turn 2: LLM calls exit_plan_mode — we script a DENY on the plan
    # approval first to prove the gate rejects when the user says No.
    turn_2 = FakeTurn(
        message=AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc_exit_deny",
                    "name": "exit_plan_mode",
                    "args": {"plan": "1. write draft", "to_mode": "default"},
                }
            ],
        )
    )
    # Turn 3: LLM tries exit_plan_mode AGAIN — this time the user approves,
    # mode flips to default.
    turn_3 = FakeTurn(
        message=AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc_exit_yes",
                    "name": "exit_plan_mode",
                    "args": {"plan": "1. revised", "to_mode": "default"},
                }
            ],
        )
    )
    turn_4 = FakeTurn(message=AIMessage(content="done"))

    perm_asker = ScriptedPermissionAsker()
    # Permission hook is never consulted: turn 1 is plan-mode blocked, and
    # exit_plan_mode is on the hook's _PLAN_MODE_EXEMPT_TOOLS list so it
    # falls through to rule match (which we seed via a session rule).
    session = SessionRuleSet()
    from aura.core.permissions.rule import Rule

    session.add(Rule(tool="exit_plan_mode", content=None))

    # The exit_plan_mode tool has its own asker (the QuestionAsker) — No on
    # turn 2, Yes on turn 3.
    plan_asker = ScriptedAsker()
    plan_asker.queue_response("No")
    plan_asker.queue_response("Yes")

    hooks = _wire_permission_hook(
        project_root=tmp_path,
        asker=perm_asker,
        session_rules=session,
        mode="plan",
    )
    agent, _ = build_integration_agent(
        tmp_path,
        [turn_1, turn_2, turn_3, turn_4],
        enabled_tools=["write_file", "enter_plan_mode", "exit_plan_mode"],
        hooks=hooks,
        mode="plan",
        question_asker=plan_asker,
    )
    try:
        events = await drain(agent, "write a file")
    finally:
        agent.close()

    completed = [e for e in events if isinstance(e, ToolCallCompleted)]
    assert [e.name for e in completed] == [
        "write_file",
        "exit_plan_mode",
        "exit_plan_mode",
    ]
    # Turn 1: plan mode blocked the write (no asker consulted).
    assert completed[0].error is not None
    assert "plan mode" in completed[0].error.lower()
    assert len(perm_asker.calls) == 0
    # Turn 2: the exit_plan_mode user-approval gate rejected ("No").
    assert completed[1].error is not None
    assert "rejected" in completed[1].error.lower()
    # Mode did NOT flip after a denied approval.
    # (Verified by the fact that turn 3 successfully calls exit_plan_mode
    # again from plan mode.)
    # Turn 3: second attempt was approved, mode flipped to default.
    assert completed[2].error is None
    assert agent.mode == "default"
    # Approval asker was invoked for both attempts.
    assert len(plan_asker.calls) == 2


# ---------------------------------------------------------------------------
# Test 4 — accept_edits mode: write_file auto-allowed, bash still asks.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accept_edits_auto_allows_write_bash_still_prompts(
    tmp_path: Path,
) -> None:
    target = tmp_path / "edited.txt"
    turns = [
        FakeTurn(
            message=AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_w",
                        "name": "write_file",
                        "args": {
                            "path": str(target),
                            "content": "auto-allowed",
                        },
                    }
                ],
            )
        ),
        FakeTurn(
            message=AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_b",
                        "name": "bash",
                        "args": {"command": "echo from-bash"},
                    }
                ],
            )
        ),
        FakeTurn(message=AIMessage(content="done")),
    ]
    perm_asker = ScriptedPermissionAsker()
    # Only the bash call should ever reach the asker — write_file is
    # auto-allowed in accept_edits mode.
    perm_asker.queue(AskerResponse(choice="accept"))

    hooks = _wire_permission_hook(
        project_root=tmp_path,
        asker=perm_asker,
        session_rules=SessionRuleSet(),
        mode="accept_edits",
    )
    agent, _ = build_integration_agent(
        tmp_path,
        turns,
        enabled_tools=["write_file", "bash"],
        hooks=hooks,
        mode="accept_edits",
    )
    try:
        events = await drain(agent, "edit then shell")
    finally:
        agent.close()

    completed = [e for e in events if isinstance(e, ToolCallCompleted)]
    assert len(completed) == 2
    # Both tools ran.
    assert all(e.error is None for e in completed)
    # Asker was consulted exactly once — for bash only.
    assert len(perm_asker.calls) == 1
    assert perm_asker.calls[0]["tool"] == "bash"
    # File was actually written by write_file.
    assert target.read_text() == "auto-allowed"


# ---------------------------------------------------------------------------
# Extra sanity: placeholder to preserve the "_" variable discipline across
# the module. ``tmp_path`` usage elsewhere in this file; no fixture-scope
# leakage expected.
# ---------------------------------------------------------------------------


def _silence_unused_import_check() -> Any:  # pragma: no cover
    return (FakeChatModel, FakeTurn)
