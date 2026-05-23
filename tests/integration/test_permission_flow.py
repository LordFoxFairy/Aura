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
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from aura.application.hooks import HookChain
from aura.application.hooks.permission import make_permission_hook
from aura.application.permission.asker import AskerResponse
from aura.domain.permission.mode import Mode
from aura.domain.permission.session import SessionRuleSet
from aura.infrastructure import permission_store as perm_store
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
        mode=mode,  # type: ignore[arg-type]  # deliberately off-type arg to exercise path
    )
    return HookChain(pre_tool=[hook])


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
        history = agent.storage.load(agent.session_id)
    finally:
        await agent.aclose()

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


@pytest.mark.asyncio
async def test_bash_permission_deny_tool_refused_llm_sees_feedback(
    tmp_path: Path,
) -> None:
    # NOTE: probe must be a command the user wants to DENY but that is not
    # caught by the Tier A safety floor — otherwise the safety hook would
    # short-circuit before the permission asker is consulted, and this
    # test would assert on a safety-blocked message instead of a
    # user-denied one. ``rm -rf /`` was the original probe but is now
    # (correctly) caught by the ``destructive_removal`` floor.
    turns = [
        FakeTurn(
            message=AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_1",
                        "name": "bash",
                        "args": {"command": "echo sensitive-output"},
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
        history = agent.storage.load(agent.session_id)
    finally:
        await agent.aclose()

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


@pytest.mark.asyncio
async def test_plan_mode_exit_approval_flow_flips_mode_and_user_deny(
    tmp_path: Path,
) -> None:
    """Plan mode state machine — write blocked, exit_plan_mode Yes flips mode,
    exit_plan_mode No stays in plan, then the later write sees live default
    mode semantics.
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
    # Turn 4: after approval, the permission hook must re-read Agent.mode.
    # If it still enforces the construction-time "plan" value, this write
    # is dry-run blocked and the target file is never created.
    turn_4 = FakeTurn(
        message=AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc_w2",
                    "name": "write_file",
                    "args": {"path": str(target), "content": "after approval"},
                }
            ],
        )
    )
    turn_5 = FakeTurn(message=AIMessage(content="done"))

    perm_asker = ScriptedPermissionAsker()
    # Turn 4 runs after plan approval and should use default-mode semantics,
    # so the write asks once and this response lets it proceed.
    perm_asker.queue(AskerResponse(choice="accept"))
    # The exit_plan_mode tool falls through to rule match (which we seed
    # via a session rule); write_file is dry-run blocked in plan mode and
    # prompted in default mode.
    session = SessionRuleSet()
    from aura.domain.permission.rule import Rule

    session.add(Rule(tool="exit_plan_mode", content=None))

    # The exit_plan_mode tool has its own asker (the UserAsker) — No on
    # turn 2, Yes on turn 3.
    plan_asker = ScriptedAsker()
    plan_asker.queue_response("No")
    plan_asker.queue_response("Yes")

    hooks = HookChain()
    agent, _ = build_integration_agent(
        tmp_path,
        [turn_1, turn_2, turn_3, turn_4, turn_5],
        enabled_tools=["write_file", "enter_plan_mode", "exit_plan_mode"],
        hooks=hooks,
        mode="plan",
        question_asker=plan_asker,
    )
    live_permission_hook = make_permission_hook(
        asker=perm_asker,
        session=session,
        rules=perm_store.load_ruleset(tmp_path),
        project_root=tmp_path,
        mode=lambda: cast("Mode", agent.mode),
    )
    # Agent.__init__ inserts bash safety at 0 and must-read-first at the end.
    # Put permission between them, matching the normal caller-owned hook slot.
    hooks.pre_tool.insert(1, live_permission_hook)
    try:
        events = await drain(agent, "write a file")
    finally:
        await agent.aclose()

    completed = [e for e in events if isinstance(e, ToolCallCompleted)]
    assert [e.name for e in completed] == [
        "write_file",
        "exit_plan_mode",
        "exit_plan_mode",
        "write_file",
    ]
    # Turn 1: plan mode blocked the write (no asker consulted).
    assert completed[0].error is not None
    assert "plan mode" in completed[0].error.lower()
    assert len(perm_asker.calls) == 1
    # Turn 2: the exit_plan_mode user-approval gate rejected ("No").
    assert completed[1].error is not None
    assert "rejected" in completed[1].error.lower()
    # Mode did NOT flip after a denied approval.
    # (Verified by the fact that turn 3 successfully calls exit_plan_mode
    # again from plan mode.)
    # Turn 3: second attempt was approved, mode flipped to default.
    assert completed[2].error is None
    assert agent.mode == "default"
    # Turn 4: after approval, the follow-up write is governed by live
    # default-mode permission rules rather than stale plan-mode dry-run.
    assert completed[3].error is None
    assert target.read_text() == "after approval"
    assert perm_asker.calls[0]["tool"] == "write_file"
    # Approval asker was invoked for both attempts.
    assert len(plan_asker.calls) == 2


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
        await agent.aclose()

    completed = [e for e in events if isinstance(e, ToolCallCompleted)]
    assert len(completed) == 2
    # Both tools ran.
    assert all(e.error is None for e in completed)
    # Asker was consulted exactly once — for bash only.
    assert len(perm_asker.calls) == 1
    assert perm_asker.calls[0]["tool"] == "bash"
    # File was actually written by write_file.
    assert target.read_text() == "auto-allowed"


def _silence_unused_import_check() -> Any:  # pragma: no cover
    return (FakeChatModel, FakeTurn)
