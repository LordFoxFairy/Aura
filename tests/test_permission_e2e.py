"""End-to-end integration test for Phase D Task 9.

Drives a full Agent.astream call with a real permission hook wired in,
exercising the claim made by spec §11 AC 2 and AC 7:

- First run: user answers "always" → rule persisted to .aura/settings.json.
- Second run (fresh Agent): same tool call → rule on disk auto-allows; asker
  never consulted.
- /clear drops runtime session-scope rules but leaves persisted rules alone.
- --bypass-permissions mode skips the asker entirely.

The test uses ``FakeChatModel`` from conftest for the LLM, a hand-rolled
``CountingAsker`` to record / script the prompt answers, and a minimal
in-memory tool that records whether it actually ran.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.hooks import HookChain
from aura.core.hooks.permission import AskerResponse, make_permission_hook
from aura.core.permissions import store
from aura.core.permissions.rule import Rule
from aura.core.permissions.session import SessionRuleSet
from aura.core.persistence.storage import SessionStorage
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel, FakeTurn


class _BashParams(BaseModel):
    command: str


def _make_tool(run_counter: list[int]) -> BaseTool:
    """In-memory bash-shaped tool. Records every call, doesn't shell out."""

    def _matcher(args: dict[str, object], content: str) -> bool:
        return args.get("command") == content

    _matcher.key = "command"  # type: ignore[attr-defined]

    def _run(command: str) -> dict[str, Any]:
        run_counter.append(1)
        return {"stdout": "ok", "stderr": "", "exit_code": 0}

    return build_tool(
        name="bash",
        description="fake bash",
        args_schema=_BashParams,
        func=_run,
        is_destructive=True,
        rule_matcher=_matcher,
        args_preview=lambda args: f"command: {args.get('command', '')}",
    )


def _minimal_cfg() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["bash"]},
    })


def _turns_with_one_bash_then_final() -> list[FakeTurn]:
    return [
        FakeTurn(message=AIMessage(
            content="",
            tool_calls=[{
                "id": "tc_1",
                "name": "bash",
                "args": {"command": "npm test"},
            }],
        )),
        FakeTurn(message=AIMessage(content="done")),
    ]


@dataclass
class _CountingAsker:
    """Asker that returns a scripted response and counts invocations."""

    response: AskerResponse
    call_count: int = 0
    received: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Rule,
    ) -> AskerResponse:
        self.call_count += 1
        self.received.append({"tool": tool.name, "args": dict(args)})
        return self.response


def _build_agent_with_perms(
    *,
    tmp_path: Path,
    project_root: Path,
    asker: Callable[..., Any],
    session_rules: SessionRuleSet,
    tool: BaseTool,
    turns: list[FakeTurn],
    mode: str = "default",
) -> Agent:
    """Wire a full permission hook around a FakeChatModel-driven Agent.

    Mirrors what aura/cli/__main__.py does, minus the banner/argparse.
    """
    ruleset = store.load_ruleset(project_root)
    hook = make_permission_hook(
        asker=asker,
        session=session_rules,
        rules=ruleset,
        project_root=project_root,
        mode=mode,  # type: ignore[arg-type]
    )
    hooks = HookChain(pre_tool=[hook])
    cfg = _minimal_cfg()
    model = FakeChatModel(turns=turns)
    storage = SessionStorage(tmp_path / "aura.db")
    return Agent(
        config=cfg,
        model=model,
        storage=storage,
        hooks=hooks,
        available_tools={"bash": tool},
        session_rules=session_rules,
    )


async def _drain(agent: Agent, prompt: str) -> None:
    async for _ in agent.astream(prompt):
        pass


@pytest.mark.asyncio
async def test_first_run_asker_always_persists_rule_to_disk(tmp_path: Path) -> None:
    run_counter: list[int] = []
    tool = _make_tool(run_counter)
    asker = _CountingAsker(response=AskerResponse(
        choice="always",
        scope="project",
        rule=Rule(tool="bash", content="npm test"),
    ))
    session = SessionRuleSet()

    agent = _build_agent_with_perms(
        tmp_path=tmp_path,
        project_root=tmp_path,
        asker=asker,
        session_rules=session,
        tool=tool,
        turns=_turns_with_one_bash_then_final(),
    )
    try:
        await _drain(agent, "run npm test please")
    finally:
        agent.close()

    assert asker.call_count == 1
    assert run_counter == [1]

    settings = tmp_path / ".aura" / "settings.json"
    assert settings.exists()
    cfg = store.load(tmp_path)
    assert "bash(npm test)" in cfg.allow


@pytest.mark.asyncio
async def test_second_agent_honours_persisted_rule_without_prompting(
    tmp_path: Path,
) -> None:
    # Seed disk with the rule a prior run would have left.
    store.save_rule(tmp_path, Rule(tool="bash", content="npm test"), scope="project")

    run_counter: list[int] = []
    tool = _make_tool(run_counter)
    # Asker would explode if called — signals the rule matched first.
    asker = _CountingAsker(response=AskerResponse(choice="deny"))
    session = SessionRuleSet()

    agent = _build_agent_with_perms(
        tmp_path=tmp_path,
        project_root=tmp_path,
        asker=asker,
        session_rules=session,
        tool=tool,
        turns=_turns_with_one_bash_then_final(),
    )
    try:
        await _drain(agent, "run npm test")
    finally:
        agent.close()

    assert asker.call_count == 0, "persisted rule must short-circuit the asker"
    assert run_counter == [1], "tool must have run"


@pytest.mark.asyncio
async def test_end_to_end_persist_then_auto_allow_across_two_agents(
    tmp_path: Path,
) -> None:
    """AC 2: option-2 on the prompt writes the rule; next aura run honours it."""
    run_counter: list[int] = []
    tool = _make_tool(run_counter)

    # --- First run: user answers "always", rule hits disk. ---
    asker1 = _CountingAsker(response=AskerResponse(
        choice="always",
        scope="project",
        rule=Rule(tool="bash", content="npm test"),
    ))
    agent1 = _build_agent_with_perms(
        tmp_path=tmp_path,
        project_root=tmp_path,
        asker=asker1,
        session_rules=SessionRuleSet(),
        tool=tool,
        turns=_turns_with_one_bash_then_final(),
    )
    try:
        await _drain(agent1, "first run")
    finally:
        agent1.close()
    assert asker1.call_count == 1
    assert (tmp_path / ".aura" / "settings.json").exists()

    # --- Second run: fresh Agent, same project_root. Asker must stay quiet. ---
    asker2 = _CountingAsker(response=AskerResponse(choice="deny"))
    # Fresh SessionRuleSet — project rule on disk is the only thing allowing.
    agent2 = _build_agent_with_perms(
        tmp_path=tmp_path,
        project_root=tmp_path,
        asker=asker2,
        session_rules=SessionRuleSet(),
        tool=tool,
        turns=_turns_with_one_bash_then_final(),
    )
    try:
        await _drain(agent2, "second run")
    finally:
        agent2.close()
    assert asker2.call_count == 0
    assert run_counter == [1, 1]


@pytest.mark.asyncio
async def test_clear_session_does_not_nuke_disk_rules(tmp_path: Path) -> None:
    """/clear drops runtime session rules; persisted rules survive."""
    run_counter: list[int] = []
    tool = _make_tool(run_counter)

    # First run persists a rule.
    asker = _CountingAsker(response=AskerResponse(
        choice="always",
        scope="project",
        rule=Rule(tool="bash", content="npm test"),
    ))
    agent1 = _build_agent_with_perms(
        tmp_path=tmp_path,
        project_root=tmp_path,
        asker=asker,
        session_rules=SessionRuleSet(),
        tool=tool,
        turns=_turns_with_one_bash_then_final(),
    )
    try:
        await _drain(agent1, "go")
        # /clear on the first Agent — disk must be untouched.
        agent1.clear_session()
    finally:
        agent1.close()

    cfg = store.load(tmp_path)
    assert "bash(npm test)" in cfg.allow, (
        "clear_session must not delete persisted project rules"
    )

    # Third run confirms disk rule still auto-allows.
    asker3 = _CountingAsker(response=AskerResponse(choice="deny"))
    agent3 = _build_agent_with_perms(
        tmp_path=tmp_path,
        project_root=tmp_path,
        asker=asker3,
        session_rules=SessionRuleSet(),
        tool=tool,
        turns=_turns_with_one_bash_then_final(),
    )
    try:
        await _drain(agent3, "post-clear")
    finally:
        agent3.close()

    assert asker3.call_count == 0


@pytest.mark.asyncio
async def test_bypass_mode_skips_asker_entirely(tmp_path: Path) -> None:
    """AC 7: --bypass-permissions → every tool call allowed, asker untouched."""
    run_counter: list[int] = []
    tool = _make_tool(run_counter)
    asker = _CountingAsker(response=AskerResponse(choice="deny"))
    session = SessionRuleSet()

    agent = _build_agent_with_perms(
        tmp_path=tmp_path,
        project_root=tmp_path,
        asker=asker,
        session_rules=session,
        tool=tool,
        turns=_turns_with_one_bash_then_final(),
        mode="bypass",
    )
    try:
        await _drain(agent, "bypass run")
    finally:
        agent.close()

    assert asker.call_count == 0
    assert run_counter == [1]
    # Nothing should have been persisted — bypass is a per-process override,
    # not a rule-creating event.
    assert not (tmp_path / ".aura" / "settings.json").exists()
