"""F-0910-016 — ``/context`` introspection command.

Prints per-section token estimates: system / memory / skills / files /
history. Output is a "view" CommandResult.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from aura.application.commands.builtin import ContextCommand
from aura.application.commands.registry import CommandRegistry
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.skills.types import Skill
from tests.conftest import FakeChatModel, FakeTurn


def _config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def _agent(tmp_path: Path) -> Agent:
    return Agent(
        config=_config(),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="x"))] * 5),
        storage=SessionStorage(tmp_path / "aura.db"),
    )


@pytest.mark.asyncio
async def test_context_command_returns_view(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    cmd = ContextCommand()
    result = await cmd.handle("", agent)
    assert result.handled is True
    assert result.kind == "view"
    assert "Context token estimates" in result.text


@pytest.mark.asyncio
async def test_context_command_lists_all_sections(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    cmd = ContextCommand()
    result = await cmd.handle("", agent)
    for section in ("system", "memory", "skills", "files", "other", "history"):
        assert section in result.text
    assert "raw-store" in result.text


@pytest.mark.asyncio
async def test_context_command_skills_count_grows_with_invocation(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    skill = Skill(
        name="ping",
        description="ping",
        body="P" * 400,
        source_path=tmp_path / "ping.md",
        layer="project",
    )
    agent.record_skill_invocation(skill)

    cmd = ContextCommand()
    result = await cmd.handle("", agent)
    # Find the skills line and parse its number — must be > 0 once a skill
    # is invoked + rendered through Context.build.
    skills_line = next(
        line for line in result.text.splitlines() if "skills" in line
    )
    num = int(skills_line.split(":")[1].strip())
    assert num > 0


@pytest.mark.asyncio
async def test_context_command_history_count_reflects_storage(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    history = [
        HumanMessage(content="A" * 400),
        AIMessage(content="B" * 400),
    ]
    agent._storage.save(agent.session_id, history)

    cmd = ContextCommand()
    result = await cmd.handle("", agent)
    history_line = next(
        line for line in result.text.splitlines() if "history" in line
    )
    num = int(history_line.split(":")[1].strip())
    # 800 chars / 4 = 200 tokens minimum for the content; envelope adds none
    # since langchain doesn't auto-wrap. Allow >= 200.
    assert num >= 200


@pytest.mark.asyncio
async def test_context_command_counts_hidden_tool_call_args(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    history = [
        HumanMessage(content="u"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "read_file",
                "args": {"path": "重要文件.py", "note": "修复" * 200},
                "id": "tc_1",
            }],
        ),
    ]
    agent._storage.save(agent.session_id, history)

    result = await ContextCommand().handle("", agent)

    history_line = next(
        line for line in result.text.splitlines() if "history" in line
    )
    assert int(history_line.split(":")[1].strip()) > 200


@pytest.mark.asyncio
async def test_context_command_shows_manual_compact_prompt_estimate(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    history: list[BaseMessage] = []
    for i in range(8):
        history.append(HumanMessage(content=f"u-{i} " + ("x" * 200)))
        history.append(AIMessage(content=f"a-{i} " + ("y" * 200)))
    agent._storage.save(agent.session_id, history)

    result = await ContextCommand().handle("", agent)

    assert "compact" in result.text
    assert "manual /compact summary prompt" in result.text


@pytest.mark.asyncio
async def test_context_command_total_uses_live_microcompacted_history(
    tmp_path: Path,
) -> None:
    agent = Agent(
        config=_config(),
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "aura.db"),
        microcompact_trigger_pairs=2,
        microcompact_keep_recent=1,
    )
    history: list[BaseMessage] = []
    for i in range(4):
        call_id = f"tc-{i}"
        history.append(HumanMessage(content=f"u-{i}"))
        history.append(AIMessage(
            content="",
            tool_calls=[{
                "name": "read_file",
                "args": {"path": f"f-{i}.py"},
                "id": call_id,
            }],
        ))
        history.append(ToolMessage(
            content="RAW-" + ("x" * 20_000),
            tool_call_id=call_id,
            name="read_file",
        ))
    agent._storage.save(agent.session_id, history)

    result = await ContextCommand().handle("", agent)

    lines = result.text.splitlines()
    history_tokens = int(next(line for line in lines if "history" in line).split(":")[1].strip())
    raw_tokens = int(next(line for line in lines if "raw-store" in line).split(":")[1].split()[0])
    assert raw_tokens > history_tokens * 3
    await agent.aclose()


def test_context_command_can_register(tmp_path: Path) -> None:
    """The command is registry-registerable under its '/context' name."""
    r = CommandRegistry()
    r.register(ContextCommand())
    assert any(c.name == "/context" for c in r.list())
