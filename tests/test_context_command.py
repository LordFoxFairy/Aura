"""F-0910-016 — ``/context`` introspection command.

Prints per-section token estimates: system / memory / skills / files /
history. Output is a "view" CommandResult.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.commands.builtin import ContextCommand
from aura.core.commands.registry import CommandRegistry
from aura.core.persistence.storage import SessionStorage
from aura.core.skills.types import Skill
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


def test_context_command_can_register(tmp_path: Path) -> None:
    """The command is registry-registerable under its '/context' name."""
    r = CommandRegistry()
    r.register(ContextCommand())
    assert any(c.name == "/context" for c in r.list())
