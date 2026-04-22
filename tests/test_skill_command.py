"""Tests for aura.core.skills.command.SkillCommand."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.commands import CommandRegistry
from aura.core.persistence.storage import SessionStorage
from aura.core.skills.command import SkillCommand
from aura.core.skills.types import Skill
from tests.conftest import FakeChatModel


def _skill(name: str = "foo") -> Skill:
    return Skill(
        name=name,
        description=f"Description of {name}.",
        body=f"# Body of {name}\ndo the thing",
        source_path=Path(f"/tmp/{name}.md"),
        layer="user",
    )


def _agent(tmp_path: Path) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "db"),
    )


def test_skill_command_name_auto_prefixed_slash(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    cmd = SkillCommand(skill=_skill("bar"), agent=agent)
    assert cmd.name == "/bar"
    assert cmd.description == "Description of bar."
    assert cmd.source == "skill"
    agent.close()


@pytest.mark.asyncio
async def test_skill_command_handle_calls_record_invocation(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    skill = _skill("sk")
    cmd = SkillCommand(skill=skill, agent=agent)

    result = await cmd.handle("", agent)

    assert result.handled is True
    assert result.kind == "print"
    assert "sk" in result.text

    # Skill must now be present in the Context's invoked list (visible via build()).
    messages = agent._context.build([])
    contents = " ".join(str(m.content) for m in messages)
    assert '<skill-invoked name="sk">' in contents
    agent.close()


@pytest.mark.asyncio
async def test_skill_command_registers_via_command_registry_and_dispatches(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    skill = _skill("helper")
    cmd = SkillCommand(skill=skill, agent=agent)
    registry = CommandRegistry()
    registry.register(cmd)

    result = await registry.dispatch("/helper", agent)

    assert result.handled is True
    assert result.kind == "print"

    messages = agent._context.build([])
    contents = " ".join(str(m.content) for m in messages)
    assert '<skill-invoked name="helper">' in contents
    agent.close()
