"""F-0910-008 — skill re-injection after compact rebuild.

After the summary turn, every skill in the pre-compact ``_invoked_skills``
must land back in the rebuilt history as a single ``<skill-active>``
HumanMessage capped at 5000 tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
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
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="SUMMARY"))]),
        storage=SessionStorage(tmp_path / "aura.db"),
    )


def _seed_history(agent: Agent, *, pairs: int = 10) -> None:
    h: list = []
    for i in range(pairs):
        h.append(HumanMessage(content=f"u-{i}"))
        h.append(AIMessage(content=f"a-{i}"))
    agent._storage.save(agent.session_id, h)


@pytest.mark.asyncio
async def test_invoked_skills_reinjected_into_history(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_history(agent)
    skill = Skill(
        name="ping",
        description="ping",
        body="PING-BODY-CONTENT",
        source_path=tmp_path / "ping.md",
        layer="project",
    )
    agent.record_skill_invocation(skill)

    await agent.compact(source="manual")

    history = agent._storage.load(agent.session_id)
    blob = "\n".join(str(m.content) for m in history)
    assert '<skill-active name="ping">' in blob
    assert "PING-BODY-CONTENT" in blob


@pytest.mark.asyncio
async def test_state_custom_holds_preserved_invoked_skills(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_history(agent)
    skill = Skill(
        name="ping",
        description="ping",
        body="P",
        source_path=tmp_path / "ping.md",
        layer="project",
    )
    agent.record_skill_invocation(skill)

    await agent.compact(source="manual")

    preserved = agent._state.custom.get("preserved_invoked_skills")
    assert preserved is not None
    names = [s.name for s in preserved]
    assert names == ["ping"]


@pytest.mark.asyncio
async def test_skill_body_truncated_when_oversize(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_history(agent)
    huge_body = "Z" * (5_000 * 4 + 100)  # exceeds the 5_000-token cap
    skill = Skill(
        name="big",
        description="big",
        body=huge_body,
        source_path=tmp_path / "big.md",
        layer="project",
    )
    agent.record_skill_invocation(skill)

    await agent.compact(source="manual")

    history = agent._storage.load(agent.session_id)
    skill_msg = next(
        m for m in history if "<skill-active" in str(m.content)
    )
    body = str(skill_msg.content)
    assert "… (truncated)" in body
    # The cap is 5000 * 4 = 20000 chars + envelope/marker; original was 20100
    # of Z's so the rendered Z-run must be capped.
    z_run = body.count("Z")
    assert z_run <= 5_000 * 4
