"""Tests for the LLM-invocable ``skill`` tool.

Covers:

- Known skill name -> success envelope + recorder called.
- Unknown skill name -> ToolError with an "available: [..]" list.
- Empty name -> pydantic ValidationError before body runs.
- Tool metadata: not destructive, read-only.
- Tool is registered in the default ``tools.enabled`` list.
- Tool is wired on the Agent and invokes through the real recorder.
- Arguments: skill with declared args renders placeholders; missing args
  raises ToolError naming the missing positional; skill with no declared
  args ignores incoming arguments (doesn't error).
- ${AURA_SKILL_DIR} is substituted at tool-invoke time.
- ${AURA_SESSION_ID} is substituted from the injected provider.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from aura.core.skills.registry import SkillRegistry
from aura.core.skills.types import Skill
from aura.schemas.tool import ToolError
from aura.tools.skill import SkillTool
from tests.conftest import FakeChatModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skill(
    name: str = "foo",
    *,
    path: Path | None = None,
    body: str | None = None,
    arguments: tuple[str, ...] = (),
) -> Skill:
    source = path if path is not None else Path(f"/tmp/{name}.md")
    return Skill(
        name=name,
        description=f"Description of {name}.",
        body=body if body is not None else f"# Body of {name}\ndo the thing",
        source_path=source,
        layer="user",
        arguments=arguments,
    )


class _RecorderSpy:
    """Fake recorder mirrors Agent.record_skill_invocation's signature."""

    def __init__(self) -> None:
        self.calls: list[Skill] = []

    def __call__(self, skill: Skill) -> None:
        self.calls.append(skill)


def _tool(
    registry: SkillRegistry,
    spy: _RecorderSpy,
    *,
    session_id: str = "sid-test",
) -> SkillTool:
    return SkillTool(
        recorder=spy,
        registry=registry,
        session_id_provider=lambda: session_id,
    )


# ---------------------------------------------------------------------------
# Unit tests — tool in isolation
# ---------------------------------------------------------------------------


def test_skill_tool_known_name_returns_success_envelope() -> None:
    reg = SkillRegistry([_skill("alpha")])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    result = tool.invoke({"name": "alpha"})
    assert result == {
        "skill": "alpha",
        "invoked": True,
        "source": "/tmp/alpha.md",
    }
    assert len(spy.calls) == 1
    assert spy.calls[0].name == "alpha"


def test_skill_tool_unknown_name_raises_tool_error_with_available_list() -> None:
    reg = SkillRegistry([_skill("alpha"), _skill("beta")])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    with pytest.raises(ToolError) as excinfo:
        tool.invoke({"name": "nope"})
    msg = str(excinfo.value)
    assert "nope" in msg
    assert "available:" in msg
    assert "alpha" in msg
    assert "beta" in msg
    # Recorder never called on failure.
    assert spy.calls == []


def test_skill_tool_empty_name_rejected_by_schema() -> None:
    # min_length=1 on the param — pydantic must reject before the tool body
    # so no recorder call can slip through with an invalid name.
    reg = SkillRegistry([_skill("alpha")])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    with pytest.raises(ValidationError):
        tool.invoke({"name": ""})
    assert spy.calls == []


def test_skill_tool_metadata_is_read_only_and_not_destructive() -> None:
    reg = SkillRegistry()
    tool = _tool(reg, _RecorderSpy())
    meta = tool.metadata or {}
    assert meta.get("is_read_only") is True
    assert meta.get("is_destructive") is False


def test_skill_tool_in_default_enabled_tools() -> None:
    # Regression: the default config ships with skill enabled.
    assert "skill" in AuraConfig().tools.enabled


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def test_skill_tool_substitutes_argument_placeholders(tmp_path: Path) -> None:
    skill = _skill(
        "greet",
        path=tmp_path / "greet.md",
        body="Hello ${who}!",
        arguments=("who",),
    )
    reg = SkillRegistry([skill])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    tool.invoke({"name": "greet", "arguments": ["alice"]})
    assert len(spy.calls) == 1
    # The recorder receives a rendered-body clone; the original skill stays
    # pristine (frozen dataclass + dataclasses.replace).
    assert spy.calls[0].body == "Hello alice!"
    assert skill.body == "Hello ${who}!"


def test_skill_tool_missing_argument_raises_tool_error() -> None:
    skill = _skill("greet", body="Hello ${who}", arguments=("who",))
    reg = SkillRegistry([skill])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    with pytest.raises(ToolError) as excinfo:
        tool.invoke({"name": "greet"})
    msg = str(excinfo.value)
    # Error tells the LLM exactly what's missing so it can re-plan.
    assert "who" in msg
    assert "missing" in msg
    assert spy.calls == []


def test_skill_tool_ignores_arguments_when_skill_declares_none() -> None:
    """LLM may pass [] or irrelevant args defensively — shouldn't error."""
    skill = _skill("nullary")
    reg = SkillRegistry([skill])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    # Extra arguments are silently dropped.
    tool.invoke({"name": "nullary", "arguments": ["extra", "stuff"]})
    assert len(spy.calls) == 1


def test_skill_tool_substitutes_skill_dir_and_session_id(tmp_path: Path) -> None:
    skill_file = tmp_path / "mydir" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\ndescription: d\n---\nDir=${AURA_SKILL_DIR}\nSid=${AURA_SESSION_ID}\n",
        encoding="utf-8",
    )
    skill = Skill(
        name="s",
        description="d",
        body="Dir=${AURA_SKILL_DIR}\nSid=${AURA_SESSION_ID}\n",
        source_path=skill_file,
        layer="user",
    )
    reg = SkillRegistry([skill])
    spy = _RecorderSpy()
    tool = _tool(reg, spy, session_id="my-sid")
    tool.invoke({"name": "s"})
    rendered = spy.calls[0].body
    assert str(skill_file.parent) in rendered
    assert "my-sid" in rendered


# ---------------------------------------------------------------------------
# Integration — tool -> Agent.record_skill_invocation -> Context
# ---------------------------------------------------------------------------


def _make_agent(tmp_path: Path, skills: list[Skill]) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["skill"]},
    })
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "db"),
        pre_loaded_skills=SkillRegistry(skills),
    )


def test_skill_tool_wired_on_agent_flows_into_context_invoked_list(
    tmp_path: Path,
) -> None:
    agent = _make_agent(tmp_path, [_skill("helper")])
    try:
        tool = agent._available_tools["skill"]
        result = tool.invoke({"name": "helper"})
        assert result["invoked"] is True
        assert result["skill"] == "helper"
        messages = agent._context.build([])
        contents = " ".join(str(m.content) for m in messages)
        assert '<skill-invoked name="helper">' in contents
    finally:
        agent.close()


def test_skill_tool_wired_on_agent_dedups_across_double_invocation(
    tmp_path: Path,
) -> None:
    agent = _make_agent(tmp_path, [_skill("once")])
    try:
        tool = agent._available_tools["skill"]
        tool.invoke({"name": "once"})
        tool.invoke({"name": "once"})
        messages = agent._context.build([])
        contents = " ".join(str(m.content) for m in messages)
        assert contents.count('<skill-invoked name="once">') == 1
    finally:
        agent.close()
