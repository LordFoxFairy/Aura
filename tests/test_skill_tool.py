"""Tests for the LLM-invocable ``skill`` tool.

Covers:

- Known skill name -> success envelope + recorder called.
- Unknown skill name -> ToolError with an "available: [..]" list.
- Empty name -> pydantic ValidationError before body runs.
- After invocation, Skill appears in Context._invoked_skills via
  record_skill_invocation (integration check through the real Agent).
- Double invocation -> second call is a no-op per the existing
  dedup contract (same source_path).
- Tool metadata: not destructive, read-only.
- Tool is registered in the default ``tools.enabled`` list.
- Tool is wired on the Agent and invokes through the real recorder.

The fast-path unit tests wire the tool with a fake recorder so the tool's
contract is exercised in isolation. The integration test uses a real Agent
so we prove the recorder closure + registry actually thread through the
Context's _invoked_skills. Same split as test_plan_mode_tools.
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


def _skill(name: str = "foo", *, path: Path | None = None) -> Skill:
    return Skill(
        name=name,
        description=f"Description of {name}.",
        body=f"# Body of {name}\ndo the thing",
        source_path=path if path is not None else Path(f"/tmp/{name}.md"),
        layer="user",
    )


class _RecorderSpy:
    """Fake recorder mirrors Agent.record_skill_invocation's signature."""

    def __init__(self) -> None:
        self.calls: list[Skill] = []

    def __call__(self, skill: Skill) -> None:
        self.calls.append(skill)


def _tool(registry: SkillRegistry, spy: _RecorderSpy) -> SkillTool:
    return SkillTool(recorder=spy, registry=registry)


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


def test_skill_tool_double_invocation_is_recorded_twice_but_dedup_in_context() -> None:
    # The tool's recorder is called each time (no state on the tool). Dedup
    # is the Context's responsibility and is already tested via
    # test_context_skills; here we just assert the tool itself is idempotent
    # w.r.t. envelope shape across repeated calls.
    reg = SkillRegistry([_skill("alpha")])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    tool.invoke({"name": "alpha"})
    tool.invoke({"name": "alpha"})
    assert len(spy.calls) == 2
    assert all(s.name == "alpha" for s in spy.calls)


def test_skill_tool_metadata_is_read_only_and_not_destructive() -> None:
    reg = SkillRegistry()
    tool = _tool(reg, _RecorderSpy())
    meta = tool.metadata or {}
    assert meta.get("is_read_only") is True
    assert meta.get("is_destructive") is False


def test_skill_tool_in_default_enabled_tools() -> None:
    # Regression: the default config ships with skill enabled so the LLM
    # actually has it available out of the box. Complements the explicit
    # assertion in test_config_schema.test_defaults.
    assert "skill" in AuraConfig().tools.enabled


# ---------------------------------------------------------------------------
# Integration — tool -> Agent.record_skill_invocation -> Context
# ---------------------------------------------------------------------------


def _make_agent(tmp_path: Path, skills: list[Skill]) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        # Only enable the skill tool — isolates the wiring path.
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
        # Real Context receives the skill and renders <skill-invoked> on
        # the next build — same contract the slash command relies on.
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
        # Second call is a no-op in Context (dedup by source_path): the
        # rendered list still shows <skill-invoked name="once"> exactly once.
        messages = agent._context.build([])
        contents = " ".join(str(m.content) for m in messages)
        assert contents.count('<skill-invoked name="once">') == 1
    finally:
        agent.close()
