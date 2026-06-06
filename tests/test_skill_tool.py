"""Tests for the LLM-invocable ``skill`` tool.

Covers:

- Known skill name -> success envelope + recorder called.
- Unknown skill name -> ToolError with an "available: [..]" list.
- Empty name -> pydantic ValidationError before body runs.
- Tool metadata: not destructive, read-only.
- Tool is registered in the default ``tools.enabled`` list.
- Tool is wired on the AgentSession and invokes through the real recorder.
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

from aura.application.loop_state import LoopState
from aura.application.session import AgentSession
from aura.config.schema import AuraConfig
from aura.domain.permission.session import SessionRuleSet
from aura.domain.skill import Skill
from aura.domain.tool import ToolError
from aura.domain.tool_meta_access import meta_dict
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.skills.registry import SkillRegistry
from aura.tools.skill import SkillResult, SkillTool, _preview
from tests.conftest import FakeChatModel


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
    """Fake recorder mirrors AgentSession.record_skill_invocation's signature."""

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
    meta = meta_dict(tool)
    assert meta.get("is_read_only") is True
    assert meta.get("is_destructive") is False


def test_skill_tool_in_default_enabled_tools() -> None:
    # Regression: the default config ships with skill enabled.
    assert "skill" in AuraConfig().tools.enabled


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


def _make_agent(tmp_path: Path, skills: list[Skill]) -> AgentSession:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["skill"]},
    })
    return AgentSession(
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


def _gated_skill(
    name: str = "gated",
    *,
    allowed_tools: frozenset[str] = frozenset(),
    restrict_tools: frozenset[str] = frozenset(),
    disable_model_invocation: bool = False,
) -> Skill:
    """Skill carrying permission/restrict metadata the success path must honour."""
    return Skill(
        name=name,
        description=f"Description of {name}.",
        body=f"# Body of {name}",
        source_path=Path(f"/tmp/{name}.md"),
        layer="user",
        allowed_tools=allowed_tools,
        restrict_tools=restrict_tools,
        disable_model_invocation=disable_model_invocation,
    )


def test_skill_tool_hidden_skill_surfaces_as_missing() -> None:
    """A model-hidden skill must look absent so retry can't probe hidden ones."""
    hidden = _gated_skill("secret", disable_model_invocation=True)
    reg = SkillRegistry([hidden, _skill("public")])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    with pytest.raises(ToolError) as excinfo:
        tool.invoke({"name": "secret"})
    msg = str(excinfo.value)
    assert "secret" in msg
    # The hidden skill itself never appears in the available list it leaks.
    assert "'secret'" not in msg.split("available:")[1]
    assert "public" in msg
    assert spy.calls == []


def test_skill_tool_available_list_excludes_hidden_skills() -> None:
    """Unknown-name error lists only model-visible skills, never hidden ones."""
    reg = SkillRegistry(
        [_skill("shown"), _gated_skill("masked", disable_model_invocation=True)]
    )
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    with pytest.raises(ToolError) as excinfo:
        tool.invoke({"name": "ghost"})
    available = str(excinfo.value).split("available:")[1]
    assert "shown" in available
    assert "masked" not in available


def test_skill_tool_installs_session_allow_rules_for_declared_tools() -> None:
    """A skill's allowed-tools become session auto-allow rules on invocation."""
    rules = SessionRuleSet()
    skill = _gated_skill("granter", allowed_tools=frozenset({"bash", "read"}))
    reg = SkillRegistry([skill])
    spy = _RecorderSpy()
    tool = SkillTool(
        recorder=spy,
        registry=reg,
        session_rules_provider=lambda: rules,
    )
    tool.invoke({"name": "granter"})
    installed = {r.tool for r in rules.rules()}
    assert installed == {"bash", "read"}


def test_skill_tool_allow_rules_are_idempotent_across_double_invoke() -> None:
    """Invoking a skill twice must not duplicate its session allow rules."""
    rules = SessionRuleSet()
    skill = _gated_skill("twice", allowed_tools=frozenset({"bash"}))
    reg = SkillRegistry([skill])
    tool = SkillTool(
        recorder=_RecorderSpy(),
        registry=reg,
        session_rules_provider=lambda: rules,
    )
    tool.invoke({"name": "twice"})
    tool.invoke({"name": "twice"})
    assert len(rules.rules()) == 1


def test_skill_tool_installs_restrict_lease_when_loop_state_present() -> None:
    """A restrict-tools skill leases a turn-scoped whitelist into loop state."""
    state = LoopState(turn_count=3)
    skill = _gated_skill("locked", restrict_tools=frozenset({"read"}))
    reg = SkillRegistry([skill])
    tool = SkillTool(
        recorder=_RecorderSpy(),
        registry=reg,
        loop_state_provider=lambda: state,
    )
    tool.invoke({"name": "locked"})
    leases = state.slots.skill_restrict_leases
    assert len(leases) == 1
    assert leases[0].install_turn == 3
    assert leases[0].tools == frozenset({"read"})


def test_skill_tool_restrict_lease_idempotent_within_one_turn() -> None:
    """Re-invoking the same restrict skill in one turn must not stack leases."""
    state = LoopState(turn_count=0)
    skill = _gated_skill("guard", restrict_tools=frozenset({"read"}))
    reg = SkillRegistry([skill])
    tool = SkillTool(
        recorder=_RecorderSpy(),
        registry=reg,
        loop_state_provider=lambda: state,
    )
    tool.invoke({"name": "guard"})
    tool.invoke({"name": "guard"})
    assert len(state.slots.skill_restrict_leases) == 1


def test_skill_tool_no_restrict_metadata_leaves_loop_state_clean() -> None:
    """A skill without restrict-tools must not write any lease (no-op branch)."""
    state = LoopState(turn_count=1)
    reg = SkillRegistry([_skill("plain")])
    tool = SkillTool(
        recorder=_RecorderSpy(),
        registry=reg,
        loop_state_provider=lambda: state,
    )
    tool.invoke({"name": "plain"})
    assert state.slots.skill_restrict_leases == []


def test_skill_tool_no_allowed_tools_leaves_rules_empty() -> None:
    """A skill with no allowed-tools must not mutate the session rule set."""
    rules = SessionRuleSet()
    reg = SkillRegistry([_skill("bare")])
    tool = SkillTool(
        recorder=_RecorderSpy(),
        registry=reg,
        session_rules_provider=lambda: rules,
    )
    tool.invoke({"name": "bare"})
    assert rules.rules() == ()


async def test_skill_tool_async_path_records_and_returns_envelope() -> None:
    """The async tool entry must behave identically to the sync one."""
    reg = SkillRegistry([_skill("acme")])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    result: SkillResult = await tool.ainvoke({"name": "acme"})
    assert result == {
        "skill": "acme",
        "invoked": True,
        "source": "/tmp/acme.md",
    }
    assert len(spy.calls) == 1
    assert spy.calls[0].name == "acme"


async def test_skill_tool_async_unknown_name_raises_tool_error() -> None:
    """The async path enforces the same unknown-skill guard as the sync path."""
    reg = SkillRegistry([_skill("known")])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    with pytest.raises(ToolError):
        await tool.ainvoke({"name": "absent"})
    assert spy.calls == []


def test_skill_tool_empty_argument_list_with_declared_args_raises() -> None:
    """An explicit empty arguments list still trips the missing-args guard."""
    skill = _skill("needy", body="Hello ${who}", arguments=("who",))
    reg = SkillRegistry([skill])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    with pytest.raises(ToolError) as excinfo:
        tool.invoke({"name": "needy", "arguments": []})
    assert "who" in str(excinfo.value)
    assert spy.calls == []


def test_skill_tool_extra_arguments_beyond_declared_are_truncated() -> None:
    """Surplus positional args are dropped; only declared placeholders render."""
    skill = _skill("solo", body="X=${x}", arguments=("x",))
    reg = SkillRegistry([skill])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    tool.invoke({"name": "solo", "arguments": ["used", "ignored"]})
    assert spy.calls[0].body == "X=used"


def test_skill_tool_invocation_is_idempotent_on_recorder() -> None:
    """Two identical invocations each record once; no silent dedup at the tool."""
    reg = SkillRegistry([_skill("repeat")])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    tool.invoke({"name": "repeat"})
    tool.invoke({"name": "repeat"})
    assert len(spy.calls) == 2


def test_skill_tool_unicode_name_routes_to_registry_lookup() -> None:
    """Non-ASCII skill names must resolve exactly, not be mangled by the lookup."""
    reg = SkillRegistry([_skill("café-技能")])
    spy = _RecorderSpy()
    tool = _tool(reg, spy)
    result = tool.invoke({"name": "café-技能"})
    assert result["skill"] == "café-技能"
    assert result["invoked"] is True


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"name": "do"}, "skill: do"),
        ({"name": "do", "arguments": []}, "skill: do"),
        ({"name": "do", "arguments": None}, "skill: do"),
        ({"name": "do", "arguments": ["a", "b"]}, "skill: do(a b)"),
        ({"name": "do", "arguments": [1, 2]}, "skill: do(1 2)"),
        ({}, "skill: "),
    ],
)
def test_skill_tool_preview_renders_compact_invocation_line(
    args: dict[str, object], expected: str,
) -> None:
    """The args-preview powers the live transcript; it must never raise on edges."""
    assert _preview(args) == expected
