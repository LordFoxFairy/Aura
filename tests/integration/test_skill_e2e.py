"""Integration: SKILL.md → loader → skill tool → <skill-invoked> → next turn.

Unit tests cover each stage in isolation (loader parses, SkillTool
records, Context builds messages). This tier writes a real SKILL.md to
disk, constructs an Agent with ``cwd`` pointing at the tmp dir, and
asserts that after the LLM calls ``skill(...)`` the body the LLM sees
on its NEXT turn contains the promised content.

The probe model is a custom FakeChatModel that captures the ``messages``
argument to ``_agenerate`` so the test can inspect exactly what the LLM
received on turn 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aura.core.skills.loader import activate_conditional_skills_for_paths
from aura.schemas.events import ToolCallCompleted
from tests.conftest import FakeChatModel, FakeTurn
from tests.integration.conftest import build_integration_agent, drain


def _write_skill(root: Path, name: str, frontmatter: str, body: str) -> Path:
    skill_dir = root / ".aura" / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(f"---\n{frontmatter}\n---\n{body}")
    return skill_file


class _CaptureChatModel(FakeChatModel):
    """Fake model that records every ``messages`` arg it's called with.

    Drives turn N from the scripted ``_turns`` queue, same as the base
    class — the only addition is the capture list. Tests inspect
    ``received_messages[N]`` to see what the agent sent on turn N+1.
    """

    def __init__(self, turns: list[FakeTurn]) -> None:
        super().__init__(turns=turns)
        # Can't use a normal attribute due to pydantic's extra=allow config
        # interacting with __dict__; mirror FakeChatModel's pattern.
        self.__dict__["received_messages"] = []

    @property
    def received_messages(self) -> list[list[BaseMessage]]:
        return self.__dict__["received_messages"]  # type: ignore[no-any-return]

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["received_messages"].append(list(messages))
        self.__dict__["ainvoke_calls"] += 1
        turns: list[FakeTurn] = self.__dict__["_turns"]
        turn = turns.pop(0)
        return ChatResult(generations=[ChatGeneration(message=turn.message)])


def _chdir_to(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the skill loader at ``path`` by switching cwd.

    Agent.__init__ reads ``Path.cwd()`` once to seed :func:`load_skills`.
    Test uses monkeypatch.chdir so the rest of the session sees the same
    location (required for path-based conditional-skill activation).
    """
    monkeypatch.chdir(path)


# ---------------------------------------------------------------------------
# Test 1 — skill body appears in next turn's context.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_invocation_injects_body_into_next_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_skill(
        tmp_path,
        "debug",
        "description: Debug this issue.",
        "Follow steps 1-3.",
    )
    _chdir_to(tmp_path, monkeypatch)

    turn_1 = FakeTurn(
        message=AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc_sk",
                    "name": "skill",
                    "args": {"name": "debug"},
                }
            ],
        )
    )
    turn_2 = FakeTurn(message=AIMessage(content="skill applied"))

    capture_model = _CaptureChatModel(turns=[turn_1, turn_2])
    # Use build_integration_agent to get the full Agent machinery but swap in
    # our capture model.
    agent, _ = build_integration_agent(
        tmp_path, [turn_1, turn_2], enabled_tools=["skill"],
    )
    agent._model = capture_model
    agent._loop = agent._build_loop()
    try:
        events = await drain(agent, "use debug")
    finally:
        await agent.aclose()

    # Tool call completed successfully.
    completed = [e for e in events if isinstance(e, ToolCallCompleted)]
    assert len(completed) == 1
    assert completed[0].name == "skill"
    assert completed[0].error is None

    # Turn 2's messages must contain a <skill-invoked> block with the body.
    assert len(capture_model.received_messages) == 2
    turn2_msgs = capture_model.received_messages[1]
    texts = [
        m.content for m in turn2_msgs if isinstance(m.content, str)
    ]
    joined = "\n".join(texts)
    assert "<skill-invoked" in joined
    assert "Follow steps 1-3." in joined


# ---------------------------------------------------------------------------
# Test 2 — skill with arguments substitutes placeholders.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_invocation_substitutes_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_skill(
        tmp_path,
        "my-skill",
        "description: Fix a specific bug.\narguments: [target]",
        "Fix ${target}",
    )
    _chdir_to(tmp_path, monkeypatch)

    turn_1 = FakeTurn(
        message=AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc_sk",
                    "name": "skill",
                    "args": {
                        "name": "my-skill",
                        "arguments": ["auth bug"],
                    },
                }
            ],
        )
    )
    turn_2 = FakeTurn(message=AIMessage(content="fixing"))

    capture_model = _CaptureChatModel(turns=[turn_1, turn_2])
    agent, _ = build_integration_agent(
        tmp_path, [turn_1, turn_2], enabled_tools=["skill"],
    )
    agent._model = capture_model
    agent._loop = agent._build_loop()
    try:
        await drain(agent, "fix auth")
    finally:
        await agent.aclose()

    assert len(capture_model.received_messages) == 2
    turn2_msgs = capture_model.received_messages[1]
    texts = [
        m.content for m in turn2_msgs if isinstance(m.content, str)
    ]
    joined = "\n".join(texts)
    # Substituted body is present; the raw placeholder is not.
    assert "Fix auth bug" in joined
    assert "${target}" not in joined


# ---------------------------------------------------------------------------
# Test 3 — unknown skill → ToolError, LLM can pivot next turn.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_unknown_returns_tool_error_llm_pivots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Install one real skill so the "available" list in the error is non-empty.
    _write_skill(
        tmp_path,
        "alpha",
        "description: Alpha skill.",
        "ALPHA-BODY",
    )
    _chdir_to(tmp_path, monkeypatch)

    turn_1 = FakeTurn(
        message=AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc_sk",
                    "name": "skill",
                    "args": {"name": "nonexistent"},
                }
            ],
        )
    )
    turn_2 = FakeTurn(
        message=AIMessage(content="I'll try a different approach"),
    )

    agent, _ = build_integration_agent(
        tmp_path, [turn_1, turn_2], enabled_tools=["skill"],
    )
    try:
        events = await drain(agent, "try a skill")
    finally:
        await agent.aclose()

    completed = [e for e in events if isinstance(e, ToolCallCompleted)]
    assert len(completed) == 1
    assert completed[0].error is not None
    err = completed[0].error
    # Error must name the bad skill and list the available ones — the
    # LLM-visible failure mode lets the model self-correct.
    assert "nonexistent" in err
    assert "alpha" in err

    # Loop didn't crash — turn 2 ran to completion (Final event present).
    from aura.schemas.events import Final

    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1
    assert finals[0].message == "I'll try a different approach"


# ---------------------------------------------------------------------------
# Test 4 — conditional skill activates only after matching path is touched.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conditional_skill_activation_promotes_to_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conditional skill → not in registry at load → activate for matching
    path → next Agent build has it in the registry.

    The on-disk SKILL.md declares ``paths: ['src/**']`` so the loader
    stashes it in the module-level conditional bucket instead of the
    returned registry. After
    :func:`activate_conditional_skills_for_paths` matches a touched
    path, a rebuilt Agent sees the skill in
    :class:`SkillRegistry` — the promotion is sticky across the process.

    Note: today's :class:`Context` filters out ``is_conditional()`` skills
    from the ``<skills-available>`` block it renders to the LLM, even
    after activation (the Skill dataclass keeps its ``paths`` field
    regardless of activation state). So the observable surface we assert
    is the registry membership, NOT the message-level injection. The
    gap is reported in the integration-test findings.
    """
    skill_file = _write_skill(
        tmp_path,
        "pyhelp",
        "description: Python helper.\npaths:\n  - 'src/**'",
        "PY-BODY",
    )
    assert skill_file.is_file()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "foo.py").write_text("# stub\n")
    _chdir_to(tmp_path, monkeypatch)

    # First Agent: conditional skill is stashed in the module bucket, not
    # the registry.
    from aura.core.skills.loader import get_conditional_skills

    agent1, _ = build_integration_agent(
        tmp_path,
        [FakeTurn(message=AIMessage(content="idle"))],
        enabled_tools=["read_file"],
    )
    try:
        # Bundled skills (F-0910-011) are present at every Agent init; the
        # invariant we care about is that the user's conditional ``pyhelp``
        # is stashed in the module bucket and NOT in the registry yet.
        registry_names = {s.name for s in agent1._skill_registry.list()}
        assert "pyhelp" not in registry_names
        assert [s.name for s in get_conditional_skills()] == ["pyhelp"]
    finally:
        await agent1.aclose()

    # Activate: a tool-style "touch src/foo.py" promotes the skill.
    activated = activate_conditional_skills_for_paths(
        ["src/foo.py"], cwd=tmp_path,
    )
    assert activated == ["pyhelp"]
    # The skill left the conditional bucket.
    assert [s.name for s in get_conditional_skills()] == []

    # A fresh Agent — same cwd — now loads ``pyhelp`` into its registry.
    agent2, _ = build_integration_agent(
        tmp_path,
        [FakeTurn(message=AIMessage(content="idle"))],
        enabled_tools=["read_file"],
    )
    try:
        names = {s.name for s in agent2._skill_registry.list()}
        assert "pyhelp" in names, (
            f"expected 'pyhelp' in registry after activation; got {sorted(names)}"
        )
        # Body survives the activation path (sanity; loader re-parses the
        # file during the second ``load_skills`` call).
        skill = agent2._skill_registry.get("pyhelp")
        assert skill is not None
        assert skill.body.strip() == "PY-BODY"
    finally:
        await agent2.aclose()
