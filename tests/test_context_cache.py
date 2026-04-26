"""F-03-002 + F-03-014 — prompt-cache breakpoint detection in Context.build."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import ConfigDict

from aura.core.memory.context import Context
from aura.core.memory.rules import Rule, RulesBundle
from aura.core.persistence import journal
from aura.core.skills.types import Skill


class _AnthropicLikeModel(BaseChatModel):
    """Stub model that reports ``_llm_type = "anthropic-chat"``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "anthropic-chat"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        **_: Any,
    ) -> Runnable[Any, AIMessage]:
        return self

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=""))]
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        raise NotImplementedError


class _OpenAILikeModel(_AnthropicLikeModel):
    """Stub model that reports ``_llm_type = "openai-chat"``."""

    @property
    def _llm_type(self) -> str:
        return "openai-chat"


def _make_context(*, model: BaseChatModel | None, tmp_path: Path) -> Context:
    return Context(
        cwd=tmp_path,
        system_prompt="SYS PROMPT",
        primary_memory="PRIMARY MEMORY",
        rules=RulesBundle(unconditional=[], conditional=[]),
        model=model,
    )


def test_cache_breakpoints_set_for_anthropic_model(tmp_path: Path) -> None:
    ctx = _make_context(model=_AnthropicLikeModel(), tmp_path=tmp_path)
    msgs = ctx.build([])
    assert isinstance(msgs[0], SystemMessage)
    # End of system prefix — cache_control on the SystemMessage.
    assert (
        msgs[0].additional_kwargs.get("cache_control")
        == {"type": "ephemeral"}
    )
    # End of project-memory — cache_control on the <project-memory>
    # HumanMessage.
    project_memory = next(
        m for m in msgs[1:]
        if "<project-memory>" in str(m.content)
    )
    assert (
        project_memory.additional_kwargs.get("cache_control")
        == {"type": "ephemeral"}
    )


def test_cache_breakpoints_skipped_for_openai_model(tmp_path: Path) -> None:
    ctx = _make_context(model=_OpenAILikeModel(), tmp_path=tmp_path)
    msgs = ctx.build([])
    # SystemMessage — no cache_control kwarg.
    assert "cache_control" not in msgs[0].additional_kwargs
    # <project-memory> — also no marker.
    project_memory = next(
        m for m in msgs[1:]
        if "<project-memory>" in str(m.content)
    )
    assert "cache_control" not in project_memory.additional_kwargs


def test_cache_breakpoints_skipped_for_none_model(tmp_path: Path) -> None:
    # Test harnesses (and the Context fixture in conftest) build a
    # Context without a model. That path must not crash and must not
    # stamp any markers.
    ctx = _make_context(model=None, tmp_path=tmp_path)
    msgs = ctx.build([])
    assert "cache_control" not in msgs[0].additional_kwargs


def test_cache_breakpoints_journal_event_emitted(
    tmp_path: Path,
) -> None:
    log = tmp_path / "events.jsonl"
    journal.reset()
    journal.configure(log)
    try:
        ctx = _make_context(model=_AnthropicLikeModel(), tmp_path=tmp_path)
        ctx.build([])
        events = [json.loads(line) for line in log.read_text().splitlines()]
        cache_events = [e for e in events if e["event"] == "cache_breakpoints_set"]
        assert len(cache_events) == 1
        # SystemMessage + project-memory ⇒ count 2.
        assert cache_events[0]["count"] == 2
        assert cache_events[0]["provider"] == "anthropic-chat"
    finally:
        journal.reset()


def test_cache_breakpoints_no_journal_event_for_openai(
    tmp_path: Path,
) -> None:
    log = tmp_path / "events.jsonl"
    journal.reset()
    journal.configure(log)
    try:
        ctx = _make_context(model=_OpenAILikeModel(), tmp_path=tmp_path)
        ctx.build([])
        if log.exists():
            events = [json.loads(line) for line in log.read_text().splitlines()]
            cache_events = [
                e for e in events if e["event"] == "cache_breakpoints_set"
            ]
            assert cache_events == []
    finally:
        journal.reset()


def test_cache_breakpoints_only_system_when_eager_empty(tmp_path: Path) -> None:
    # No primary memory / no unconditional rules ⇒ no <project-memory>
    # message gets emitted, so only the SystemMessage carries a marker.
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(unconditional=[], conditional=[]),
        model=_AnthropicLikeModel(),
    )
    msgs = ctx.build([])
    assert (
        msgs[0].additional_kwargs.get("cache_control")
        == {"type": "ephemeral"}
    )
    # Only the SystemMessage in the prefix; no <project-memory>.
    assert not any(
        "<project-memory>" in str(m.content)
        for m in msgs
    )


def test_cache_breakpoint_with_unconditional_rule_only(tmp_path: Path) -> None:
    # Even with empty primary_memory, an unconditional Rule produces
    # <project-memory> content, so the marker SHOULD land there too.
    rule_path = tmp_path / "rule.md"
    rule_path.write_text("RULE BODY")
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(
            unconditional=[
                Rule(
                    content="RULE BODY",
                    source_path=rule_path,
                    base_dir=tmp_path,
                    globs=(),
                ),
            ],
            conditional=[],
        ),
        model=_AnthropicLikeModel(),
    )
    msgs = ctx.build([])
    project_memory = next(
        m for m in msgs[1:]
        if "<project-memory>" in str(m.content)
    )
    assert (
        project_memory.additional_kwargs.get("cache_control")
        == {"type": "ephemeral"}
    )


# ---------------------------------------------------------------------------
# F-03-014 — <skills-available> belongs INSIDE the cached prefix and renders
# byte-identically across consecutive builds when the registry is stable.
# ---------------------------------------------------------------------------


def _mk_skill(name: str, tmp_path: Path, *, description: str = "desc") -> Skill:
    """Construct a minimal Skill rooted under *tmp_path*.

    Helper kept local to this test module — no test should reach into the
    loader/registry layer just to assemble a stub. ``source_path`` is set
    to a unique path under tmp_path so two skills don't collide.
    """
    src = tmp_path / "skills" / name / "SKILL.md"
    return Skill(
        name=name,
        description=description,
        body="BODY",
        source_path=src,
        layer="user",
    )


def test_skills_section_appears_inside_cached_prefix(tmp_path: Path) -> None:
    """``<skills-available>`` must sit BEFORE the variable-tail content
    AND its message must carry ``cache_control``.
    """
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="PRIMARY",
        rules=RulesBundle(unconditional=[], conditional=[]),
        skills=[_mk_skill("alpha", tmp_path)],
        model=_AnthropicLikeModel(),
    )
    msgs = ctx.build([])
    # Locate the skills-available message.
    skills_idx = next(
        i for i, m in enumerate(msgs)
        if isinstance(m, HumanMessage) and "<skills-available>" in str(m.content)
    )
    project_memory_idx = next(
        i for i, m in enumerate(msgs)
        if "<project-memory>" in str(m.content)
    )
    # Skills land AFTER project-memory but BEFORE any nested-memory / rule /
    # invoked-skill / todos / history messages — i.e. it's part of the
    # static prefix tail.
    assert skills_idx == project_memory_idx + 1
    # The skills-available message itself carries a cache_control marker.
    skills_msg = msgs[skills_idx]
    assert (
        skills_msg.additional_kwargs.get("cache_control")
        == {"type": "ephemeral"}
    )


def test_skills_section_byte_stable_across_two_renders(tmp_path: Path) -> None:
    """Two consecutive ``build`` calls on the same Context produce
    byte-identical ``<skills-available>`` content when the registry is
    unchanged. This is the cache-stability invariant F-03-014 protects.
    """
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="PRIMARY",
        rules=RulesBundle(unconditional=[], conditional=[]),
        skills=[
            _mk_skill("zeta", tmp_path, description="z"),
            _mk_skill("alpha", tmp_path, description="a"),
            _mk_skill("mu", tmp_path, description="m"),
        ],
        model=_AnthropicLikeModel(),
    )
    msgs1 = ctx.build([])
    msgs2 = ctx.build([])
    s1 = next(
        str(m.content) for m in msgs1
        if isinstance(m, HumanMessage) and "<skills-available>" in str(m.content)
    )
    s2 = next(
        str(m.content) for m in msgs2
        if isinstance(m, HumanMessage) and "<skills-available>" in str(m.content)
    )
    assert s1 == s2
    # Sort-by-name normalisation: alpha < mu < zeta in the rendered order.
    body = s1.split("<skills-available>\n", 1)[1]
    body = body.rsplit("\n</skills-available>", 1)[0]
    lines = body.splitlines()
    names_in_order = [ln.split(":")[0].lstrip("- ") for ln in lines]
    assert names_in_order == sorted(names_in_order)


def test_skills_section_changes_when_registry_changes(tmp_path: Path) -> None:
    """Adding a new skill between builds MUST change the rendered text."""
    skills = [_mk_skill("alpha", tmp_path)]
    ctx_before = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="PRIMARY",
        rules=RulesBundle(unconditional=[], conditional=[]),
        skills=list(skills),
        model=_AnthropicLikeModel(),
    )
    s_before = next(
        str(m.content) for m in ctx_before.build([])
        if isinstance(m, HumanMessage) and "<skills-available>" in str(m.content)
    )

    skills.append(_mk_skill("beta", tmp_path))
    ctx_after = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="PRIMARY",
        rules=RulesBundle(unconditional=[], conditional=[]),
        skills=list(skills),
        model=_AnthropicLikeModel(),
    )
    s_after = next(
        str(m.content) for m in ctx_after.build([])
        if isinstance(m, HumanMessage) and "<skills-available>" in str(m.content)
    )
    assert s_before != s_after
    assert "beta" in s_after
    assert "beta" not in s_before


def test_skills_section_omitted_when_no_visible_skills(tmp_path: Path) -> None:
    """An empty skills list ⇒ no <skills-available> message at all
    (no empty block, no stale cache-control marker)."""
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="PRIMARY",
        rules=RulesBundle(unconditional=[], conditional=[]),
        skills=[],
        model=_AnthropicLikeModel(),
    )
    msgs = ctx.build([])
    assert not any(
        isinstance(m, HumanMessage) and "<skills-available>" in str(m.content)
        for m in msgs
    )


def test_skills_section_breakpoint_count_three(tmp_path: Path) -> None:
    """SystemMessage + project-memory + skills-available ⇒ count == 3."""
    log = tmp_path / "events.jsonl"
    journal.reset()
    journal.configure(log)
    try:
        ctx = Context(
            cwd=tmp_path,
            system_prompt="SYS",
            primary_memory="PRIMARY",
            rules=RulesBundle(unconditional=[], conditional=[]),
            skills=[_mk_skill("alpha", tmp_path)],
            model=_AnthropicLikeModel(),
        )
        ctx.build([])
        events = [json.loads(line) for line in log.read_text().splitlines()]
        cache_events = [
            e for e in events if e["event"] == "cache_breakpoints_set"
        ]
        assert len(cache_events) == 1
        assert cache_events[0]["count"] == 3
    finally:
        journal.reset()


def test_skills_section_position_does_not_break_existing_invariants(
    tmp_path: Path,
) -> None:
    """With skills present, project-memory still keeps its breakpoint and
    the SystemMessage's breakpoint remains intact. Defense-in-depth so the
    F-03-014 move doesn't accidentally drop F-03-002 markers."""
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="PRIMARY",
        rules=RulesBundle(unconditional=[], conditional=[]),
        skills=[_mk_skill("alpha", tmp_path)],
        model=_AnthropicLikeModel(),
    )
    msgs = ctx.build([])
    assert (
        msgs[0].additional_kwargs.get("cache_control")
        == {"type": "ephemeral"}
    )
    project_memory = next(
        m for m in msgs[1:]
        if "<project-memory>" in str(m.content)
    )
    assert (
        project_memory.additional_kwargs.get("cache_control")
        == {"type": "ephemeral"}
    )
