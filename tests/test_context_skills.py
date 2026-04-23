"""Tests for Context skill-tag rendering (<skills-available> / <skill-invoked>)."""

from __future__ import annotations

from pathlib import Path

from aura.core.memory.context import Context
from aura.core.memory.rules import RulesBundle
from aura.core.skills.types import Skill


def _skill(
    name: str,
    desc: str = "d",
    body: str = "b",
    *,
    when_to_use: str | None = None,
    paths: frozenset[str] = frozenset(),
    disable_model_invocation: bool = False,
) -> Skill:
    return Skill(
        name=name,
        description=desc,
        body=body,
        source_path=Path(f"/tmp/{name}.md"),
        layer="user",
        when_to_use=when_to_use,
        paths=paths,
        disable_model_invocation=disable_model_invocation,
    )


def test_skills_available_tag_rendered_when_non_empty(tmp_path: Path) -> None:
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        skills=[_skill("refactor", "refactor helper"), _skill("summarize", "summarize it")],
    )
    out = ctx.build([])
    contents = [str(m.content) for m in out]
    avail = [c for c in contents if c.startswith("<skills-available>")]
    assert len(avail) == 1
    msg = avail[0]
    assert msg.endswith("</skills-available>")
    assert "- refactor: refactor helper" in msg
    assert "- summarize: summarize it" in msg


def test_skills_available_tag_omitted_when_empty(tmp_path: Path) -> None:
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        skills=[],
    )
    out = ctx.build([])
    for m in out:
        assert "<skills-available>" not in str(m.content)


def test_skill_invoked_appended_after_record_invocation(tmp_path: Path) -> None:
    s = _skill("refactor", "refactor helper", "# Instructions\nRefactor carefully.")
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        skills=[s],
    )
    ctx.record_skill_invocation(s)
    out = ctx.build([])
    invoked = [c for c in (str(m.content) for m in out) if c.startswith("<skill-invoked")]
    assert len(invoked) == 1
    msg = invoked[0]
    assert msg.startswith('<skill-invoked name="refactor">')
    assert msg.endswith("</skill-invoked>")
    assert "# Instructions" in msg
    assert "Refactor carefully." in msg


def test_multiple_invoked_skills_each_render_own_tag(tmp_path: Path) -> None:
    s1 = _skill("a", "desc a", "A-BODY")
    s2 = _skill("b", "desc b", "B-BODY")
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        skills=[s1, s2],
    )
    ctx.record_skill_invocation(s1)
    ctx.record_skill_invocation(s2)
    # Invoke same skill twice → dedup; still one tag per distinct skill.
    ctx.record_skill_invocation(s1)

    out = ctx.build([])
    invoked = [c for c in (str(m.content) for m in out) if c.startswith("<skill-invoked")]
    assert len(invoked) == 2
    # Preserve invocation order.
    assert invoked[0].startswith('<skill-invoked name="a">')
    assert "A-BODY" in invoked[0]
    assert invoked[1].startswith('<skill-invoked name="b">')
    assert "B-BODY" in invoked[1]


def test_invoked_skills_positioned_after_skills_available_before_history(
    tmp_path: Path,
) -> None:
    s = _skill("helper", "h", "H-BODY")
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        skills=[s],
    )
    ctx.record_skill_invocation(s)
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

    history: list[BaseMessage] = [HumanMessage("u1"), AIMessage("a1")]
    out = ctx.build(history)
    contents = [str(m.content) for m in out]

    avail_idx = next(i for i, c in enumerate(contents) if c.startswith("<skills-available>"))
    inv_idx = next(i for i, c in enumerate(contents) if c.startswith("<skill-invoked"))
    assert avail_idx < inv_idx
    assert out[inv_idx + 1] is history[0]
    assert out[inv_idx + 2] is history[1]


def test_skills_available_includes_when_to_use_when_present(tmp_path: Path) -> None:
    """``when_to_use`` frontmatter is rendered inline as triggering guidance."""
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        skills=[
            _skill("refactor", "refactor helper", when_to_use="when code is messy"),
            _skill("summarize", "summarize it"),  # no when_to_use
        ],
    )
    out = ctx.build([])
    avail = next(
        c for c in (str(m.content) for m in out) if c.startswith("<skills-available>")
    )
    # Skills with when_to_use carry the [when to use: ...] suffix.
    assert "- refactor: refactor helper [when to use: when code is messy]" in avail
    # Skills without it render as before (plain name: description).
    assert "- summarize: summarize it" in avail
    # No trailing "[when to use: ..." on the summarize line.
    for line in avail.splitlines():
        if line.startswith("- summarize:"):
            assert "when to use" not in line


def test_disable_model_invocation_skills_hidden_from_skills_available(
    tmp_path: Path,
) -> None:
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        skills=[
            _skill("public", "public desc"),
            _skill("hidden", "hidden desc", disable_model_invocation=True),
        ],
    )
    out = ctx.build([])
    avail = next(
        c for c in (str(m.content) for m in out) if c.startswith("<skills-available>")
    )
    assert "public" in avail
    assert "hidden" not in avail


def test_conditional_skills_not_in_skills_available_until_activated(
    tmp_path: Path,
) -> None:
    """Conditional skills are hidden while they sit in the lazy bucket."""
    # A skill whose ``paths`` is non-empty is conditional — the loader
    # wouldn't have added it to the registry yet, but an integration test
    # passing it directly to Context should still see it filtered.
    conditional = _skill("lazy", paths=frozenset({"src/**"}))
    unconditional = _skill("eager")
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        skills=[conditional, unconditional],
    )
    out = ctx.build([])
    avail = next(
        c for c in (str(m.content) for m in out) if c.startswith("<skills-available>")
    )
    assert "eager" in avail
    assert "lazy" not in avail


def test_skills_available_omitted_when_only_hidden_skills(tmp_path: Path) -> None:
    """All skills filtered → no empty tag. Keeps token budget tidy."""
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        skills=[_skill("h", disable_model_invocation=True)],
    )
    out = ctx.build([])
    for m in out:
        assert "<skills-available>" not in str(m.content)


def test_clear_session_resets_invoked_skills(tmp_path: Path) -> None:
    """New Context = fresh _invoked_skills; skills_available persists via ctor arg."""
    s = _skill("k")
    ctx1 = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        skills=[s],
    )
    ctx1.record_skill_invocation(s)
    out1 = ctx1.build([])
    assert any(str(m.content).startswith("<skill-invoked") for m in out1)

    # Simulate /clear: rebuild Context with same skills list, same way Agent does.
    ctx2 = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        skills=[s],
    )
    out2 = ctx2.build([])
    assert not any(str(m.content).startswith("<skill-invoked") for m in out2)
    # But <skills-available> is still present.
    assert any(str(m.content).startswith("<skills-available>") for m in out2)
