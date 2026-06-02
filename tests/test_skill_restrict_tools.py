"""Tests for V14-RESTRICT-TOOLS: skill ``restrict-tools`` strict whitelist.

Semantic under test (distinct from v0.13's permissive ``allowed-tools``):

When a skill declares ``restrict-tools: [read_file, grep]`` in frontmatter,
invoking that skill installs a turn-scoped lease. While the lease is
active:

- Calls to ``read_file`` / ``grep`` follow the normal permission flow —
  they are NOT auto-allowed by ``restrict-tools`` (that is what
  ``allowed-tools`` does); they are simply *not blocked* by the lease.
- Calls to any OTHER tool short-circuit to
  ``Decision(allow=False, reason="restrict_tools_blocked")`` BEFORE
  bypass / safety / rule resolution.
- Internal tools (``ask_user_question``) are exempt — the model-facing
  restrict surface does not block infrastructure plumbing the agent
  itself relies on.

Lease scope: from skill invocation through the end of the model's
response chain that processed the skill's body. Implemented as a
turn-count sentinel on ``LoopState``: install at invocation time
(captures ``state.turn_count``), expire when ``state.turn_count``
advances past the captured value.

Multiple active skills with restrict-tools: union of declared sets
(tool allowed if ANY active skill declares it).

Empty / missing ``restrict-tools``: no lease installed, no restriction.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.application.hooks.permission import make_permission_hook
from aura.application.permission.asker import AskerResponse
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.domain.permission.rule import Rule
from aura.domain.permission.session import RuleSet, SessionRuleSet
from aura.domain.skill import Skill
from aura.domain.tool import ToolResult
from aura.infrastructure.persistence import journal as journal_module
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.skills.command import SkillCommand
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel


def _sc(outcome: object) -> ToolResult | None:
    """Extract the short-circuit result from Replace Outcome."""
    from aura.domain.permission.outcome import Replace
    if isinstance(outcome, Replace):
        return outcome.result
    return getattr(outcome, "short_circuit", None)


def _skill(
    name: str = "foo",
    *,
    restrict_tools: frozenset[str] = frozenset(),
    allowed_tools: frozenset[str] = frozenset(),
) -> Skill:
    return Skill(
        name=name,
        description=f"Description of {name}.",
        body=f"# Body of {name}",
        source_path=Path(f"/tmp/{name}.md"),
        layer="user",
        allowed_tools=allowed_tools,
        restrict_tools=restrict_tools,
    )


def _agent(
    tmp_path: Path,
    *,
    session_rules: SessionRuleSet | None = None,
) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "db"),
        session_rules=session_rules,
    )


@pytest.fixture
def journal_path(tmp_path: Path) -> Iterator[Path]:
    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    yield log
    journal_module.reset()


def _journal_events(log: Path) -> list[dict[str, object]]:
    if not log.exists():
        return []
    return [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class _NoopParams(BaseModel):
    pass


def _build_probe_tool(name: str) -> BaseTool:
    def _noop() -> dict[str, Any]:
        return {}
    return build_tool(
        name=name,
        description=name,
        args_schema=_NoopParams,
        func=_noop,
    )


class _SpyAsker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Rule,
    ) -> AskerResponse:
        self.calls.append({"tool": tool.name, "args": dict(args)})
        return AskerResponse(choice="accept")


@pytest.mark.asyncio
async def test_restrict_tools_blocks_undeclared_tool(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill("narrow", restrict_tools=frozenset({"read_file"}))
        cmd = SkillCommand(skill=skill, agent=agent)
        await cmd.handle("", agent)

        asker = _SpyAsker()
        hook = make_permission_hook(
            asker=asker,
            session=session,
            rules=RuleSet(),
            project_root=tmp_path,
        )
        bash_tool = _build_probe_tool("bash")
        outcome = await hook(
            tool=bash_tool,
            args={},
            state=agent.state,
        )
        # Hook denied + asker never consulted.
        sc = _sc(outcome)
        assert sc is not None
        assert sc.ok is False
        assert asker.calls == []
        from aura.domain.permission.outcome import Block, Replace
        assert isinstance(outcome, (Block, Replace))
        assert outcome.decision is not None
        assert outcome.decision.reason == "restrict_tools_blocked"
        assert outcome.decision.allow is False
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_restrict_tools_allows_declared_tool(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill("read-only", restrict_tools=frozenset({"read_file"}))
        cmd = SkillCommand(skill=skill, agent=agent)
        await cmd.handle("", agent)

        asker = _SpyAsker()
        hook = make_permission_hook(
            asker=asker,
            session=session,
            rules=RuleSet(),
            project_root=tmp_path,
        )
        read_tool = _build_probe_tool("read_file")
        outcome = await hook(
            tool=read_tool,
            args={},
            state=agent.state,
        )
        # restrict_tools does NOT auto-allow — it just doesn't block. The
        # asker is consulted (no rule, no auto-allow path), and answers
        # accept, so the call is allowed via user_accept rather than
        # rule_allow.
        assert len(asker.calls) == 1
        assert asker.calls[0]["tool"] == "read_file"
        assert _sc(outcome) is None
        from aura.domain.permission.outcome import Allow, Block, Replace
        assert isinstance(outcome, (Allow, Block, Replace))
        assert outcome.decision is not None
        assert outcome.decision.reason == "user_accept"
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_lease_expires_when_turn_advances(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        # Pin a known turn for the install-time sentinel.
        agent.state.turn_count = 5

        skill = _skill("ephemeral", restrict_tools=frozenset({"read_file"}))
        cmd = SkillCommand(skill=skill, agent=agent)
        await cmd.handle("", agent)

        asker = _SpyAsker()
        hook = make_permission_hook(
            asker=asker,
            session=session,
            rules=RuleSet(),
            project_root=tmp_path,
        )
        bash_tool = _build_probe_tool("bash")

        # Same turn → lease active → bash is blocked.
        outcome = await hook(
            tool=bash_tool, args={}, state=agent.state,
        )
        from aura.domain.permission.outcome import Block, Replace
        assert isinstance(outcome, (Block, Replace))
        assert outcome.decision is not None
        assert outcome.decision.reason == "restrict_tools_blocked"

        # Advance turn → lease expires → bash falls through to asker.
        agent.state.turn_count = 6
        outcome2 = await hook(
            tool=bash_tool, args={}, state=agent.state,
        )
        assert _sc(outcome2) is None
        assert len(asker.calls) == 1
        assert asker.calls[0]["tool"] == "bash"
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_multiple_active_skills_union(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill_a = _skill("reader", restrict_tools=frozenset({"read_file"}))
        skill_b = _skill("greppy", restrict_tools=frozenset({"grep"}))
        await SkillCommand(skill=skill_a, agent=agent).handle("", agent)
        await SkillCommand(skill=skill_b, agent=agent).handle("", agent)

        asker = _SpyAsker()
        hook = make_permission_hook(
            asker=asker,
            session=session,
            rules=RuleSet(),
            project_root=tmp_path,
        )

        # Either declared tool flows through.
        for declared in ("read_file", "grep"):
            outcome = await hook(
                tool=_build_probe_tool(declared),
                args={},
                state=agent.state,
            )
            assert _sc(outcome) is None, (
                f"{declared!r} should not be blocked"
            )

        # Undeclared tool blocked.
        outcome = await hook(
            tool=_build_probe_tool("bash"),
            args={},
            state=agent.state,
        )
        from aura.domain.permission.outcome import Block, Replace
        assert isinstance(outcome, (Block, Replace))
        assert outcome.decision is not None
        assert outcome.decision.reason == "restrict_tools_blocked"
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_empty_restrict_tools_no_restriction(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill("bare")  # no restrict_tools
        await SkillCommand(skill=skill, agent=agent).handle("", agent)

        asker = _SpyAsker()
        hook = make_permission_hook(
            asker=asker,
            session=session,
            rules=RuleSet(),
            project_root=tmp_path,
        )
        # An arbitrary tool should fall through to the asker — no lease
        # was installed, so no short-circuit fires.
        outcome = await hook(
            tool=_build_probe_tool("bash"),
            args={},
            state=agent.state,
        )
        assert _sc(outcome) is None
        assert len(asker.calls) == 1
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_restrict_blocks_emit_journal_event(
    tmp_path: Path, journal_path: Path,
) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill("audit", restrict_tools=frozenset({"read_file"}))
        await SkillCommand(skill=skill, agent=agent).handle("", agent)

        asker = _SpyAsker()
        hook = make_permission_hook(
            asker=asker,
            session=session,
            rules=RuleSet(),
            project_root=tmp_path,
        )
        await hook(
            tool=_build_probe_tool("bash"),
            args={},
            state=agent.state,
        )

        events = _journal_events(journal_path)
        decisions = [
            e for e in events
            if e.get("event") == "permission_decision"
            and e.get("reason") == "restrict_tools_blocked"
        ]
        assert len(decisions) == 1
        assert decisions[0]["tool"] == "bash"
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_restrict_lease_via_tool_path(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill("via-tool", restrict_tools=frozenset({"read_file"}))
        agent._skill_registry.register(skill)
        tool = agent._available_tools["skill"]
        result = tool.invoke({"name": "via-tool"})
        assert result["invoked"] is True

        asker = _SpyAsker()
        hook = make_permission_hook(
            asker=asker,
            session=session,
            rules=RuleSet(),
            project_root=tmp_path,
        )
        outcome = await hook(
            tool=_build_probe_tool("bash"),
            args={},
            state=agent.state,
        )
        from aura.domain.permission.outcome import Block, Replace
        assert isinstance(outcome, (Block, Replace))
        assert outcome.decision is not None
        assert outcome.decision.reason == "restrict_tools_blocked"
        assert asker.calls == []
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_internal_tools_exempt_from_restrict(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill("locked", restrict_tools=frozenset({"read_file"}))
        await SkillCommand(skill=skill, agent=agent).handle("", agent)

        asker = _SpyAsker()
        hook = make_permission_hook(
            asker=asker,
            session=session,
            rules=RuleSet(),
            project_root=tmp_path,
        )
        # ask_user_question is internal infrastructure (the permission
        # asker itself uses it on some paths). It must NOT be blocked
        # by restrict_tools even though it isn't declared.
        outcome = await hook(
            tool=_build_probe_tool("ask_user_question"),
            args={},
            state=agent.state,
        )
        # Not short-circuited by restrict — flows through to asker.
        from aura.domain.permission.outcome import Allow, Block, Replace
        assert isinstance(outcome, (Allow, Block, Replace))
        assert outcome.decision is not None
        assert outcome.decision.reason != "restrict_tools_blocked"
    finally:
        await agent.aclose()


def test_loader_parses_restrict_tools_frontmatter(tmp_path: Path) -> None:
    from aura.infrastructure.skills.loader import load_skills

    skills_dir = tmp_path / ".aura" / "skills" / "demo"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        """---
description: A demo skill
restrict-tools: [read_file, grep]
---
# Body
""",
        encoding="utf-8",
    )
    registry = load_skills(cwd=tmp_path, home=tmp_path / "no-such-home")
    skill = registry.get("demo")
    assert skill is not None
    assert skill.restrict_tools == frozenset({"read_file", "grep"})
