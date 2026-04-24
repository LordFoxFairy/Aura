"""Tests for v0.13 runtime enforcement of Skill ``allowed_tools``.

Semantic under test: permissive auto-allow (matches claude-code's
``context.alwaysAllowRules.command``). Invoking a skill installs one
session-scoped ``AllowRule(tool=name)`` per entry in the skill's
``allowed-tools`` frontmatter. Rules persist until ``/clear`` drops the
session ruleset (same scope as any user-added session rule).

Coverage:

1. SkillCommand.handle installs rules for each declared tool.
2. Re-invoking the same skill is idempotent (rules not duplicated).
3. Undeclared tools still reach the asker (permissive, not restrictive).
4. Declared tools auto-allow via the rule path (``rule_allow``).
5. SkillTool._invoke installs rules symmetrically with the slash path.
6. Empty ``allowed_tools`` adds no rules.
7. ``skill_auto_allow_installed`` journal event fires once per rule
   actually installed (not once per skill invocation).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.hooks.permission import (
    AskerResponse,
    make_permission_hook,
)
from aura.core.permissions.rule import Rule
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.core.persistence import journal as journal_module
from aura.core.persistence.storage import SessionStorage
from aura.core.skills.command import SkillCommand
from aura.core.skills.types import Skill
from aura.schemas.state import LoopState
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel

# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------


def _skill(
    name: str = "foo",
    *,
    allowed_tools: frozenset[str] = frozenset(),
) -> Skill:
    return Skill(
        name=name,
        description=f"Description of {name}.",
        body=f"# Body of {name}",
        source_path=Path(f"/tmp/{name}.md"),
        layer="user",
        allowed_tools=allowed_tools,
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
    """Minimal BaseTool for probing the permission hook's decision."""

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


# ---------------------------------------------------------------------------
# 1. Slash-command path installs allow-rules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_invocation_installs_allow_rules(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill(
            "helper",
            allowed_tools=frozenset({"read_file", "grep"}),
        )
        cmd = SkillCommand(skill=skill, agent=agent)
        await cmd.handle("", agent)

        installed = {r.tool for r in session.rules()}
        assert installed == {"read_file", "grep"}
        # All installed rules are tool-wide (content=None) — this is an
        # auto-allow, not a pattern gate.
        for r in session.rules():
            assert r.content is None
    finally:
        await agent.aclose()


# ---------------------------------------------------------------------------
# 2. Idempotent re-invocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_invocation_idempotent(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill("dup", allowed_tools=frozenset({"grep"}))
        cmd = SkillCommand(skill=skill, agent=agent)
        await cmd.handle("", agent)
        await cmd.handle("", agent)
        rules = session.rules()
        assert len(rules) == 1
        assert rules[0].tool == "grep"
    finally:
        await agent.aclose()


# ---------------------------------------------------------------------------
# 3. Undeclared tool still reaches asker (permissive, not restrictive)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undeclared_tool_still_prompts(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill(
            "narrow", allowed_tools=frozenset({"read_file"}),
        )
        cmd = SkillCommand(skill=skill, agent=agent)
        await cmd.handle("", agent)

        # After skill invocation, a call to ``bash`` (NOT declared) must
        # still fall through to the asker — permissive semantics means
        # undeclared tools are not silently denied.
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
            state=LoopState(),
        )
        # Asker was consulted → no rule short-circuited the ask path.
        assert len(asker.calls) == 1
        assert asker.calls[0]["tool"] == "bash"
        # Because the spy answers "accept", the hook should allow.
        assert outcome.short_circuit is None
    finally:
        await agent.aclose()


# ---------------------------------------------------------------------------
# 4. Declared tool auto-allowed via rule_allow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_declared_tool_auto_allowed(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill("greppy", allowed_tools=frozenset({"grep"}))
        cmd = SkillCommand(skill=skill, agent=agent)
        await cmd.handle("", agent)

        # Next turn: LLM calls ``grep`` → permission hook must return
        # ``rule_allow`` without consulting the asker.
        asker = _SpyAsker()
        hook = make_permission_hook(
            asker=asker,
            session=session,
            rules=RuleSet(),
            project_root=tmp_path,
        )
        grep_tool = _build_probe_tool("grep")
        outcome = await hook(
            tool=grep_tool,
            args={},
            state=LoopState(),
        )
        assert outcome.short_circuit is None
        assert asker.calls == []
        assert outcome.decision is not None
        assert outcome.decision.reason == "rule_allow"
        assert outcome.decision.rule is not None
        assert outcome.decision.rule.tool == "grep"
    finally:
        await agent.aclose()


# ---------------------------------------------------------------------------
# 5. Tool path (SkillTool._invoke) installs rules too
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_path_invocation_also_installs_rules(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill(
            "via-tool",
            allowed_tools=frozenset({"read_file", "bash"}),
        )
        agent._skill_registry.register(skill)
        # Use the live SkillTool wired on the Agent so we exercise the
        # same injector path the real model-driven code takes.
        tool = agent._available_tools["skill"]
        result = tool.invoke({"name": "via-tool"})
        assert result["invoked"] is True

        installed = {r.tool for r in session.rules()}
        assert installed == {"read_file", "bash"}
    finally:
        await agent.aclose()


# ---------------------------------------------------------------------------
# 6. Empty allowed_tools → no rules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allowed_tools_empty_no_rules_added(tmp_path: Path) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill("bare")  # no allowed_tools
        cmd = SkillCommand(skill=skill, agent=agent)
        await cmd.handle("", agent)
        assert session.rules() == ()
    finally:
        await agent.aclose()


# ---------------------------------------------------------------------------
# 7. Journal event per installed rule
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_auto_allow_installed_journal_event(
    tmp_path: Path, journal_path: Path,
) -> None:
    session = SessionRuleSet()
    agent = _agent(tmp_path, session_rules=session)
    try:
        skill = _skill(
            "audited",
            allowed_tools=frozenset({"read_file", "grep", "glob"}),
        )
        cmd = SkillCommand(skill=skill, agent=agent)
        await cmd.handle("", agent)

        events = _journal_events(journal_path)
        installed = [
            e for e in events if e.get("event") == "skill_auto_allow_installed"
        ]
        # Three distinct tools → three events.
        assert len(installed) == 3
        tools_logged = {e["tool"] for e in installed}
        assert tools_logged == {"read_file", "grep", "glob"}
        # Every event carries skill name + source layer for audit.
        for e in installed:
            assert e["skill_name"] == "audited"
            assert e["source_layer"] == "user"

        # Re-invoking the same skill installs no NEW rules → no new events.
        await cmd.handle("", agent)
        events2 = _journal_events(journal_path)
        installed2 = [
            e for e in events2 if e.get("event") == "skill_auto_allow_installed"
        ]
        assert len(installed2) == 3  # unchanged — idempotent
    finally:
        await agent.aclose()
