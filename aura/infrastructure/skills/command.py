"""SkillCommand — slash command that records a Skill invocation on the Agent.

Registered into the per-session :class:`CommandRegistry` at Agent
construction (one per user-invocable Skill). ``handle()`` delegates to
``agent.record_skill_invocation`` to thread the (rendered) Skill into the
Context's ``_invoked_skills`` list. The Agent is injected at construction,
not via the ``handle(arg, agent)`` parameter — SkillCommand is inherently
per-Agent and the binding removes the cross-Agent footgun.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

from aura.application.commands.types import CommandResult, CommandSource
from aura.application.loop_state import LoopState
from aura.domain.permission.rule import Rule
from aura.domain.permission.session import SessionRuleSet
from aura.domain.skill import Skill
from aura.infrastructure.persistence import journal
from aura.infrastructure.skills.errors import format_missing_args_error
from aura.infrastructure.skills.loader import render_skill_body
from aura.infrastructure.skills.restrict import install_restrict_lease


class _SkillCommandAgent(Protocol):
    @property
    def session_id(self) -> str:
        ...

    @property
    def session_rules(self) -> SessionRuleSet | None:
        ...

    @property
    def state(self) -> LoopState:
        ...

    def record_skill_invocation(self, skill: Skill) -> None:
        ...


def install_skill_allow_rules(
    skill: Skill, session_rules: SessionRuleSet | None,
) -> None:
    """Install one tool-wide ``Rule`` per ``skill.allowed_tools`` entry.

    Permissive auto-allow: declared tools are granted without prompting for
    the rest of the session. Shared by slash + tool surfaces so both apply
    the same side-effect. Idempotent — re-invoking the skill does not
    duplicate rules, and ``skill_auto_allow_installed`` is journaled only
    for newly added rules.
    """
    if session_rules is None or not skill.allowed_tools:
        return
    existing = set(session_rules.rules())
    for tool_name in sorted(skill.allowed_tools):
        rule = Rule(tool=tool_name, content=None)
        if rule in existing:
            continue
        session_rules.add(rule)
        journal.write(
            "skill_auto_allow_installed",
            skill_name=skill.name, tool=tool_name, source_layer=skill.layer,
        )


class SkillCommand:
    source: CommandSource = "skill"

    def __init__(self, *, skill: Skill, agent: _SkillCommandAgent) -> None:
        self._skill = skill
        self._agent = agent
        self.name = f"/{skill.name}"
        self.description = skill.description
        # Sorted tuple for deterministic snapshot tests; matches the
        # registry's sorted-by-name list() invariant.
        self.allowed_tools: tuple[str, ...] = tuple(sorted(skill.allowed_tools))
        self.argument_hint: str | None = skill.argument_hint

    async def handle(self, arg: str, agent: _SkillCommandAgent) -> CommandResult:
        arg_values = arg.split() if arg.strip() else []
        declared = self._skill.arguments
        if declared and len(arg_values) < len(declared):
            return CommandResult(
                handled=True, kind="print",
                text=format_missing_args_error(
                    self._skill.name, declared, len(arg_values),
                ),
            )
        rendered_body = render_skill_body(
            self._skill, session_id=self._agent.session_id,
            argument_values=arg_values,
        )
        # Clone with the rendered body so Context's dedup-by-source_path
        # keeps working but the copy carries the substituted text.
        invoked = dataclasses.replace(self._skill, body=rendered_body)
        self._agent.record_skill_invocation(invoked)
        install_skill_allow_rules(self._skill, self._agent.session_rules)
        install_restrict_lease(self._skill, self._agent.state)
        journal.write(
            "skill_invoked",
            name=self._skill.name, invocation="slash",
            source=self._skill.layer,
            allowed_tools=sorted(self._skill.allowed_tools),
        )
        return CommandResult(
            handled=True, kind="print",
            text=f"skill invoked: {self._skill.name}",
        )
