"""SkillCommand — a slash command that records a Skill invocation on the Agent.

The command is registered into the per-session :class:`CommandRegistry` at
Agent construction time (one SkillCommand per user-invocable Skill).
``handle()`` delegates to ``agent.record_skill_invocation``, which threads a
Skill (with body variables already substituted) into the Context's
append-only ``_invoked_skills`` list.

``agent`` is injected at construction (mirrors ``HelpCommand`` which takes a
``registry``). We chose construction-time injection over the ``handle(arg,
agent)`` parameter even though the parameter is available: SkillCommand is
inherently per-Agent (the Skill's invocation state lives in that Agent's
Context), so binding at construction removes the asymmetry where ``handle``
could be called with a different Agent than the one we're registered on.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from aura.core.commands.types import CommandResult, CommandSource
from aura.core.skills.loader import render_skill_body
from aura.core.skills.types import Skill

if TYPE_CHECKING:
    from aura.core.agent import Agent


class SkillCommand:
    source: CommandSource = "skill"

    def __init__(self, *, skill: Skill, agent: Agent) -> None:
        self._skill = skill
        self._agent = agent
        self.name = f"/{skill.name}"
        self.description = skill.description

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        from aura.core.persistence import journal

        # Argument passing on the slash-command path: anything after the
        # slash becomes one whitespace-split list, matching claude-code's
        # default ``substituteArguments`` positional binding. Empty string
        # → no args. We keep it intentionally simple; quoted-args parsing
        # is a v0.8 concern.
        arg_values = arg.split() if arg.strip() else []
        rendered_body = render_skill_body(
            self._skill,
            session_id=self._agent.session_id,
            argument_values=arg_values,
        )
        # Clone the Skill with the rendered body so Context's existing
        # dedup-by-source_path keeps working (same path → same skill) but
        # the rendered copy carries the substituted text.
        invoked = dataclasses.replace(self._skill, body=rendered_body)
        self._agent.record_skill_invocation(invoked)
        journal.write(
            "skill_invoked", name=self._skill.name, source="slash",
        )
        return CommandResult(
            handled=True,
            kind="print",
            text=f"skill invoked: {self._skill.name}",
        )
