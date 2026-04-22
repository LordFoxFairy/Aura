"""SkillCommand — a slash command that records a Skill invocation on the Agent.

The command is registered into the global :class:`CommandRegistry` at Agent
construction time (one SkillCommand per Skill). ``handle()`` delegates to
``agent.record_skill_invocation``, which threads the Skill into the Context's
append-only ``_invoked_skills`` list — the body is rendered as a
``<skill-invoked>`` HumanMessage on the next ``build()``.

``agent`` is injected at construction (mirrors ``HelpCommand`` which takes a
``registry``). We chose construction-time injection over the ``handle(arg,
agent)`` parameter even though the parameter is available: SkillCommand is
inherently per-Agent (the Skill's invocation state lives in that Agent's
Context), so binding at construction removes the asymmetry where ``handle``
could be called with a different Agent than the one we're registered on.
Doing so would be a programming error; not allowing it in the first place
is simpler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aura.core.commands.types import CommandResult, CommandSource
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
        from aura.core import journal

        self._agent.record_skill_invocation(self._skill)
        journal.write("skill_invoked", name=self._skill.name)
        return CommandResult(
            handled=True,
            kind="print",
            text=f"skill invoked: {self._skill.name}",
        )
