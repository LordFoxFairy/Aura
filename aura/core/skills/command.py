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
from aura.core.skills.errors import format_missing_args_error
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
        # Propagate frontmatter metadata onto the Command surface so UI /
        # future permission layers can render/inspect them without
        # reaching back through ``self._skill``. ``allowed_tools`` is a
        # frozenset on Skill; flatten to a sorted tuple for stable
        # ordering across runs (matches ``registry.list()`` sorted-by-name
        # invariant and keeps snapshot tests deterministic).
        self.allowed_tools: tuple[str, ...] = tuple(
            sorted(skill.allowed_tools)
        )
        self.argument_hint: str | None = skill.argument_hint

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        from aura.core.persistence import journal

        # Argument passing on the slash-command path: anything after the
        # slash becomes one whitespace-split list, matching claude-code's
        # default ``substituteArguments`` positional binding. Empty string
        # → no args. We keep it intentionally simple; quoted-args parsing
        # is a v0.8 concern.
        arg_values = arg.split() if arg.strip() else []
        # Symmetric with SkillTool._invoke — missing required args produce
        # a user-visible error instead of a silently-broken body render.
        # Shares ``format_missing_args_error`` with the tool path so the
        # text is byte-for-byte identical across surfaces (users who see
        # the tool error recognise the slash-path variant).
        declared = self._skill.arguments
        if declared and len(arg_values) < len(declared):
            return CommandResult(
                handled=True,
                kind="print",
                text=format_missing_args_error(
                    self._skill.name, declared, len(arg_values),
                ),
            )
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
        # Audit surface for allowed_tools: enforcement is a v0.13 concern
        # (see ``Command.allowed_tools`` docstring); until then the journal
        # event captures declared intent so post-hoc audits can reconstruct
        # what each skill asked to scope down to. ``source`` is the Skill's
        # origin layer (user / project / managed); ``invocation`` is the
        # path we reached this event through (slash / tool).
        journal.write(
            "skill_invoked",
            name=self._skill.name,
            invocation="slash",
            source=self._skill.layer,
            allowed_tools=sorted(self._skill.allowed_tools),
        )
        return CommandResult(
            handled=True,
            kind="print",
            text=f"skill invoked: {self._skill.name}",
        )
