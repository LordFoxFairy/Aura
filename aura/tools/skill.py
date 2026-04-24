"""skill — LLM-invocable tool that injects a predefined Skill into the next turn.

Companion to the user-facing ``/<skill-name>`` slash command. Both paths end
at :meth:`Agent.record_skill_invocation`, which threads the Skill into the
Context's append-only ``_invoked_skills`` list — the body is rendered as a
``<skill-invoked>`` HumanMessage on the next ``Context.build``.

Argument passing mirrors claude-code's slash-command arg contract: the LLM
optionally supplies an ``arguments: [...]`` list; placeholders named after
the skill's ``arguments:`` frontmatter (``${arg-name}``) are substituted
into the body before it's recorded. A skill that declares no arguments
ignores any arguments the LLM passes (don't error — the LLM may hand over
empty lists defensively). A mismatched arg count raises a ToolError so the
LLM can re-plan.

State is injected at Agent construction time — see ``Agent.__init__``'s
stateful-tools wiring block. Dependencies: a ``recorder`` closure that
proxies to :meth:`Agent.record_skill_invocation`, the ``SkillRegistry``
for name lookup, and a ``session_id_provider`` callable so the body-render
step can substitute ``${AURA_SESSION_ID}``. Passing a provider (vs a bare
string) avoids re-wiring the tool every ``/clear`` (though Aura today does
rebuild it; the callable is cheap insurance against a future refactor).

Read-only + not destructive: invoking a skill only schedules a next-turn
context injection. No filesystem, no network, no side effects outside the
in-memory Context.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.core.skills.errors import format_missing_args_error
from aura.core.skills.loader import render_skill_body
from aura.core.skills.registry import SkillRegistry
from aura.core.skills.types import Skill
from aura.schemas.tool import ToolError, tool_metadata

SkillRecorder = Callable[[Skill], None]
SessionIdProvider = Callable[[], str]


class SkillParams(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        description=(
            "The skill name to invoke. Must match a Skill.name registered in "
            "the SkillRegistry (see <skills-available> in the pinned context "
            "for the catalogue). Names are case-sensitive and do NOT include "
            "the leading slash used by the human-typed ``/<name>`` command."
        ),
    )
    arguments: list[str] | None = Field(
        default=None,
        description=(
            "Positional argument values for skills that declare an "
            "``arguments:`` frontmatter field. Order matches the declared "
            "names. Pass null / omit when the skill takes no arguments. "
            "Extra values beyond the declared count are ignored; too few "
            "values raises a ToolError naming the missing argument."
        ),
    )


def _preview(args: dict[str, Any]) -> str:
    name = args.get("name", "")
    arg_list = args.get("arguments") or []
    if arg_list:
        return f"skill: {name}({' '.join(str(a) for a in arg_list)})"
    return f"skill: {name}"


class SkillTool(BaseTool):
    """Invoke a predefined skill by name."""

    # ``SkillRecorder`` is a bare Callable alias; ``SkillRegistry`` is a
    # plain class (not a pydantic model). Same rationale as EnterPlanMode.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "skill"
    description: str = (
        "Invoke a predefined skill by name. The skill's instructions are "
        "injected into the next turn's context so you can execute against "
        "them. Use this when the current task matches a skill's purpose. "
        "See the <skills-available> block in your context for the catalogue "
        "of registered skills (with optional 'when to use' guidance) and "
        "their declared arguments."
    )
    args_schema: type[BaseModel] = SkillParams
    metadata: dict[str, Any] | None = tool_metadata(
        # Read-only: only mutates in-memory Context (next-turn injection),
        # no filesystem / network side effects. Not destructive for the
        # same reason. Concurrency-safe: the recorder dedups by
        # source_path so even racing invocations collapse cleanly.
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        max_result_size_chars=2000,
        args_preview=_preview,
    )
    # PrivateAttr: pydantic would try to coerce bare callables / non-model
    # types as fields; stash them behind the model instead.
    _recorder: SkillRecorder = PrivateAttr()
    _registry: SkillRegistry = PrivateAttr()
    _session_id_provider: SessionIdProvider = PrivateAttr()

    def __init__(
        self,
        *,
        recorder: SkillRecorder,
        registry: SkillRegistry,
        session_id_provider: SessionIdProvider | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._recorder = recorder
        self._registry = registry
        # Default to a constant "default" session id when no provider was
        # wired — keeps unit tests that construct SkillTool directly (no
        # Agent) simple; Agent.__init__ always passes a real provider.
        self._session_id_provider = session_id_provider or (lambda: "default")

    def _run(
        self, name: str, arguments: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._invoke(name, arguments)

    async def _arun(
        self, name: str, arguments: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._invoke(name, arguments)

    def _invoke(
        self, name: str, arguments: list[str] | None,
    ) -> dict[str, Any]:
        skill = self._registry.get(name)
        # ``disable_model_invocation=True`` skills are hidden from the
        # model's <skills-available> catalogue (see
        # SkillRegistry.model_visible). If the model somehow references
        # one by name anyway (stale context, hand-authored transcript),
        # refuse the invocation — matching claude-code's contract that
        # ``disable-model-invocation`` is the hard "model cannot call this
        # skill" flag. The error shape mirrors the unknown-skill branch so
        # the model's retry logic treats "hidden" the same as "missing".
        if skill is not None and skill.disable_model_invocation:
            available = [s.name for s in self._registry.model_visible()]
            raise ToolError(
                f"no skill named {name!r}; available: {available}"
            )
        if skill is None:
            available = [s.name for s in self._registry.model_visible()]
            raise ToolError(
                f"no skill named {name!r}; available: {available}"
            )

        # Validate argument count against the skill's declared ``arguments``.
        declared = skill.arguments
        values = list(arguments) if arguments else []
        if declared and len(values) < len(declared):
            raise ToolError(
                format_missing_args_error(name, declared, len(values))
            )
        # Skills with NO declared arguments silently ignore extras — the
        # LLM may defensively pass [] or unrelated lists and we don't want
        # to error for the "pass-through" case.

        rendered_body = render_skill_body(
            skill,
            session_id=self._session_id_provider(),
            argument_values=values if declared else [],
        )
        invoked_skill = dataclasses.replace(skill, body=rendered_body)
        self._recorder(invoked_skill)
        # Mirror SkillCommand.handle's audit emit — same event shape,
        # same ``allowed_tools`` and origin-layer ``source`` — so the
        # audit trail captures declared intent regardless of which path
        # (slash vs tool) invoked the skill. ``invocation="tool"``
        # distinguishes the model-driven path from the user's slash
        # command. Enforcement of ``allowed_tools`` remains a v0.13
        # concern (see ``Command.allowed_tools`` docstring).
        from aura.core.persistence import journal

        journal.write(
            "skill_invoked",
            name=skill.name,
            invocation="tool",
            source=skill.layer,
            allowed_tools=sorted(skill.allowed_tools),
        )
        return {
            "skill": skill.name,
            "invoked": True,
            "source": str(skill.source_path),
        }
