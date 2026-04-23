"""skill — LLM-invocable tool that injects a predefined Skill into the next turn.

Companion to the user-facing ``/<skill-name>`` slash command. Both paths end
at :meth:`Agent.record_skill_invocation`, which threads the Skill into the
Context's append-only ``_invoked_skills`` list — the body is rendered as a
``<skill-invoked>`` HumanMessage on the next ``Context.build``. The LLM sees
the catalogue via the already-pinned ``<skills-available>`` block, so this
tool's description can stay static (no snapshot baked at registration time —
hot-reload would otherwise desync the description vs the live registry).

State is injected at Agent construction time — see ``Agent.__init__``'s
stateful-tools wiring block. The dependencies here are a ``recorder``
closure that proxies to :meth:`Agent.record_skill_invocation` and the
``SkillRegistry`` for name lookup. Kept as closures rather than a direct
Agent handle so the tool's reach into the Agent is one arrow only:
"record this skill".

Read-only + not destructive: invoking a skill only schedules a
next-turn context injection. No filesystem, no network, no side effects
outside the in-memory Context.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.core.skills.registry import SkillRegistry
from aura.core.skills.types import Skill
from aura.schemas.tool import ToolError, tool_metadata

SkillRecorder = Callable[[Skill], None]


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


def _preview(args: dict[str, Any]) -> str:
    return f"skill: {args.get('name', '')}"


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
        "of registered skills and their descriptions."
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

    def __init__(
        self,
        *,
        recorder: SkillRecorder,
        registry: SkillRegistry,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._recorder = recorder
        self._registry = registry

    def _run(self, name: str) -> dict[str, Any]:
        return self._invoke(name)

    async def _arun(self, name: str) -> dict[str, Any]:
        return self._invoke(name)

    def _invoke(self, name: str) -> dict[str, Any]:
        skill = self._registry.get(name)
        if skill is None:
            available = [s.name for s in self._registry.list()]
            raise ToolError(
                f"no skill named {name!r}; available: {available}"
            )
        self._recorder(skill)
        return {
            "skill": skill.name,
            "invoked": True,
            "source": str(skill.source_path),
        }
