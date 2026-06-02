"""skill — inject a predefined Skill into the next turn's context."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.application.loop_state import LoopState
from aura.domain.permission.session import SessionRuleSet
from aura.domain.skill import Skill
from aura.domain.tool import ToolError, ToolMetadata
from aura.infrastructure.persistence import journal
from aura.infrastructure.skills.command import install_skill_allow_rules
from aura.infrastructure.skills.errors import format_missing_args_error
from aura.infrastructure.skills.loader import render_skill_body
from aura.infrastructure.skills.registry import SkillRegistry
from aura.infrastructure.skills.restrict import install_restrict_lease

SkillRecorder = Callable[[Skill], None]
SessionIdProvider = Callable[[], str]
SessionRulesProvider = Callable[[], "SessionRuleSet | None"]
LoopStateProvider = Callable[[], "LoopState | None"]


class SkillParams(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        description="Skill name; case-sensitive; no leading slash.",
    )
    arguments: list[str] | None = Field(
        default=None,
        description="Positional arg values matching the skill's declared arguments.",
    )


class SkillResult(TypedDict):
    skill: str
    invoked: Literal[True]
    source: str


def _preview(args: dict[str, Any]) -> str:
    name = args.get("name", "")
    arg_list = args.get("arguments") or []
    if arg_list:
        return f"skill: {name}({' '.join(str(a) for a in arg_list)})"
    return f"skill: {name}"


class SkillTool(BaseTool):
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
    args_schema: type[BaseModel] = SkillParams  # pyright: ignore[reportIncompatibleVariableOverride]  # langchain BaseTool declares args_schema as mutable ArgsSchema|None; subclass narrows widely on purpose.
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        rule_matcher=None,
        args_preview=_preview,
        timeout_sec=None,
    )
    _recorder: SkillRecorder = PrivateAttr()
    _registry: SkillRegistry = PrivateAttr()
    _session_id_provider: SessionIdProvider = PrivateAttr()
    _session_rules_provider: SessionRulesProvider = PrivateAttr()
    _loop_state_provider: LoopStateProvider = PrivateAttr()

    def __init__(
        self,
        *,
        recorder: SkillRecorder,
        registry: SkillRegistry,
        session_id_provider: SessionIdProvider | None = None,
        session_rules_provider: SessionRulesProvider | None = None,
        loop_state_provider: LoopStateProvider | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._recorder = recorder
        self._registry = registry
        self._session_id_provider = session_id_provider or (lambda: "default")
        self._session_rules_provider = (
            session_rules_provider or (lambda: None)
        )
        self._loop_state_provider = (
            loop_state_provider or (lambda: None)
        )

    def _run(
        self, name: str, arguments: list[str] | None = None,
    ) -> SkillResult:
        return self._invoke(name, arguments)

    async def _arun(
        self, name: str, arguments: list[str] | None = None,
    ) -> SkillResult:
        return self._invoke(name, arguments)

    def _invoke(
        self, name: str, arguments: list[str] | None,
    ) -> SkillResult:
        skill = self._registry.get(name)
        # disable_model_invocation skills surface as "missing" so retry can't tell hidden vs absent.
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

        declared = skill.arguments
        values = list(arguments) if arguments else []
        if declared and len(values) < len(declared):
            raise ToolError(
                format_missing_args_error(name, declared, len(values))
            )

        rendered_body = render_skill_body(
            skill,
            session_id=self._session_id_provider(),
            argument_values=values if declared else [],
        )
        invoked_skill = dataclasses.replace(skill, body=rendered_body)
        self._recorder(invoked_skill)
        install_skill_allow_rules(skill, self._session_rules_provider())
        loop_state = self._loop_state_provider()
        if loop_state is not None:
            install_restrict_lease(skill, loop_state)
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
