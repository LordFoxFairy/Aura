"""Spawn an isolated child agent per subagent task."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from aura.application.hooks import HookChain
from aura.application.hooks.permission import make_permission_hook
from aura.application.permission.asker import AskerResponse
from aura.application.tasks.spawn_port import SpawnedAgent
from aura.config.schema import AuraConfig, ToolsConfig
from aura.domain.abort import AbortController
from aura.domain.agent_definition import AGENT_DISALLOWED_TOOLS
from aura.domain.permission.mode import Mode
from aura.domain.permission.rule import Rule
from aura.domain.permission.safety import DEFAULT_SAFETY, SafetyPolicy
from aura.domain.permission.session import RuleSet, SessionRuleSet
from aura.domain.state_values import ReadCarryover
from aura.infrastructure import llm
from aura.infrastructure.agents import get_agent_def
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.skills import SkillRegistry

A = TypeVar("A", bound=SpawnedAgent)

SUBAGENT_AUTO_DENY_FEEDBACK = "subagent_auto_deny"


class _SubagentPermissionAsker:
    """Hook-native asker for subagents: every would-be ask becomes deny."""

    async def __call__(
        self,
        *,
        tool: BaseTool,  # noqa: ARG002  # required by PermissionAsker Protocol; noop impl ignores it
        args: dict[str, Any],  # noqa: ARG002  # required by PermissionAsker Protocol; noop impl ignores it
        rule_hint: Rule,  # noqa: ARG002  # required by PermissionAsker Protocol; noop impl ignores it
    ) -> AskerResponse:
        return AskerResponse(
            choice="deny",
            scope="session",
            rule=None,
            feedback=SUBAGENT_AUTO_DENY_FEEDBACK,
        )


_SUBAGENT_AUTO_DENY_ASKER = _SubagentPermissionAsker()


def _default_storage() -> SessionStorage:
    return SessionStorage(Path(":memory:"))


@dataclass(frozen=True)
class SpawnContext(Generic[A]):
    """Inherited parent state for a spawn; ``build_child`` injected to avoid a session import."""

    build_child: Callable[..., A]
    parent_config: AuraConfig
    parent_model_spec: str
    parent_skills: SkillRegistry | None = None
    parent_carryover_provider: Callable[[], ReadCarryover] | None = None
    parent_ruleset: RuleSet | None = None
    parent_safety: SafetyPolicy | None = None
    parent_mode_provider: Callable[[], str] | None = None
    # Not inherited — child gets a fresh SessionRuleSet; held only for test inspection.
    parent_session: SessionRuleSet | None = None
    parent_deny_rules: RuleSet | None = None
    parent_ask_rules: RuleSet | None = None
    model_factory: Callable[[], BaseChatModel] | None = None
    storage_factory: Callable[[], SessionStorage] | None = None
    parent_abort_event: asyncio.Event | None = None
    parent_storage: SessionStorage | None = None
    parent_hooks: HookChain | None = None
    parent_model: BaseChatModel | None = None
    parent_session_id: str | None = None
    register_abort: Callable[[str, AbortController], None] | None = None


class SubagentSpawner(Generic[A]):
    """Create a standalone AgentSession for a single subagent run."""

    def __init__(self, ctx: SpawnContext[A]) -> None:
        self._ctx = ctx

    @property
    def abort_event(self) -> asyncio.Event | None:
        return self._ctx.parent_abort_event

    @property
    def parent_config(self) -> AuraConfig:
        return self._ctx.parent_config

    @property
    def parent_model(self) -> BaseChatModel | None:
        return self._ctx.parent_model

    @property
    def parent_hooks(self) -> HookChain | None:
        return self._ctx.parent_hooks

    @property
    def parent_storage(self) -> SessionStorage | None:
        return self._ctx.parent_storage

    @property
    def parent_session_id(self) -> str | None:
        return self._ctx.parent_session_id

    @property
    def parent_model_spec(self) -> str:
        return self._ctx.parent_model_spec

    def validate_model_spec(self, spec: str) -> None:
        """Raise :class:`UnknownModelSpecError` if ``spec`` cannot resolve. Pure validation."""
        llm.resolve(spec, cfg=self._ctx.parent_config)

    def spawn(
        self,
        prompt: str,  # noqa: ARG002  # positional API kept for caller compatibility; child reads it via TaskRecord
        allowed_tools: list[str] | None = None,
        *,
        agent_type: str = "general-purpose",
        task_id: str | None = None,
        model_spec: str | None = None,
    ) -> A:
        type_def = get_agent_def(agent_type)

        # Restricted agent_type MUST raise on missing tools, not silently drop them.
        parent_enabled = list(self._ctx.parent_config.tools.enabled)
        if type_def.tools:
            missing = type_def.tools - set(parent_enabled)
            if missing:
                raise ValueError(
                    f"agent_type {agent_type!r} requires tools "
                    f"{sorted(missing)} but parent has not enabled them; "
                    "enable them on the parent config or pick a different "
                    "agent_type."
                )
            effective_allow: set[str] | None = set(type_def.tools)
        else:
            effective_allow = None  # inherit-all sentinel

        # One-level recursion: a subagent never spawns another subagent.
        child_tools = ToolsConfig(
            enabled=[
                name
                for name in parent_enabled
                if name not in AGENT_DISALLOWED_TOOLS
                and (allowed_tools is None or name in allowed_tools)
                and (effective_allow is None or name in effective_allow)
            ]
        )
        child_cfg = self._ctx.parent_config.model_copy(update={"tools": child_tools})
        if self._ctx.model_factory is not None:
            model = self._ctx.model_factory()
        elif model_spec is not None:
            model = llm.make_model_for_spec(model_spec, self._ctx.parent_config)
        else:
            provider, model_name = llm.resolve(
                self._ctx.parent_model_spec,
                cfg=self._ctx.parent_config,
            )
            model = llm.create(provider, model_name)
        storage = (self._ctx.storage_factory or _default_storage)()
        carryover: ReadCarryover | None
        if self._ctx.parent_carryover_provider is not None:
            carryover = self._ctx.parent_carryover_provider()
        else:
            carryover = None

        child_session = SessionRuleSet()
        child_hooks: HookChain | None = None
        child_mode: str = "default"
        if (
            self._ctx.parent_ruleset is not None
            and self._ctx.parent_safety is not None
            and self._ctx.parent_mode_provider is not None
        ):
            parent_mode = self._ctx.parent_mode_provider()
            child_mode = "bypass" if parent_mode == "bypass" else "default"
            # Freeze the mode: a mid-turn parent flip must NOT bleed into the child.
            _resolved_mode: Mode = "bypass" if parent_mode == "bypass" else "default"
            perm_hook = make_permission_hook(
                asker=_SUBAGENT_AUTO_DENY_ASKER,
                session=child_session,
                rules=self._ctx.parent_ruleset,
                deny_rules=self._ctx.parent_deny_rules or RuleSet(),
                ask_rules=self._ctx.parent_ask_rules or RuleSet(),
                project_root=self._ctx.parent_config.resolved_storage_path().parent,
                mode=_resolved_mode,
                safety=self._ctx.parent_safety or DEFAULT_SAFETY,
            )
            child_hooks = HookChain(pre_tool=[perm_hook])

        # Unique session_id per child: save is DELETE-then-INSERT, so a shared key wipes peers.
        child_session_id = (
            f"subagent-{task_id}" if task_id is not None else f"subagent-{uuid4().hex[:8]}"
        )
        # Child controller registered with the parent so one Ctrl+C cascades.
        child_abort = AbortController()
        child_agent = self._ctx.build_child(
            config=child_cfg,
            model=model,
            storage=storage,
            hooks=child_hooks,
            session_id=child_session_id,
            session_rules=child_session,
            pre_loaded_skills=self._ctx.parent_skills,
            system_prompt_suffix=type_def.system_prompt_suffix,
            carryover=carryover,
            mode=child_mode,
            parent_abort=child_abort,
        )
        if self._ctx.register_abort is not None:
            self._ctx.register_abort(
                task_id if task_id is not None else child_session_id,
                child_abort,
            )
        return child_agent
