"""Spawn an isolated child agent per subagent task.

Invariants:
- One-level recursion: every child has ``AGENT_DISALLOWED_TOOLS`` stripped, so a
  subagent can never spawn another subagent.
- Fresh chat model per spawn (the chat model classes are stateful — sharing risks
  cross-talk).
- MCP servers inherited at config level; each child runs its own ``aconnect`` so
  parent and child hold INDEPENDENT connections to the same servers.
- Permission rules inherited; ``SessionRuleSet`` is private to the child so
  approvals do NOT leak back to the parent. Plan / accept_edits collapse to
  ``default`` (no interactive UI); ``bypass`` inherits verbatim.
- Storage defaults to in-memory sqlite so child transcripts don't pollute the
  parent's session DB.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
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


class SubagentSpawner(Generic[A]):
    """Create a standalone AgentSession for a single subagent run.

    ``build_child`` is injected so this module never imports the concrete
    session type — the construction site binds ``A`` to its agent class.
    """

    # Class-level defaults so subclasses that skip __init__ still see sane state.
    _parent_abort_event: asyncio.Event | None = None
    _register_abort: Callable[[str, AbortController], None] | None = None

    def __init__(
        self,
        parent_config: AuraConfig,
        parent_model_spec: str,
        *,
        build_child: Callable[..., A],
        parent_skills: SkillRegistry | None = None,
        parent_carryover_provider: (
            Callable[[], ReadCarryover] | None
        ) = None,
        parent_ruleset: RuleSet | None = None,
        parent_safety: SafetyPolicy | None = None,
        parent_mode_provider: Callable[[], str] | None = None,
        parent_session: SessionRuleSet | None = None,
        parent_deny_rules: RuleSet | None = None,
        parent_ask_rules: RuleSet | None = None,
        model_factory: Callable[[], BaseChatModel] | None = None,
        storage_factory: Callable[[], SessionStorage] | None = None,
        parent_abort_event: asyncio.Event | None = None,
        parent_storage: SessionStorage | None = None,
        parent_hooks: HookChain | None = None,
        parent_model: BaseChatModel | None = None,
        parent_session_id: str | None = None,
        register_abort: Callable[[str, AbortController], None] | None = None,
    ) -> None:
        self._parent_config = parent_config
        self._parent_storage = parent_storage
        self._parent_hooks = parent_hooks
        self._parent_model = parent_model
        self._parent_session_id = parent_session_id
        self._parent_model_spec = parent_model_spec
        self._build_child = build_child
        self._parent_skills = parent_skills
        self._parent_carryover_provider = parent_carryover_provider
        self._parent_ruleset = parent_ruleset
        self._parent_safety = parent_safety
        self._parent_mode_provider = parent_mode_provider
        # ``parent_session`` is NOT inherited; kept for "child.session is fresh" verification only.
        self._parent_session = parent_session
        self._parent_deny_rules = parent_deny_rules
        self._parent_ask_rules = parent_ask_rules
        self._model_factory = model_factory
        self._storage_factory = storage_factory or _default_storage
        self._parent_abort_event = parent_abort_event
        self._register_abort = register_abort

    @property
    def abort_event(self) -> asyncio.Event | None:
        # Class-level default covers subclasses that skip __init__.
        return self._parent_abort_event

    @property
    def parent_config(self) -> AuraConfig:
        return self._parent_config

    @property
    def parent_model(self) -> BaseChatModel | None:
        return self._parent_model

    @property
    def parent_hooks(self) -> HookChain | None:
        return self._parent_hooks

    @property
    def parent_storage(self) -> SessionStorage | None:
        return self._parent_storage

    @property
    def parent_session_id(self) -> str | None:
        return self._parent_session_id

    @property
    def parent_model_spec(self) -> str:
        return self._parent_model_spec

    def validate_model_spec(self, spec: str) -> None:
        """Raise :class:`UnknownModelSpecError` if ``spec`` cannot resolve. Pure validation."""
        llm.resolve(spec, cfg=self._parent_config)

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

        # Restricted agent_type MUST raise on missing tools — silently dropping would
        # hand the child a prompt promising tools it can't see.
        parent_enabled = list(self._parent_config.tools.enabled)
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
                name for name in parent_enabled
                if name not in AGENT_DISALLOWED_TOOLS
                and (allowed_tools is None or name in allowed_tools)
                and (effective_allow is None or name in effective_allow)
            ]
        )
        child_cfg = self._parent_config.model_copy(
            update={"tools": child_tools}
        )
        if self._model_factory is not None:
            model = self._model_factory()
        elif model_spec is not None:
            model = llm.make_model_for_spec(model_spec, self._parent_config)
        else:
            provider, model_name = llm.resolve(
                self._parent_model_spec, cfg=self._parent_config,
            )
            model = llm.create(provider, model_name)
        storage = self._storage_factory()
        carryover: ReadCarryover | None
        if self._parent_carryover_provider is not None:
            carryover = self._parent_carryover_provider()
        else:
            carryover = None

        child_session = SessionRuleSet()
        child_hooks: HookChain | None = None
        child_mode: str = "default"
        if (
            self._parent_ruleset is not None
            and self._parent_safety is not None
            and self._parent_mode_provider is not None
        ):
            parent_mode = self._parent_mode_provider()
            child_mode = "bypass" if parent_mode == "bypass" else "default"
            # Freeze the mode: a mid-turn parent flip must NOT bleed into the child.
            _resolved_mode: Mode = "bypass" if parent_mode == "bypass" else "default"
            perm_hook = make_permission_hook(
                asker=_SUBAGENT_AUTO_DENY_ASKER,
                session=child_session,
                rules=self._parent_ruleset,
                deny_rules=self._parent_deny_rules or RuleSet(),
                ask_rules=self._parent_ask_rules or RuleSet(),
                project_root=self._parent_config.resolved_storage_path().parent,
                mode=_resolved_mode,
                safety=self._parent_safety or DEFAULT_SAFETY,
            )
            child_hooks = HookChain(pre_tool=[perm_hook])

        # Every subagent MUST hold a unique session_id: SessionStorage.save is
        # DELETE-then-INSERT, so concurrent children sharing a key wipe each other.
        child_session_id = (
            f"subagent-{task_id}"
            if task_id is not None
            else f"subagent-{uuid4().hex[:8]}"
        )
        # The child's inherited controller: registered with the parent so a single
        # Ctrl+C cascades, and injected so the child re-raises on abort.
        child_abort = AbortController()
        child_agent = self._build_child(
            config=child_cfg,
            model=model,
            storage=storage,
            hooks=child_hooks,
            session_id=child_session_id,
            session_rules=child_session,
            pre_loaded_skills=self._parent_skills,
            system_prompt_suffix=type_def.system_prompt_suffix,
            carryover=carryover,
            mode=child_mode,
            parent_abort=child_abort,
        )
        if self._register_abort is not None:
            self._register_abort(
                task_id if task_id is not None else child_session_id,
                child_abort,
            )
        return child_agent
