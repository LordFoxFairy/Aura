"""AgentSession — stateful conversation controller (config + model + storage + hooks)."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from aura.application.commands.types import Command
from aura.application.compact import (
    MICROCOMPACT_KEEP_RECENT,
    MICROCOMPACT_TRIGGER_PAIRS,
    CompactResult,
    MicrocompactPolicy,
    run_compact,
)
from aura.application.compact.compactor import Compactor
from aura.application.compact.constants import (
    AUTO_COMPACT_THRESHOLD,
    auto_compact_threshold_for,
)
from aura.application.hooks import HookChain
from aura.application.hooks.auto_reload import (
    make_aura_md_reload_hook,
    make_cwd_rules_reload_hook,
)
from aura.application.hooks.bash_safety import make_bash_safety_hook
from aura.application.hooks.budget import default_hooks
from aura.application.hooks.must_read_first import make_must_read_first_hook
from aura.application.loop import DEFAULT_SESSION as _DEFAULT_SESSION
from aura.application.loop import AgentLoop
from aura.application.loop_state import LoopState
from aura.application.memory import project_memory, rules
from aura.application.memory.context import Context
from aura.application.memory.system_prompt import build_system_prompt
from aura.application.runtime.mcp import McpRuntime
from aura.application.runtime.session import SessionRuntime
from aura.application.runtime.tool_factory import STATEFUL_TOOL_FACTORIES
from aura.application.runtime.tool_runtime import ToolRuntime
from aura.application.tasks.spawn import SubagentSpawner
from aura.application.tasks.store import TasksStore
from aura.application.teams.team_port import TeammateBinding, TeamPort
from aura.config.schema import AuraConfig, AuraConfigError, ToolsConfig
from aura.domain.abort import AbortController, AbortException
from aura.domain.agent_definition import AgentDefinition
from aura.domain.events import AgentEvent, AssistantDelta, Final
from aura.domain.permission.denials import PermissionDenial
from aura.domain.permission.mode import Mode
from aura.domain.permission.safety import SafetyPolicy
from aura.domain.permission.session import RuleSet, SessionRuleSet
from aura.domain.state_values import BuddyState, ReadCarryover, ReadRecord
from aura.domain.task import TaskNotification, TaskRecord
from aura.domain.tokens import estimate_message_tokens, estimate_text_tokens
from aura.domain.tool import ToolError
from aura.domain.tool_registry import ToolRegistry
from aura.infrastructure import llm
from aura.infrastructure.mcp import MCPManager
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.skills import Skill, SkillRegistry, load_skills
from aura.infrastructure.wire.events import WireEvent
from aura.infrastructure.wire.serialize import (
    task_notification_to_wire,
    task_progress_to_wire,
    task_started_to_wire,
)
from aura.tools import BUILTIN_STATEFUL_TOOLS, BUILTIN_TOOLS
from aura.tools.ask_user import FormQuestionDict, UserAsker
from aura.tools.send_message import SendMessage
from aura.tools.web_fetch import set_default_model_factory


async def _unavailable_question_asker(
    _questions: list[FormQuestionDict],
) -> dict[str, str]:
    # Surfaced as ToolError so a missing REPL doesn't crash the loop.
    raise ToolError(
        "ask_user_question is unavailable: no CLI asker was injected. "
        "Run aura through the CLI, or pass question_asker=... to "
        "build_agent(...) / AgentSession(...) when driving programmatically."
    )


def _validated_mode(mode: str, *, disable_bypass: bool, source: str) -> Mode:
    valid: tuple[Mode, ...] = ("default", "accept_edits", "plan", "bypass")
    if mode not in valid:
        raise ValueError(
            f"invalid mode {mode!r}; expected one of {sorted(valid)}"
        )
    resolved: Mode = mode
    if resolved == "bypass" and disable_bypass:
        raise AuraConfigError(
            source="PermissionsConfig",
            detail=(
                "bypass mode is disabled by config "
                "(permissions.disable_bypass=true); "
                f"refusing {source}('bypass')"
            ),
        )
    return resolved


class AgentSession:
    def __init__(
        self,
        config: AuraConfig,
        *,
        model: BaseChatModel,
        storage: SessionStorage,
        hooks: HookChain | None = None,
        available_tools: dict[str, BaseTool] | None = None,
        session_id: str = _DEFAULT_SESSION,
        session_rules: SessionRuleSet | None = None,
        question_asker: UserAsker | None = None,
        auto_compact_threshold: int = AUTO_COMPACT_THRESHOLD,
        auto_microcompact_enabled: bool = True,
        microcompact_trigger_pairs: int = MICROCOMPACT_TRIGGER_PAIRS,
        microcompact_keep_recent: int = MICROCOMPACT_KEEP_RECENT,
        session_log_dir: Path | None = None,
        pre_loaded_skills: SkillRegistry | None = None,
        mode: str = "default",
        system_prompt_suffix: str = "",
        disable_bypass: bool = False,
        carryover: ReadCarryover | None = None,
        ruleset: RuleSet | None = None,
        deny_ruleset: RuleSet | None = None,
        ask_ruleset: RuleSet | None = None,
        safety: SafetyPolicy | None = None,
        parent_abort: AbortController | None = None,
    ) -> None:
        self._config = config
        self._model = model
        # Set when this session IS a subagent: the controller it inherits from the
        # parent. Doubles as the "re-raise on abort so run_task sees terminal" signal.
        self._parent_abort = parent_abort
        # switch_model mutates the live spec; config.router stays immutable.
        self._current_model_spec = config.router.get("default", "")
        self._session_runtime = SessionRuntime(
            storage=storage,
            session_id=session_id,
            session_log_dir=session_log_dir,
            session_rules=session_rules,
            carryover=carryover,
        )
        self._hooks = hooks or HookChain()
        self._state = LoopState()
        self._state.slots = dataclasses.replace(
            self._state.slots, consecutive_compact_failures=0,
        )
        self._disable_bypass = disable_bypass
        self._mode = _validated_mode(
            mode,
            disable_bypass=disable_bypass,
            source="to construct AgentSession(mode=...)",
        )
        # enter_plan_mode stashes the prior mode for exit_plan_mode to restore.
        self._prior_mode: str | None = None
        self._auto_compact_threshold = auto_compact_threshold
        if (
            auto_microcompact_enabled
            and microcompact_trigger_pairs > 0
            and microcompact_keep_recent >= microcompact_trigger_pairs
        ):
            raise AuraConfigError(
                source="AgentConfig",
                detail=(
                    "microcompact_keep_recent "
                    f"({microcompact_keep_recent}) must be strictly less than "
                    f"microcompact_trigger_pairs ({microcompact_trigger_pairs}); "
                    "otherwise the trigger can never fire any clears."
                ),
            )
        self._auto_microcompact_enabled = auto_microcompact_enabled
        self._microcompact_trigger_pairs = microcompact_trigger_pairs
        self._microcompact_keep_recent = microcompact_keep_recent
        self._cwd = Path.cwd()
        self._current_abort: AbortController | None = None

        self._init_subagents(
            pre_loaded_skills=pre_loaded_skills,
            ruleset=ruleset,
            deny_ruleset=deny_ruleset,
            ask_ruleset=ask_ruleset,
            safety=safety,
        )
        self._init_tools(question_asker=question_asker, available_tools=available_tools)

        # Stashed so clear_session can rebuild the prompt identically.
        self._system_prompt_suffix = system_prompt_suffix
        self._auto_memory_dir = self._storage.memory_dir(cwd=self._cwd)
        self._system_prompt = (
            build_system_prompt(
                cwd=self._cwd,
                model_spec=self._current_model_spec,
                auto_memory_dir=self._auto_memory_dir,
            )
            + system_prompt_suffix
        )
        self._primary_memory = project_memory.load_project_memory(
            self._cwd,
            auto_memory_dir=self._auto_memory_dir,
        )
        self._rules = rules.load_rules(self._cwd)
        # carryover only flows into the FIRST Context; /clear and /compact
        # build fresh Contexts so a parent's read fingerprints don't resurrect.
        self._context = self._build_context(carryover=carryover)

        self._init_hooks()

        self._pending_compact_events: list[dict[str, object]] = []
        self._loop = self._build_loop()
        self._mcp_runtime = McpRuntime(
            list(self._config.mcp_servers),
            mcp_overrides_builtin=self._config.tools.mcp_overrides_builtin,
        )
        # Anchor for providers that always return 0 cache_read_input_tokens.
        self._pinned_tokens_estimate = self._estimate_pinned_tokens()
        self._user_pinned_tools_allowlist_value = (
            list(self._config.tools.enabled)
            != list(ToolsConfig().enabled)
        )
        set_default_model_factory(
            llm.make_summary_model_factory(self._config, self._model)
        )
    def _init_subagents(
        self,
        *,
        pre_loaded_skills: SkillRegistry | None,
        ruleset: RuleSet | None,
        deny_ruleset: RuleSet | None,
        ask_ruleset: RuleSet | None,
        safety: SafetyPolicy | None,
    ) -> None:
        # Subagent path injects skills directly so children inherit the parent set.
        if pre_loaded_skills is not None:
            self._skill_registry = pre_loaded_skills
        else:
            self._skill_registry = load_skills(
                cwd=self._cwd, include_bundled=True,
            )
        self._tasks_store = TasksStore()
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._running_shells: dict[str, asyncio.subprocess.Process] = {}
        # Each child registers its AbortController here so a single Ctrl+C cascades.
        self._running_aborts: dict[str, AbortController] = {}
        # parent_mode_provider closes over self so mid-session mode changes
        # are visible to every spawn.
        self.subagent_factory = SubagentSpawner(
            build_child=AgentSession,
            parent_config=self._config,
            parent_model_spec=self._config.router.get("default", ""),
            parent_skills=self._skill_registry,
            parent_carryover_provider=self._snapshot_read_carryover,
            parent_ruleset=ruleset,
            parent_safety=safety,
            parent_mode_provider=lambda: self._mode,
            parent_session=self._session_rules,
            parent_deny_rules=deny_ruleset,
            parent_ask_rules=ask_ruleset,
            parent_storage=self._storage,
            parent_hooks=self._hooks,
            parent_model=self._model,
            parent_session_id=self._session_id,
            register_abort=self._running_aborts.__setitem__,
        )
        self._team: TeamPort | None = None
        # Concrete manager the /team command stack caches across invocations;
        # distinct from _team, which holds the narrow TeamPort contract.
        self._team_manager: TeamPort | None = None
        self._team_member_name: str | None = None
        self._teammate: TeammateBinding | None = None

        def _on_terminal(rec: TaskRecord) -> None:
            summary = rec.final_result or rec.error
            notification = TaskNotification(
                task_id=rec.id,
                status=rec.status,
                summary=summary,
                description=rec.description,
            )
            self._enqueue_task_notification(notification)
            self._enqueue_protocol_event(
                task_notification_to_wire(
                    notification,
                    parent_id=self.session_id,
                ),
            )
        self._tasks_store.add_terminal_listener(_on_terminal)

        def _on_started(rec: TaskRecord) -> None:
            self._enqueue_protocol_event(
                task_started_to_wire(
                    task_id=rec.id,
                    description=rec.description,
                    parent_session_id=self.session_id,
                    started_at=rec.started_at,
                    parent_id=self.session_id,
                ),
            )
        self._tasks_store.add_started_listener(_on_started)

        def _on_activity(rec: TaskRecord, activity: str) -> None:
            self._enqueue_protocol_event(
                task_progress_to_wire(
                    task_id=rec.id,
                    tool_name=activity,
                    activity_count=rec.progress.tool_count,
                    parent_id=self.session_id,
                ),
            )
        self._tasks_store.add_activity_listener(_on_activity)

    def _init_tools(
        self,
        *,
        question_asker: UserAsker | None,
        available_tools: dict[str, BaseTool] | None,
    ) -> None:
        # Stateful tools instantiate per-AgentSession; stateless ones reuse singletons.
        self._available_tools = (
            dict(available_tools) if available_tools is not None else dict(BUILTIN_TOOLS)
        )
        tool_runtime = ToolRuntime(
            state=self._state,
            asker=question_asker or _unavailable_question_asker,
            tasks_store=self._tasks_store,
            spawner=self.subagent_factory,
            running_tasks=self._running_tasks,
            running_shells=self._running_shells,
            transcript_storage=self._storage,
            team_provider=lambda: self.team,
            member_name_provider=lambda: self._team_member_name,
        )
        for factory in STATEFUL_TOOL_FACTORIES:
            self._available_tools[factory.name] = factory.build(tool_runtime)
        # Closures below need AgentSession-bound methods so they can't use ToolRuntime.
        self._available_tools["task_output"] = BUILTIN_STATEFUL_TOOLS[
            "task_output"
        ](store=self._tasks_store)
        self._available_tools["bash_background"] = BUILTIN_STATEFUL_TOOLS[
            "bash_background"
        ](
            store=self._tasks_store,
            running_shells=self._running_shells,
            running_tasks=self._running_tasks,
        )
        self._available_tools["web_search"] = BUILTIN_STATEFUL_TOOLS[
            "web_search"
        ](config=self._config.web_search)
        self._available_tools["enter_plan_mode"] = BUILTIN_STATEFUL_TOOLS[
            "enter_plan_mode"
        ](
            mode_setter=self.set_mode,
            mode_getter=lambda: self._mode,
            save_prior_mode=self._capture_prior_mode,
        )
        self._available_tools["exit_plan_mode"] = BUILTIN_STATEFUL_TOOLS[
            "exit_plan_mode"
        ](
            mode_setter=self.set_mode,
            mode_getter=lambda: self._mode,
            asker=question_asker or _unavailable_question_asker,
            get_prior_mode=lambda: self._prior_mode,
        )
        self._available_tools["skill"] = BUILTIN_STATEFUL_TOOLS["skill"](
            recorder=self.record_skill_invocation,
            registry=self._skill_registry,
            session_id_provider=lambda: self._session_id,
            session_rules_provider=lambda: self._session_rules,
            loop_state_provider=lambda: self._state,
        )
        tools: list[BaseTool] = []
        for name in self._config.tools.enabled:
            tool = self._available_tools.get(name)
            if tool is None:
                raise AuraConfigError(
                    source="tools.enabled",
                    detail=f"unknown tool name: {name!r}",
                )
            tools.append(tool)
        self._registry = ToolRegistry(tools)

    def _init_hooks(self) -> None:
        # bash_safety anchors at pre_tool[0] — safety is orthogonal to
        # allow/deny/ask, and tracking it as a field lets clear_session
        # re-insert it idempotently.
        self._bash_safety_hook = make_bash_safety_hook(
            mode_provider=lambda: self._mode,
        )
        self._hooks.pre_tool.insert(0, self._bash_safety_hook)
        # Appended last so a denied tool doesn't also raise missing-read.
        self._must_read_first_hook = make_must_read_first_hook(self._context)
        self._hooks.pre_tool.append(self._must_read_first_hook)
        self._hooks.file_changed.append(make_aura_md_reload_hook(self))
        self._hooks.cwd_changed.append(make_cwd_rules_reload_hook(self))

    async def astream(
        self,
        prompt: str,
        *,
        attachments: list[HumanMessage] | None = None,
        abort: AbortController | None = None,
    ) -> AsyncIterator[AgentEvent | dict[str, object]]:
        # Persist the user turn before any ainvoke so a mid-stream crash
        # leaves resumable history instead of a black hole.
        ctx = (
            journal.session_scope(self._session_log_path)
            if self._session_log_path is not None
            else contextlib.nullcontext()
        )
        with ctx:
            journal.write(
                "astream_begin",
                session=self._session_id,
                prompt_preview=prompt[:200],
            )
            history = self._storage.load(self._session_id)
            # Snapshot for rollback: cancel before any AIMessage pops the user turn.
            history_len_before_user_turn = len(history)
            if attachments:
                history.extend(attachments)
            history.append(HumanMessage(content=prompt))
            self._storage.save(self._session_id, history)

            # max_turns is per-user-turn, not per-AgentSession-lifetime.
            self._state.turn_count = 0

            # Abort precedence: explicit kwarg > inherited from parent > own.
            inherited = self._parent_abort
            local_abort = (
                abort
                if abort is not None
                else (inherited if inherited is not None else AbortController())
            )
            self._current_abort = local_abort
            self._partial_assistant_text = ""
            self._pending_compact_events.clear()

            saw_ai_message = False
            try:
                try:
                    async for event in self._loop.run_turn(
                        history=history, abort=local_abort,
                    ):
                        if isinstance(event, AssistantDelta):
                            self._partial_assistant_text += event.text
                        while self._pending_compact_events:
                            yield self._pending_compact_events.pop(0)
                        yield event
                    while self._pending_compact_events:
                        yield self._pending_compact_events.pop(0)
                    saw_ai_message = any(
                        isinstance(m, AIMessage)
                        for m in history[history_len_before_user_turn:]
                    )
                except (AbortException, asyncio.CancelledError) as exc:
                    saw_ai_message = any(
                        isinstance(m, AIMessage)
                        for m in history[history_len_before_user_turn:]
                    )
                    is_abort = (
                        isinstance(exc, AbortException) or local_abort.aborted
                    )
                    journal.write(
                        "astream_cancelled",
                        session=self._session_id,
                        is_abort=is_abort,
                        had_ai=saw_ai_message,
                    )
                    if self._partial_assistant_text:
                        yield AssistantDelta(text=self._partial_assistant_text)
                        self._partial_assistant_text = ""
                    if is_abort:
                        # No AIMessage landed → drop the unanswered user turn.
                        if not saw_ai_message:
                            del history[history_len_before_user_turn:]
                        self._storage.save(self._session_id, history)
                        await self._cascade_abort_to_children(
                            local_abort.reason or "parent_aborted",
                        )
                        yield Final(message="(cancelled)", reason="aborted")
                        # Subagents re-raise so run_task sees a terminal status.
                        if isinstance(exc, asyncio.CancelledError):
                            raise
                        if self._parent_abort is not None:
                            raise
                        return
                    yield Final(message="(cancelled)")
                    raise
            finally:
                self._current_abort = None
            self._storage.save(self._session_id, history)
            journal.write(
                "astream_end",
                session=self._session_id,
                history_len=len(history),
                total_tokens=self._state.total_tokens_used,
            )
            # Auto-compact runs after the stream ends to avoid interleaving
            # compact I/O with the caller's yield loop.
            await self._compactor.auto(
                history,
                self._state.slots,
                model=self._current_model_spec,
            )
            while self._pending_compact_events:
                yield self._pending_compact_events.pop(0)

    def switch_model(self, spec: str) -> None:
        """Swap the live model and rebuild the loop; config.router stays put."""
        old_spec = self._current_model_spec
        journal.write("model_switch_attempt", old_spec=old_spec, new_spec=spec)
        provider, model_name = llm.resolve(spec, cfg=self._config)
        self._model = llm.create(provider, model_name)
        self._current_model_spec = spec
        self._loop = self._build_loop()
        journal.write(
            "model_switched",
            old_spec=old_spec,
            new_spec=spec,
            provider=provider.name,
            model=model_name,
        )

    def _capture_prior_mode(self, mode: str) -> None:
        """Stash the pre-plan mode so exit_plan_mode restores it correctly."""
        self._prior_mode = mode

    # SessionRuntime owns lifecycle state; these forwards preserve the
    # historical attribute names for the broad tail of internal callers.

    @property
    def _storage(self) -> SessionStorage:
        return self._session_runtime.storage

    @property
    def _session_id(self) -> str:
        return self._session_runtime.session_id

    @_session_id.setter
    def _session_id(self, value: str) -> None:
        # resume_session is the legitimate writer; setter preserved for tests.
        self._session_runtime.session_id = value

    @property
    def _session_log_path(self) -> Path | None:
        return self._session_runtime.session_log_path

    @property
    def _session_rules(self) -> SessionRuleSet | None:
        return self._session_runtime.session_rules

    @property
    def session_rules(self) -> SessionRuleSet | None:
        """Public alias for :attr:`_session_rules` — callers outside AgentSession use this."""
        return self._session_runtime.session_rules

    @property
    def _partial_assistant_text(self) -> str:
        return self._session_runtime.partial_assistant_text

    @_partial_assistant_text.setter
    def _partial_assistant_text(self, value: str) -> None:
        self._session_runtime.partial_assistant_text = value

    @property
    def _pending_notifications(self) -> list[TaskNotification]:
        # Live list — callers .append / .clear directly on the runtime's queue.
        return self._session_runtime.pending_notifications_live

    def clear_session(self) -> None:
        # SessionRuntime owns lifecycle (storage.clear, session_rules drop,
        # buffers, SessionStart re-arm). AgentSession owns LoopState slots, hook
        # rewiring, memory/rules cache invalidation, Context/Loop rebuild.
        self._session_runtime.clear()
        self._state.reset()
        self._state.slots.turn_denials.clear()
        self._state.slots.todos.clear()
        self._state.slots.perm_dedup_cache.clear()
        self._state.slots.invoked_skills.clear()
        self._state.slots.preserved_invoked_skills.clear()
        self._state.slots.skill_restrict_leases.clear()
        self._state.slots = dataclasses.replace(
            self._state.slots,
            active_team=None,
            consecutive_compact_failures=0,
            buddy=BuddyState(),
        )
        self._prior_mode = None
        project_memory.clear_cache(self._cwd)
        rules.clear_cache(self._cwd)
        self._primary_memory = project_memory.load_project_memory(
            self._cwd,
            auto_memory_dir=self._storage.memory_dir(cwd=self._cwd),
        )
        self._rules = rules.load_rules(self._cwd)
        self._context = self._build_context()
        self._swap_must_read_first_hook()
        # Re-anchor bash safety so "safety is first" survives future mutations.
        self._hooks.pre_tool.remove(self._bash_safety_hook)
        self._bash_safety_hook = make_bash_safety_hook(
            mode_provider=lambda: self._mode,
        )
        self._hooks.pre_tool.insert(0, self._bash_safety_hook)
        self._loop = self._build_loop()
        journal.write("session_cleared", session=self._session_id)

    def effective_auto_compact_threshold(self) -> int:
        """Resolve the live auto-compact threshold.

        ``-1`` derives from the live model's context window; ``0`` disables;
        positive values override. Recomputed so switch_model is honoured.
        """
        if self._auto_compact_threshold == -1:
            return auto_compact_threshold_for(self._current_model_spec)
        return self._auto_compact_threshold

    async def compact(
        self, *, source: Literal["manual", "auto", "reactive"] = "manual",
    ) -> CompactResult:
        """Summarize old history, preserve session state, rebuild Context."""
        return await run_compact(self, source=source)

    def apply_aura_md_reload(self) -> None:
        """Re-read AURA.md + rules, rebuild Context, refresh hook + loop."""
        project_memory.clear_cache(self._cwd)
        rules.clear_cache(self._cwd)
        self._primary_memory = project_memory.load_project_memory(self._cwd)
        self._rules = rules.load_rules(self._cwd)
        self._context = self._build_context()
        self._swap_must_read_first_hook()
        self._loop = self._build_loop()

    def change_cwd_and_reload(self, new_cwd: Path) -> None:
        """Move to ``new_cwd`` and refresh memory + rules + context."""
        self._cwd = new_cwd
        self.apply_aura_md_reload()

    def apply_compaction(
        self,
        *,
        new_history: list[BaseMessage],
        new_context: Context,
        preserved_skills: list[Skill],
    ) -> None:
        """Persist post-compact history and swap to ``new_context``."""
        # Slot mutation precedes save so a crash leaves state.slots consistent.
        self._state.slots.preserved_invoked_skills[:] = list(preserved_skills)
        self._storage.save(self._session_id, new_history)
        self._context = new_context
        self._swap_must_read_first_hook()
        self._loop = self._build_loop()

    def reload_memory_and_rules(self) -> None:
        """Re-read AURA.md + rules from disk into AgentSession state.

        Compact reloads BEFORE rebuilding Context so the new Context picks
        up disk edits made during the prior turn.
        """
        self._primary_memory = project_memory.load_project_memory(self._cwd)
        self._rules = rules.load_rules(self._cwd)

    def _swap_must_read_first_hook(self) -> None:
        # Idempotent: test fixtures that swap _hooks may not have the hook yet.
        if self._must_read_first_hook in self._hooks.pre_tool:
            self._hooks.pre_tool.remove(self._must_read_first_hook)
        self._must_read_first_hook = make_must_read_first_hook(self._context)
        self._hooks.pre_tool.append(self._must_read_first_hook)

    def record_skill_invocation(self, skill: Skill) -> None:
        """Append ``skill`` to Context's invoked list."""
        self._context.record_skill_invocation(skill)

    @property
    def current_abort(self) -> AbortController | None:
        """Live AbortController for the running astream call, or None."""
        return self._current_abort

    @property
    def cwd(self) -> Path:
        """Logical working directory; mutate via :meth:`set_cwd`."""
        return self._cwd

    @property
    def team(self) -> TeamPort | None:
        """Bound TeamManager, seen through its narrow structural contract."""
        return self._team

    @property
    def pending_notifications(self) -> tuple[TaskNotification, ...]:
        """Read-only snapshot of queued TaskNotification records."""
        return self._session_runtime.pending_notifications

    @property
    def pending_protocol_events(self) -> tuple[WireEvent, ...]:
        """Snapshot of queued coordination wire events for transports."""
        events: list[WireEvent] = list(self._session_runtime.pending_protocol_events)
        team = self._team
        if team is not None:
            events.extend(team.pending_protocol_events)
        return tuple(events)

    def buffer_partial_assistant_text(self, text: str) -> None:
        """Append ``text`` so an abort still surfaces partial reasoning."""
        self._session_runtime.buffer_partial_assistant_text(text)

    def _enqueue_task_notification(self, notif: TaskNotification) -> None:
        self._session_runtime.enqueue_task_notification(notif)

    def _drain_task_notifications(self) -> list[TaskNotification]:
        return self._session_runtime.drain_task_notifications()

    def _enqueue_protocol_event(self, event: WireEvent) -> None:
        self._session_runtime.enqueue_protocol_event(event)

    def drain_protocol_events(self) -> list[WireEvent]:
        """Pop queued coordination wire events, oldest first."""
        drained: list[WireEvent] = self._session_runtime.drain_protocol_events()
        team = self._team
        if team is not None:
            drained.extend(team.drain_protocol_events())
        return drained

    async def _cascade_abort_to_children(self, reason: str) -> None:
        """Fire every controller in :attr:`_running_aborts`.

        Yields a scheduler tick so child watchdogs observe the signal
        before astream's finally clause unwinds.
        """
        for controller in list(self._running_aborts.values()):
            if not controller.aborted:
                controller.abort(reason)
        await asyncio.sleep(0)

    def join_team(
        self,
        *,
        manager: TeamPort,
        member_name: str | None = None,
        task_id: str | None = None,
        tasks_store: TasksStore | None = None,
    ) -> None:
        """Bind to a TeamManager; auto-enable send_message when allowed."""
        if not self._config.teams.enabled:
            raise RuntimeError(
                "teams disabled — set teams.enabled=true in "
                ".aura/config.json to enable the multi-agent swarm subsystem"
            )
        if task_id is not None and tasks_store is not None:
            self._teammate = TeammateBinding(task_id=task_id, tasks_store=tasks_store)
        if self._team is manager and (
            member_name is None or self._team_member_name == member_name
        ):
            return
        self._team = manager
        if member_name is not None:
            self._team_member_name = member_name
        self._auto_enable_send_message_for_team()

    def leave_team(self) -> None:
        """Unbind and drop send_message iff we auto-added it."""
        self._team = None
        self._team_member_name = None
        self._teammate = None
        self._auto_disable_send_message_for_team()

    def _auto_enable_send_message_for_team(self) -> None:
        # Skip when teams is disabled, user pinned the allowlist, or already wired.
        if not self._config.teams.enabled:
            return
        if self._user_pinned_tools_allowlist_value:
            return
        if "send_message" in self._registry:
            return
        send_tool = SendMessage(
            team_provider=lambda: self.team,
            member_name_provider=lambda: self._team_member_name,
        )
        self._registry.register(send_tool)
        self._available_tools["send_message"] = send_tool
        self._loop.rebind_tools(self._registry.tools())

    def _auto_disable_send_message_for_team(self) -> None:
        if self._user_pinned_tools_allowlist_value:
            return
        if "send_message" not in self._registry:
            return
        self._registry.unregister("send_message")
        self._available_tools.pop("send_message", None)
        self._loop.rebind_tools(self._registry.tools())

    def resume_session(self, session_id: str) -> int:
        """Load ``session_id`` from storage; reset state to fresh-session feel.

        Raises ``KeyError`` if the requested session has no rows. Returns
        the message count of the resumed session.
        """
        count = self._session_runtime.resume(session_id)
        self._state.reset()
        self._state.slots.turn_denials.clear()
        return count

    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def storage(self) -> SessionStorage:
        return self._session_runtime.storage

    @property
    def config(self) -> AuraConfig:
        return self._config

    @property
    def definition(self) -> AgentDefinition:
        """Static config snapshot for this turn (model/mode are live, so rebuilt)."""
        return AgentDefinition(
            system_prompt=self._system_prompt,
            model_spec=self._current_model_spec,
            permission_mode=self._mode,
            tool_names=frozenset(self._config.tools.enabled),
        )

    @property
    def hooks(self) -> HookChain:
        return self._hooks

    @property
    def model(self) -> BaseChatModel:
        return self._model

    @property
    def mcp_manager(self) -> MCPManager | None:
        """Live MCP manager, or ``None`` before/outside ``aconnect``."""
        return self._mcp_runtime.manager

    # Back-compat shims: external callers poke _mcp_manager / _mcp_commands;
    # data lives on McpRuntime, so forward reads + writes both ways.
    @property
    def _mcp_manager(self) -> MCPManager | None:
        return self._mcp_runtime.manager

    @_mcp_manager.setter
    def _mcp_manager(self, value: MCPManager | None) -> None:
        self._mcp_runtime.manager = value

    @property
    def _mcp_commands(self) -> list[Command[object]]:
        return self._mcp_runtime.commands

    @_mcp_commands.setter
    def _mcp_commands(self, value: list[Command[object]]) -> None:
        self._mcp_runtime.commands = list(value)

    @property
    def mcp_commands(self) -> list[Command[object]]:
        return self._mcp_runtime.commands

    @property
    def context(self) -> Context:
        return self._context

    @property
    def microcompact_policy(self) -> MicrocompactPolicy | None:
        """Live loop's microcompact policy; compact's summary view reuses it."""
        return self._loop.microcompact_policy

    @property
    def skill_registry(self) -> SkillRegistry:
        return self._skill_registry

    @property
    def tasks_store(self) -> TasksStore:
        return self._tasks_store

    @property
    def running_tasks(self) -> dict[str, asyncio.Task[None]]:
        return self._running_tasks

    @property
    def running_aborts(self) -> dict[str, AbortController]:
        return self._running_aborts

    @property
    def current_model(self) -> str:
        """Live model spec; ``config.router["default"]`` until switch_model fires."""
        return self._current_model_spec

    @property
    def mode(self) -> str:
        """Effective permission mode — default / accept_edits / plan / bypass."""
        return self._mode

    async def set_cwd(self, path: Path) -> None:
        """Resolve ``path``, retarget ``_cwd``, fire CwdChangedHook consumers.

        The process CWD stays put — AgentSession's logical workdir is enough for
        memory + rules + skill loading, and mutating os.cwd would race tools.
        """
        new_cwd = Path(path).expanduser().resolve()
        old_cwd = self._cwd
        if new_cwd == old_cwd:
            return
        self._cwd = new_cwd
        journal.write(
            "cwd_changed",
            session=self._session_id,
            old_cwd=str(old_cwd),
            new_cwd=str(new_cwd),
        )
        await self._hooks.run_cwd_changed(
            old_cwd=old_cwd, new_cwd=new_cwd, state=self._state,
        )

    def set_mode(self, mode: str) -> None:
        """Update the permission mode; rejects ``bypass`` when disabled by config."""
        self._mode = _validated_mode(
            mode,
            disable_bypass=self._disable_bypass,
            source="set_mode",
        )
        journal.write("mode_changed", session=self._session_id, mode=self._mode)

    @property
    def context_window(self) -> int:
        """Effective context window; ``AuraConfig.context_window`` overrides."""
        if self._config.context_window is not None:
            return self._config.context_window
        return llm.get_context_window(self.current_model)

    @property
    def pinned_tokens_estimate(self) -> int:
        """Char-count/4 approximation of the pinned prompt prefix in tokens.

        Anchors the status bar for providers (deepseek etc.) that always
        return 0 cache_read_input_tokens, and gives the REPL a non-zero
        starting figure before the first turn.
        """
        return self._pinned_tokens_estimate

    def estimate_history_tokens(self, history: list[BaseMessage]) -> int:
        """Fallback token count when ``usage_metadata`` is missing.

        Covers the full per-turn prompt (pinned prefix + history) — counting
        history alone underestimates by ~5–10k tokens on a real config.
        """
        history_tokens = sum(estimate_message_tokens(msg) for msg in history)
        return history_tokens + self._estimate_pinned_tokens()

    def _estimate_pinned_tokens(self) -> int:
        tokens = 0
        for message in self._context.build([]):
            content = message.content
            if isinstance(content, str):
                tokens += estimate_text_tokens(content)
        # Tool schemas count toward the cached prefix the provider bills.
        for tool in self._registry.tools():
            tokens += estimate_text_tokens(tool.name or "")
            tokens += estimate_text_tokens(tool.description or "")
            try:
                schema = json.dumps(
                    tool.args,
                    default=str,
                    ensure_ascii=False,
                )
            except (TypeError, ValueError):
                schema = ""
            tokens += estimate_text_tokens(schema)
        return tokens

    @property
    def router_aliases(self) -> dict[str, str]:
        """``provider:model`` mapping for every alias except ``default``."""
        return {k: v for k, v in self._config.router.items() if k != "default"}

    @property
    def session_id(self) -> str:
        return self._session_id

    def last_turn_denials(self) -> tuple[PermissionDenial, ...]:
        """Immutable snapshot of permission denials from the most recent turn.

        Cleared at the start of each run_turn; between turns the tuple
        reflects the just-finished turn so SDK/UI code can inspect denials
        without parsing journal JSONL.
        """
        return tuple(self._state.slots.turn_denials)

    def _build_loop(self) -> AgentLoop:
        policy: MicrocompactPolicy | None
        if (
            self._auto_microcompact_enabled
            and self._microcompact_trigger_pairs > 0
        ):
            policy = MicrocompactPolicy(
                trigger_pairs=self._microcompact_trigger_pairs,
                keep_recent=self._microcompact_keep_recent,
            )
        else:
            policy = None
        # Fresh Compactor per call so switch_model picks up the live policy;
        # event_emitter feeds astream's drain so wire consumers see compact events.
        self._compactor = Compactor(
            agent=self,
            config=self._config.compact,
            summary_model=self._model,
            microcompact_policy=policy,
            session_id=self._session_id,
            turn_provider=lambda: self._state.turn_count,
            event_emitter=self._pending_compact_events.append,
        )
        return AgentLoop(
            model=self._model,
            registry=self._registry,
            context=self._context,
            hooks=self._hooks,
            state=self._state,
            retry_config=self._config.retry,
            session_id=self._session_id,
            microcompact_policy=policy,
            compact_callback=self._reactive_compact_callback,
            compactor=self._compactor,
        )

    async def _reactive_compact_callback(
        self, history: list[BaseMessage],
    ) -> None:
        """Compact + refresh ``history`` in place on context-overflow."""
        await self.compact(source="reactive")
        history[:] = self._storage.load(self._session_id)

    def _build_context(
        self,
        *,
        carryover: ReadCarryover | None = None,
    ) -> Context:
        return Context(
            cwd=self._cwd,
            system_prompt=self._system_prompt,
            primary_memory=self._primary_memory,
            rules=self._rules,
            skills=self._skill_registry.list(),
            todos_provider=lambda: self._state.slots.todos,
            notifications_drainer=self._drain_task_notifications,
            carryover=carryover,
        )

    def _snapshot_read_carryover(self) -> ReadCarryover:
        """Build a ReadCarryover from live Context for SubagentSpawner spawn."""
        turn = self._state.turn_count
        records: dict[Path, ReadRecord] = {}
        for path, rec in self._context.read_records.items():
            records[path] = ReadRecord(
                path=path,
                mtime_at_read=rec.mtime,
                size_at_read=rec.size,
                read_at_turn=turn,
            )
        return ReadCarryover(
            records=records,
            source_session_id=self._session_id,
            generated_at_turn=turn,
        )

    async def aconnect(self) -> None:
        """Establish MCP connections and register discovered tools.

        Failures are journaled and swallowed inside the runtime so MCP
        outages degrade the session gracefully. MCP resources surface via
        the ``@server:uri`` preprocessor in cli.attachments, not as tools.
        """
        merged = await self._mcp_runtime.connect_all(self._registry.tools())
        if merged is None:
            return
        self._mcp_runtime.replace_registry_contents(self._registry, merged)
        self._loop.rebind_tools(self._registry.tools())

    def _teardown_local_tasks(self) -> None:
        """Cancel subagent tasks and SIGKILL lingering shell subprocesses."""
        for task_id, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
            self._running_tasks.pop(task_id, None)
        # Watcher cancel sends SIGTERM→SIGKILL; SIGKILL here is belt-and-braces.
        for task_id, proc in list(self._running_shells.items()):
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, Exception):
                    proc.kill()
            self._running_shells.pop(task_id, None)

    async def aclose(self, *, mcp_timeout: float = 5.0) -> None:
        """Async teardown: local tasks → team cleanup → MCP → storage."""
        self._teardown_local_tasks()

        # Only the leader fires team cleanup — teammates share the leader's
        # set and would cancel siblings mid-flight.
        if self._team is not None and self._team_member_name is None:
            with contextlib.suppress(Exception):
                await self._team.cleanup_session_teams()

        await self._mcp_runtime.disconnect_all(
            session_id=self._session_id,
            timeout_sec=mcp_timeout,
        )

        self._session_runtime.close_storage()

    def close(self, *, mcp_timeout: float = 5.0) -> None:
        """Sync teardown wrapper; refuses fire-and-forget on a live MCP manager."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose(mcp_timeout=mcp_timeout))
            return
        if self._mcp_manager is not None:
            raise RuntimeError(
                "AgentSession.close() called inside a running event loop with a "
                "live MCP manager. Use `await agent.aclose(mcp_timeout=...)` "
                "instead — fire-and-forget close was removed in v0.11."
            )
        self._teardown_local_tasks()
        self._session_runtime.close_storage()

    async def __aenter__(self) -> AgentSession:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: object | None,
    ) -> None:
        await self.aclose()


def build_agent(
    config: AuraConfig,
    *,
    hooks: HookChain | None = None,
    available_tools: dict[str, BaseTool] | None = None,
    session_id: str = _DEFAULT_SESSION,
    session_rules: SessionRuleSet | None = None,
    question_asker: UserAsker | None = None,
    mode: str = "default",
    disable_bypass: bool = False,
    ruleset: RuleSet | None = None,
    deny_ruleset: RuleSet | None = None,
    ask_ruleset: RuleSet | None = None,
    safety: SafetyPolicy | None = None,
) -> AgentSession:
    # Production convenience: resolves model + storage; AgentSession ctor stays DI-pure.
    provider, model_name = llm.resolve(config.router["default"], cfg=config)
    model = llm.create(provider, model_name)
    storage = SessionStorage(config.resolved_storage_path())
    return AgentSession(
        config=config,
        model=model,
        storage=storage,
        hooks=default_hooks().merge(hooks or HookChain()),
        available_tools=available_tools,
        session_id=session_id,
        session_rules=session_rules,
        question_asker=question_asker,
        mode=mode,
        disable_bypass=disable_bypass,
        ruleset=ruleset,
        deny_ruleset=deny_ruleset,
        ask_ruleset=ask_ruleset,
        safety=safety,
    )
