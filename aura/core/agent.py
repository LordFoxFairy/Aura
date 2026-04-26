"""Agent facade — config + model + storage + hooks 组装成一条对话的入口层。"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from aura.core.tasks.types import TaskNotification

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from aura.config.schema import AuraConfig, AuraConfigError
from aura.core import llm
from aura.core.abort import AbortController, AbortException
from aura.core.compact import (
    MICROCOMPACT_KEEP_RECENT,
    MICROCOMPACT_TRIGGER_PAIRS,
    CompactResult,
    MicrocompactPolicy,
    run_compact,
)
from aura.core.compact.constants import AUTO_COMPACT_THRESHOLD
from aura.core.hooks import HookChain
from aura.core.hooks.bash_safety import make_bash_safety_hook
from aura.core.hooks.budget import default_hooks
from aura.core.hooks.must_read_first import make_must_read_first_hook
from aura.core.loop import DEFAULT_SESSION as _DEFAULT_SESSION
from aura.core.loop import AgentLoop
from aura.core.mcp import MCPManager
from aura.core.memory import project_memory, rules
from aura.core.memory.context import Context, _ReadRecord
from aura.core.memory.system_prompt import build_system_prompt
from aura.core.permissions.denials import DENIALS_SINK_KEY, PermissionDenial
from aura.core.permissions.mode import Mode
from aura.core.permissions.safety import SafetyPolicy
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.registry import ToolRegistry
from aura.core.skills import Skill, SkillRegistry, load_skills
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.store import TasksStore
from aura.schemas.events import AgentEvent, AssistantDelta, Final
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolError
from aura.tools import BUILTIN_STATEFUL_TOOLS, BUILTIN_TOOLS
from aura.tools.ask_user import QuestionAsker

# Substring signatures that identify provider-level "context length exceeded"
# errors. We match on stringified message — not exception type — so we don't
# have to import openai / anthropic SDK types and so custom/wrapper models
# keep working. Lowercased at compare time; covers OpenAI, Anthropic,
# Google, and generic "too many tokens" phrasings.
_CONTEXT_OVERFLOW_PHRASES: tuple[str, ...] = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "prompt is too long",
    "too many tokens",
)


def _is_context_overflow(exc: BaseException) -> bool:
    """True iff ``exc``'s message matches a known context-overflow signature."""
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _CONTEXT_OVERFLOW_PHRASES)


async def _unavailable_question_asker(
    question: str, options: list[str] | None, default: str | None,
) -> str:
    # Registered when no ``question_asker`` was injected (e.g. SDK caller
    # drives astream without a REPL). The tool stays visible to the LLM —
    # invoking it surfaces this error as a ToolError in the tool result,
    # not a crash, so the model can pivot.
    raise ToolError(
        "ask_user_question is unavailable: no CLI asker was injected. "
        "Run aura through the CLI, or pass question_asker=... to "
        "build_agent(...) / Agent(...) when driving programmatically."
    )


class Agent:
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
        question_asker: QuestionAsker | None = None,
        auto_compact_threshold: int = AUTO_COMPACT_THRESHOLD,
        auto_microcompact_enabled: bool = True,
        microcompact_trigger_pairs: int = MICROCOMPACT_TRIGGER_PAIRS,
        microcompact_keep_recent: int = MICROCOMPACT_KEEP_RECENT,
        session_log_dir: Path | None = None,
        pre_loaded_skills: SkillRegistry | None = None,
        mode: str = "default",
        system_prompt_suffix: str = "",
        disable_bypass: bool = False,
        inherited_reads: Mapping[Path, _ReadRecord] | None = None,
        ruleset: RuleSet | None = None,
        safety: SafetyPolicy | None = None,
    ) -> None:
        # ``session_rules``: CLI hands in the same SessionRuleSet that was used
        # to build the permission hook; Agent.clear_session drops its runtime
        # rules alongside history and state so /clear is coherent.
        self._config = config
        self._model = model
        # Live model spec — distinct from ``config.router["default"]`` which
        # is the CONFIG surface and stays immutable. ``switch_model`` mutates
        # only this field so the status bar + /model status reflect the
        # currently-in-use model, while a subsequent ``clear_session`` or a
        # fresh CLI run still starts from the configured default.
        self._current_model_spec = config.router.get("default", "")
        self._storage = storage
        self._hooks = hooks or HookChain()
        self._state = LoopState()
        # G5: per-turn deny records. The permission hook appends to this
        # list (via ``state.custom[DENIALS_SINK_KEY]`` which aliases the
        # same object), Loop clears it at the start of each run_turn, and
        # ``last_turn_denials()`` exposes an immutable tuple view. The
        # sink MUST be seeded into state.custom BEFORE the hook runs the
        # first time, so we wire it eagerly here. Shared-by-reference:
        # when Loop ``.clear()``s the list at turn start, state.custom's
        # value (same object) is cleared too — no re-seeding needed.
        self._turn_denials: list[PermissionDenial] = []
        self._state.custom[DENIALS_SINK_KEY] = self._turn_denials
        self._session_rules = session_rules
        # Permission mode — the CLI resolves the effective mode (config +
        # --bypass-permissions flag) and hands it in. Stored here so the
        # status bar can surface it without reaching back into the store
        # each render. Valid values: "default" / "accept_edits" / "plan" /
        # "bypass"; enforcement still happens in the permission hook.
        # Org-level kill switch for bypass mode. When true, any attempt
        # to enter ``mode="bypass"`` (at construction time OR via
        # ``set_mode``) is refused with ``AuraConfigError``. Threaded in
        # from ``PermissionsConfig.disable_bypass`` by the CLI so a single
        # config flag can centrally refuse bypass in shared / CI /
        # compliance environments. Set BEFORE the mode assignment so the
        # same guard fires on both paths.
        self._disable_bypass = disable_bypass
        if disable_bypass and mode == "bypass":
            raise AuraConfigError(
                source="PermissionsConfig",
                detail=(
                    "bypass mode is disabled by config "
                    "(permissions.disable_bypass=true); "
                    "refusing to construct Agent(mode='bypass')"
                ),
            )
        self._mode = mode
        # prePlanMode parity with claude-code: remember whichever mode the
        # user was in BEFORE enter_plan_mode flipped them into ``plan``, so
        # exit_plan_mode can restore it on approval instead of always
        # landing on ``default``. Written exactly once per enter cycle via
        # ``_capture_prior_mode``; cleared on ``clear_session`` so /clear
        # doesn't leak a stale value into the next session. ``None`` =
        # "no plan entry has happened yet on this session".
        self._prior_mode: str | None = None
        # Auto-compact trigger. Non-zero = enabled. When a turn completes
        # successfully and total_tokens_used crosses the threshold, astream
        # calls self.compact(source="auto") before returning. 0 disables it.
        self._auto_compact_threshold = auto_compact_threshold
        # G2 microcompact configuration. Parity with auto_compact's
        # "zero disables" pattern: ``trigger_pairs <= 0`` OR
        # ``auto_microcompact_enabled=False`` disables the feature
        # entirely (``_build_loop`` passes ``None`` to AgentLoop in that
        # case). Validation runs at construction — fail fast on a
        # misconfig that would silently never clear anything. The guard
        # is skipped on the disabled paths so explicit ``trigger_pairs=0``
        # (the documented "zero disables" handle) doesn't trip it.
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
        # Skills: user-layer (~/.aura/skills/) + project-layer (<cwd>/.aura/skills/).
        # Loaded once at Agent init; not re-scanned on /clear (v0.2.0 MVP — no
        # hot reload). Collision resolution inside the loader logs to journal.
        # When ``pre_loaded_skills`` is passed in (subagent path), use that
        # registry directly — skips the disk scan and guarantees exact parity
        # with the parent's skill set.
        self._cwd = Path.cwd()
        if pre_loaded_skills is not None:
            self._skill_registry = pre_loaded_skills
        else:
            self._skill_registry = load_skills(cwd=self._cwd)
        # Subagent plumbing. Built AFTER _skill_registry so the factory can
        # hand the parent's (this Agent's) pre-loaded skills through to any
        # child Agent it spawns — matches claude-code's "subagent inherits
        # parent tool set" semantics.
        self._tasks_store = TasksStore()
        # ``parent_read_records_provider`` — a live view into this Agent's
        # Context._read_records. Factory calls it at each ``spawn`` to
        # snapshot the LATEST parent reads (not the startup state), so
        # files the parent read mid-session before calling task_create
        # still show as fresh in the child (Workstream G8). Closes over
        # ``self`` so ``clear_session`` (which swaps _context) is tracked
        # automatically — the next spawn reads through the refreshed
        # attribute rather than a stale Context reference.
        # C1: plumb permission inputs into the factory so every spawned
        # subagent gets a hook with the same rules + safety + live mode
        # as the parent. ``parent_mode_provider`` closes over ``self`` so
        # mid-session mode changes (shift+tab, enter_plan_mode) are
        # visible to spawn. ``ruleset`` / ``safety`` are immutable
        # snapshots captured at Agent construction — matching how the
        # parent's own hook was built from the same snapshots at CLI
        # startup. When either is ``None`` the factory skips installing
        # the hook (tests / SDK callers that never set up permissions).
        self._subagent_factory = SubagentFactory(
            parent_config=self._config,
            parent_model_spec=self._config.router.get("default", ""),
            parent_skills=self._skill_registry,
            parent_read_records_provider=lambda: self._context._read_records,
            parent_ruleset=ruleset,
            parent_safety=safety,
            parent_mode_provider=lambda: self._mode,
            parent_session=self._session_rules,
        )
        # F-07-005 abort cascade — wrap ``spawn`` so every child Agent
        # gets a registered :class:`AbortController` in this Agent's
        # ``_running_aborts`` map. The child's astream picks the
        # controller up via the contextvar (the surrounding
        # asyncio.Task carries our parent context), and the
        # ``_cascade_abort_to_children`` fan-out fires every entry so
        # a single user Ctrl+C tears the whole subagent tree down.
        # The child Agent's stamped ``_current_abort`` is the SAME
        # controller registered here, so reading
        # ``parent._running_aborts.values()`` yields live controllers
        # the test (and the cascade) can flip directly.
        original_spawn = self._subagent_factory.spawn

        def _spawn_with_abort(*args, **kwargs):  # type: ignore[no-untyped-def]
            # ``model_spec`` may not be accepted by the legacy factory
            # surface — drop it before calling through. ``task_id`` is
            # the key we register under.
            task_id = kwargs.get("task_id")
            try:
                child = original_spawn(*args, **kwargs)
            except TypeError as exc:
                if "model_spec" in str(exc) and "model_spec" in kwargs:
                    kwargs.pop("model_spec", None)
                    child = original_spawn(*args, **kwargs)
                else:
                    raise
            # Allocate + register the child's controller. Keyed by
            # task_id when the caller supplied one (task_create flow);
            # fall back to the child's own session_id otherwise so the
            # registry remains uniquely keyed.
            controller = AbortController()
            key = task_id if task_id is not None else child.session_id
            self._running_aborts[key] = controller
            # Stamp the controller on the child so its astream picks
            # it up directly — bypassing the "create my own" branch.
            # Astream reads ``self._current_abort`` only as the
            # external observation surface; the actual signal it uses
            # is whatever it sets at the top. We extend astream to
            # honour a pre-set ``_inherited_abort`` if present.
            object.__setattr__(child, "_inherited_abort", controller)
            # When the child finishes (naturally or via cascade), the
            # entry stays in _running_aborts until parent's astream
            # finally clause clears it on a clean turn end. Tests that
            # assert "every child controller flipped" rely on the
            # controller being kept alive past the run_task done
            # callback, so we don't add a remove-on-done hook here.
            return child

        self._subagent_factory.spawn = _spawn_with_abort  # type: ignore[method-assign]
        # Map: task_id -> the detached asyncio.Task handle. Shared with the
        # ``task_create`` tool so Agent.close() can cancel still-running
        # subagents without reaching back into the tool's internals.
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        # Map: task_id -> the live asyncio.subprocess.Process for shell
        # (bash_background) tasks. Shared with ``bash_background`` (which
        # writes on spawn + removes on natural exit) and ``task_stop``
        # (which reads + kills). Lives on the Agent for the same reason
        # ``_running_tasks`` does — so ``Agent.close()`` can tear down
        # orphan children deterministically.
        self._running_shells: dict[str, asyncio.subprocess.Process] = {}
        # F-07-005 / Round 6L abort registry. ``task_id`` (or
        # ``team-member`` synthetic id) → AbortController. Populated by
        # ``run_task`` (subagents) and ``TeamManager.add_member`` (team
        # runtimes); the parent's ``current_abort.abort()`` cascade
        # iterates this dict and flips every child's controller so a
        # single user Ctrl+C tears the whole tree down.
        self._running_aborts: dict[str, AbortController] = {}
        # Round 6L. Populated by ``join_team``; ``None`` outside a team.
        # Typed loose (``object | None``) to avoid a circular import on
        # :class:`aura.core.teams.manager.TeamManager`.
        self._team: object | None = None
        # Round 6L. ``None`` for the leader / non-team agents; set to
        # the member name by :meth:`join_team` for teammates so
        # :class:`SendMessage` can stamp the right ``sender``.
        self._team_member_name: str | None = None
        # F-05-003 partial-text buffer. ``Agent.astream`` appends to
        # this on every AssistantDelta event; if abort fires before the
        # final AIMessage we yield this as one last AssistantDelta so
        # the user doesn't lose half-streamed reasoning. Reset at the
        # start of each astream call.
        self._partial_assistant_text: str = ""
        # F-04-014: SessionStart fires exactly once per session.
        # Cleared by ``clear_session`` / ``resume_session`` (re-arm).
        self._session_start_fired = False
        # Round 4F notification queue. Populated by external producers
        # (TasksStore terminal-listener), drained by Context.build at
        # the start of each prompt envelope. Owned by Agent so /clear
        # can wipe it.
        from aura.core.tasks.types import TaskNotification as _TN
        self._pending_notifications: list[_TN] = []

        # Round 4F — wire the TasksStore terminal listener so subagent
        # completions / failures flow into the parent's notification
        # queue. The listener closes over ``self`` so the listener
        # outlives any specific record reference; the bounded
        # _enqueue_task_notification call drops oldest on overflow.
        def _on_terminal(rec: object) -> None:
            from aura.core.tasks.types import TaskNotification, TaskRecord
            if not isinstance(rec, TaskRecord):
                return
            summary = (
                rec.progress.latest_summary
                or rec.final_result
                or rec.error
            )
            self._enqueue_task_notification(TaskNotification(
                task_id=rec.id,
                status=rec.status,
                summary=summary,
                description=rec.description,
            ))
        self._tasks_store.add_terminal_listener(_on_terminal)
        # F-01-001: live abort controller for the running astream call.
        # Set at the top of :meth:`astream` and cleared on exit.
        self._current_abort: AbortController | None = None
        # Stateless built-ins come from shared singletons; stateful ones are
        # instantiated per-Agent so each gets its own dependency (LoopState
        # for todo_write, QuestionAsker for ask_user_question).
        self._available_tools = (
            dict(available_tools) if available_tools is not None else dict(BUILTIN_TOOLS)
        )
        for name, cls in BUILTIN_STATEFUL_TOOLS.items():
            # Explicit per-tool wiring. Ugly if/elif, but readable: each
            # stateful tool gets the dependency it asked for. Revisit if a
            # third stateful tool lands with a different dep shape.
            if name == "todo_write":
                self._available_tools[name] = cls(state=self._state)
            elif name == "ask_user_question":
                self._available_tools[name] = cls(
                    asker=question_asker or _unavailable_question_asker,
                )
            elif name == "task_create":
                self._available_tools[name] = cls(
                    store=self._tasks_store,
                    factory=self._subagent_factory,
                    running=self._running_tasks,
                    transcript_storage=self._storage,
                )
            elif name == "task_output" or name == "task_get" or name == "task_list":
                self._available_tools[name] = cls(store=self._tasks_store)
            elif name == "task_stop":
                self._available_tools[name] = cls(
                    store=self._tasks_store,
                    running=self._running_tasks,
                    running_shells=self._running_shells,
                )
            elif name == "bash_background":
                self._available_tools[name] = cls(
                    store=self._tasks_store,
                    running_shells=self._running_shells,
                    running_tasks=self._running_tasks,
                )
            elif name == "web_search":
                # web_search takes an optional WebSearchConfig; when the user
                # did not declare ``web_search:`` in config, the tool falls
                # back to its own defaults (DuckDuckGo, max_results=5).
                self._available_tools[name] = cls(config=self._config.web_search)
            elif name == "enter_plan_mode":
                # Plan-mode control tools. Close over ``set_mode`` and the
                # ``_mode`` read rather than injecting ``self`` so the
                # tool's reach into the Agent is exactly one arrow: flip
                # the permission mode. Same "closure-over-method" pattern
                # as QuestionAsker.
                # ``save_prior_mode`` remembers the pre-plan mode so the
                # companion exit_plan_mode can restore it on approval
                # (claude-code prePlanMode parity).
                self._available_tools[name] = cls(
                    mode_setter=self.set_mode,
                    mode_getter=lambda: self._mode,
                    save_prior_mode=self._capture_prior_mode,
                )
            elif name == "exit_plan_mode":
                # Same mode_setter/mode_getter pattern as enter_plan_mode,
                # PLUS an ``asker`` because exit_plan_mode requires user
                # approval BEFORE mutating mode (matches claude-code's
                # ExitPlanModeV2Tool.checkPermissions ask-behavior). The
                # same QuestionAsker that ask_user_question uses is wired
                # through here — reusing the CLI's prompt_toolkit Yes/No
                # picker for free. ``get_prior_mode`` lets the tool
                # restore to whatever mode was active before plan.
                self._available_tools[name] = cls(
                    mode_setter=self.set_mode,
                    mode_getter=lambda: self._mode,
                    asker=question_asker or _unavailable_question_asker,
                    get_prior_mode=lambda: self._prior_mode,
                )
            elif name == "skill":
                # LLM-invocable skill trigger. ``recorder`` closes over
                # Agent.record_skill_invocation so the tool never holds
                # a reference to Agent itself — same "one arrow" pattern
                # as enter_plan_mode. Registry is handed in directly
                # because name-lookup is pure read. ``session_id_provider``
                # feeds ``${AURA_SESSION_ID}`` substitution; closure over
                # self so a late session_id change (should never happen
                # today) still resolves to the live value.
                self._available_tools[name] = cls(
                    recorder=self.record_skill_invocation,
                    registry=self._skill_registry,
                    # ``_session_id`` is assigned later in __init__ (line ~249);
                    # the lambda closes over ``self`` and reads the live
                    # attribute at invocation time, so the ordering is safe.
                    # ``getattr`` + default keeps mypy happy (the attribute
                    # isn't visible to it at this point in the method).
                    session_id_provider=(
                        lambda: getattr(self, "_session_id", _DEFAULT_SESSION)
                    ),
                    # v0.13 ``allowed-tools`` enforcement: hand the tool a
                    # live view into this Agent's SessionRuleSet so skill
                    # invocation can install permissive auto-allow rules
                    # for its declared tools (matches
                    # ``SkillCommand.handle`` on the slash path). Closure
                    # over ``self._session_rules`` — ``/clear`` calls
                    # ``.clear()`` on the same instance, so the tool
                    # automatically sees the fresh (empty) ruleset on the
                    # next invocation without re-wiring.
                    session_rules_provider=lambda: self._session_rules,
                    # V14 ``restrict-tools`` lease — closure over
                    # ``self._state`` so install can read the live
                    # turn_count and stamp the expiry sentinel. /clear
                    # mutates ``_state`` in place, so the lambda always
                    # returns the live instance.
                    loop_state_provider=lambda: self._state,
                )
            elif name == "send_message":
                # Phase A teams. The tool walks ``self._team`` /
                # ``self._team_member_name`` at invocation time, so
                # it only needs an Agent back-reference. ``join_team``
                # auto-registers; outside a team the tool errors clean.
                self._available_tools[name] = cls(agent=self)
            else:  # pragma: no cover — guardrail for future additions
                raise RuntimeError(f"unwired stateful tool: {name}")
        self._session_id = session_id
        # Session-scoped journal: when ``session_log_dir`` is passed, every
        # journal.write made during astream routes to a per-session JSONL
        # file. Enables two concurrent Agents in the same process (subagents,
        # server workers) to keep their audit trails fully separate.
        if session_log_dir is not None:
            session_log_dir.mkdir(parents=True, exist_ok=True)
            self._session_log_path: Path | None = (
                session_log_dir / f"{self._session_id}.jsonl"
            )
        else:
            self._session_log_path = None
        # config.tools.enabled → lookup → ToolRegistry. Built once per Agent.
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
        # ``system_prompt_suffix`` is appended verbatim to the base system
        # prompt. Populated only by the subagent factory today (per the
        # selected agent_type); always empty for top-level Agents. Stored on
        # self so ``clear_session`` can rebuild the prompt identically.
        self._system_prompt_suffix = system_prompt_suffix
        self._system_prompt = build_system_prompt() + system_prompt_suffix
        self._primary_memory = project_memory.load_project_memory(self._cwd)
        self._rules = rules.load_rules(self._cwd)
        # ``inherited_reads`` (Workstream G8) only flows into the FIRST
        # Context construction — /clear and /compact build their own fresh
        # Contexts and must NOT resurrect a long-gone parent's read
        # fingerprints, so we do NOT store this on self. Subagent spawn
        # re-snapshots the parent at each ``SubagentFactory.spawn`` call.
        self._context = self._build_context(inherited_reads=inherited_reads)
        # Hard-floor bash safety — Tier A shell attacks (zsh builtins, CR
        # parser differential, malformed+separator, cd+git compound). Inserted
        # at pre_tool[0] so it precedes any caller-supplied permission hook —
        # safety is a separate axis from permission and CANNOT be overridden
        # by rules or ``--bypass-permissions``. Stateless; tracked as a field
        # so clear_session can re-insert it at position 0 idempotently.
        # Live mode provider: safety hook must honor ``mode == "bypass"``
        # (user opted in) and track mid-session ``set_mode`` changes, same
        # as the permission hook. Closing over ``self`` means a shift+tab
        # / enter_plan_mode mid-turn is visible on the next tool call.
        # ``self._mode`` is typed ``str`` on Agent (no circular dep on
        # permissions.mode); cast here since every writer guarantees a
        # valid Mode literal — same pattern used in ``aura/cli/__main__.py``
        # for the permission hook's ``_live_mode``.
        self._bash_safety_hook = make_bash_safety_hook(
            mode_provider=lambda: cast("Mode", self._mode),
        )
        self._hooks.pre_tool.insert(0, self._bash_safety_hook)
        # Tool-intrinsic invariant (matches claude-code FileEditTool): edit_file
        # rejects before any user-supplied gate would run. Appended AFTER the
        # caller's hooks so permission (CLI-installed) runs first — if the user
        # denies the tool, we don't also yell about the missing read. Tracked as
        # a field so clear_session can swap it when Context is rebuilt.
        self._must_read_first_hook = make_must_read_first_hook(self._context)
        self._hooks.pre_tool.append(self._must_read_first_hook)
        # V14-HOOK-CATALOG: register the default file_changed +
        # cwd_changed consumers. These need a back-reference to the
        # Agent (they refresh ``_primary_memory`` / ``_context`` /
        # ``_rules`` in place), which is why they can't live in
        # ``default_hooks()`` (called before the Agent exists). Adding
        # them here mirrors the bash_safety / must_read_first wiring
        # above — the Agent is the single owner of its hook chain
        # post-construction. Imported lazily to avoid an import cycle:
        # auto_reload imports Agent for type-checking, Agent imports
        # auto_reload at runtime.
        from aura.core.hooks.auto_reload import (
            make_aura_md_reload_hook,
            make_cwd_rules_reload_hook,
        )
        self._hooks.file_changed.append(make_aura_md_reload_hook(self))
        self._hooks.cwd_changed.append(make_cwd_rules_reload_hook(self))
        self._loop = self._build_loop()
        # MCP is wired at construction to declare the slots, but no
        # connection happens here — aconnect() does that work async. Sync
        # construction MUST remain sync so the existing Agent(...) call
        # sites (tests, SDK users) don't have to thread an event loop.
        self._mcp_manager: MCPManager | None = None
        self._mcp_commands: list[object] = []
        # Estimated size of the pinned prompt prefix (system msg + memory +
        # rules + skill catalogue + tool schemas) in tokens. Computed once
        # at construction so the status bar has a number to anchor against
        # BEFORE the first turn, and also serves as the fallback indicator
        # on providers that don't support prompt caching (deepseek, etc.)
        # where ``cache_read_input_tokens`` will always be 0. Char count /
        # 4 is the standard rough approximation.
        self._pinned_tokens_estimate = self._estimate_pinned_tokens()
        # Round 6L. Snapshot "did the user pin a custom tools.enabled?"
        # so :meth:`join_team` can decide whether to auto-add
        # ``send_message`` (default: yes) or respect the user's pin.
        from aura.config.schema import ToolsConfig as _ShippedToolsConfig
        self._user_pinned_tools_allowlist_value = (
            list(self._config.tools.enabled)
            != list(_ShippedToolsConfig().enabled)
        )
        # Round 4E: hand the web_fetch tool a summary-model factory.
        # Best-effort — both ``make_summary_model_factory`` and
        # ``set_default_model_factory`` are sibling-tier surfaces that
        # may not yet be importable; skip silently when missing.
        # ``getattr``-based introspection avoids mypy errors when the
        # symbols haven't been added yet upstream (Tier D/F).
        try:
            from aura.core import llm as _llm_mod
            from aura.tools import web_fetch as _wf_mod
            _make_factory = getattr(_llm_mod, "make_summary_model_factory", None)
            _set_default = getattr(_wf_mod, "set_default_model_factory", None)
            if _make_factory is not None and _set_default is not None:
                _set_default(_make_factory(self._config, self._model))
        except ImportError:
            pass
        # Round 5H: auto-fire SessionStart on construction when a
        # running event loop is available. Sync construction sites
        # (CLI bootstrap before asyncio.run) skip the schedule and
        # rely on ``astream``'s safety-net call.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            asyncio.ensure_future(self.fire_session_start())

    async def astream(
        self,
        prompt: str,
        *,
        attachments: list[HumanMessage] | None = None,
        abort: AbortController | None = None,
    ) -> AsyncIterator[AgentEvent]:
        # Persistence order (matches claude-code QueryEngine.ts:431+451):
        #   1. Build history with attachments + user HumanMessage,
        #   2. ``storage.save`` BEFORE any model.ainvoke call,
        #   3. Run the turn; on success save again (tool results + assistant).
        # Why: if the model call crashes / the process is killed / the user
        # Ctrl-C's mid-stream, the user's input is ALREADY on disk. The next
        # session-resume sees the interrupted turn, not a black hole. Closes
        # the B2/G1 audit gap against claude-code.
        #
        # ``attachments``: optional HumanMessages to prepend BEFORE the user's
        # HumanMessage. CLI-layer @mention preprocessing (aura.cli.attachments)
        # builds ``<mcp-resource>`` envelopes that land here. Persisted with
        # the user turn so reactive-compact (and any retry) can read them
        # from history without re-injection.
        #
        # The session_scope context routes every journal.write made on this
        # task — including ones emitted deep inside the loop / tools — to the
        # per-session JSONL file. contextvars propagate across ``await`` and
        # ``async for`` on the same task, so nested awaits inherit the scope
        # automatically.
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
            # Round 5H lifecycle: SessionStart safety net for sync
            # construction sites that skipped the auto-fire path.
            await self.fire_session_start()
            # F-04-014 lifecycle: UserPromptSubmit composes
            # left-to-right and may rewrite ``prompt`` in place. The
            # runner is optional — survivor HookChain may not yet
            # carry the lifecycle slots; getattr-with-fallback keeps
            # us forward-compatible without forcing every minimal
            # HookChain test fixture to learn the new shape.
            ups_runner = getattr(self._hooks, "run_user_prompt_submit", None)
            if ups_runner is not None:
                prompt = await ups_runner(
                    session_id=self._session_id,
                    turn_count=self._state.turn_count,
                    user_text=prompt,
                    state=self._state,
                )
            history = self._storage.load(self._session_id)
            # F-05-004 user-turn rollback boundary: snapshot the
            # pre-attachment length so a cancel BEFORE any AIMessage
            # can pop the unanswered user turn off persisted history.
            history_len_before_user_turn = len(history)
            if attachments:
                history.extend(attachments)
            history.append(HumanMessage(content=prompt))
            self._storage.save(self._session_id, history)

            # F-01-003 / Bug 2 fix #1 — reset per-astream turn budget so
            # ``max_turns`` is per-user-turn, not per-Agent-lifetime
            # (claude-code parity: ``turnCount = 1`` initialised at
            # every ``query`` entry). Without this, a long prior
            # session pre-trips the cap on every fresh user prompt.
            self._state.turn_count = 0

            # F-01-001 abort plumbing. Precedence:
            # 1. Explicit ``abort=`` kwarg (callers like teams runtime).
            # 2. ``_inherited_abort`` stamped by the parent's
            #    ``_subagent_factory.spawn`` wrapper (F-07-005 cascade).
            # 3. A fresh controller we own ourselves.
            inherited = getattr(self, "_inherited_abort", None)
            local_abort = (
                abort
                if abort is not None
                else (inherited if inherited is not None else AbortController())
            )
            self._current_abort = local_abort
            self._partial_assistant_text = ""

            saw_ai_message = False
            retry_exc: BaseException | None = None
            try:
                try:
                    async for event in self._loop.run_turn(
                        history=history, abort=local_abort,
                    ):
                        if isinstance(event, AssistantDelta):
                            self._partial_assistant_text += event.text
                        yield event
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
                    # F-05-003 partial assistant text — flush whatever
                    # streamed before the abort so the user sees it.
                    if self._partial_assistant_text:
                        yield AssistantDelta(text=self._partial_assistant_text)
                        self._partial_assistant_text = ""
                    if is_abort:
                        # F-05-004 rollback: when no AIMessage landed,
                        # the user's HumanMessage is unanswered — drop
                        # it so the next astream sees a clean slate.
                        if not saw_ai_message:
                            del history[history_len_before_user_turn:]
                        self._storage.save(self._session_id, history)
                        # Cascade to children — single Ctrl+C tears
                        # the whole subagent / teammate tree down.
                        await self._cascade_abort_to_children(
                            local_abort.reason or "parent_aborted",
                        )
                        yield Final(message="(cancelled)", reason="aborted")
                        # Subagent path: when this Agent inherited an
                        # abort controller from a parent (i.e. WE are
                        # a subagent), re-raise the original exception
                        # so the surrounding ``run_task`` flips the
                        # TaskRecord to a terminal status (cancelled
                        # / failed) instead of marking ``completed``.
                        # Top-level agents (no inheritance) swallow
                        # the abort — they already yielded the Final
                        # so the caller's iteration ends cleanly.
                        if isinstance(exc, asyncio.CancelledError):
                            raise
                        if getattr(self, "_inherited_abort", None) is not None:
                            raise
                        return
                    # Pure CancelledError — preserve legacy behaviour.
                    yield Final(message="(cancelled)")
                    raise
            except Exception as exc:
                if not _is_context_overflow(exc):
                    raise
                journal.write(
                    "reactive_compact_triggered",
                    session=self._session_id,
                    error=str(exc),
                )
                retry_exc = exc
            finally:
                self._current_abort = None

            if retry_exc is not None:
                await self.compact(source="reactive")
                history = self._storage.load(self._session_id)
                async for event in self._loop.run_turn(history=history):
                    yield event

            self._storage.save(self._session_id, history)
            journal.write(
                "astream_end",
                session=self._session_id,
                history_len=len(history),
                total_tokens=self._state.total_tokens_used,
            )
            # Auto-compact post-turn. Bug 2 fix #2 — providers that
            # don't populate ``usage_metadata`` (DashScope error 1261,
            # some Ollama builds) leave ``total_tokens_used`` at zero
            # forever, so the trigger never armed. Fall back to a
            # char-based history estimator (len/4) so the trigger can
            # still arm even without provider usage envelopes.
            if self._auto_compact_threshold > 0:
                used = self._state.total_tokens_used
                used_estimator = used == 0
                if used_estimator:
                    used = self._estimate_history_tokens(history)
                if used > self._auto_compact_threshold:
                    journal.write(
                        "auto_compact_triggered",
                        session=self._session_id,
                        tokens=used,
                        threshold=self._auto_compact_threshold,
                        used_estimator=used_estimator,
                    )
                    await self.compact(source="auto")

    def switch_model(self, spec: str) -> None:
        """Swap the live model. Raises ``AuraConfigError`` on failure.

        Resolves ``spec`` (router alias or ``provider:model``), constructs a
        fresh LangChain model, and rebuilds the loop so subsequent turns use
        it. Tools / hooks / context / history are untouched — the ongoing
        conversation continues with the new model seeing the same state.

        ``config.router["default"]`` stays unchanged by design: it's the
        boot-time config, not the live spec. The live spec lives on
        ``self._current_model_spec``.
        """
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
        """Stash the pre-plan mode for the companion exit_plan_mode to read.

        Invoked ONLY by the ``enter_plan_mode`` tool via its injected
        ``save_prior_mode`` closure — no other code path writes this
        attribute. Single writer guarantees the prior-mode value is
        exactly "the mode the user was in when they first entered plan
        on this session" and never an intermediate state.
        """
        self._prior_mode = mode

    def clear_session(self) -> None:
        # F-04-014: fire Stop(reason="clear") via ensure_future so sync
        # call sites (the CLI's /clear command) don't have to thread an
        # event loop through every call.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            asyncio.ensure_future(self.fire_stop(reason="clear"))
        self._storage.clear(self._session_id)
        self._state.reset()
        # ``LoopState.reset`` wipes ``state.custom`` — the G5 denials sink
        # lived there and just vanished. Re-seed the slot + drop any
        # captured denials so the next turn opens clean. Pointing
        # ``_turn_denials`` at the freshly-seeded list (rather than
        # clearing in place) keeps the invariant "the list in state.custom
        # is the SAME object Agent exposes" simple to reason about.
        self._turn_denials = []
        self._state.custom[DENIALS_SINK_KEY] = self._turn_denials
        # Drop any captured prior mode — /clear starts a fresh session so
        # a leftover "accept_edits" from a previous plan cycle shouldn't
        # bleed into the next one.
        self._prior_mode = None
        if self._session_rules is not None:
            self._session_rules.clear()
        # /clear 语义：同时 invalidate memory/rules caches + 重建 Context。
        # progressive 状态（nested fragments / matched rules）随新实例自然清空 ——
        # 不做原地 reset，避免遗漏字段。
        project_memory.clear_cache(self._cwd)
        rules.clear_cache(self._cwd)
        self._primary_memory = project_memory.load_project_memory(self._cwd)
        self._rules = rules.load_rules(self._cwd)
        self._context = self._build_context()
        # Swap the must-read-first hook so it closes over the NEW Context —
        # the old one's _read_records is empty but tied to a dead instance.
        self._hooks.pre_tool.remove(self._must_read_first_hook)
        self._must_read_first_hook = make_must_read_first_hook(self._context)
        self._hooks.pre_tool.append(self._must_read_first_hook)
        # Re-anchor bash safety at pre_tool[0]. The hook is stateless so we
        # could skip this, but the swap keeps the invariant "safety is first"
        # independent of any future list mutations in clear_session.
        self._hooks.pre_tool.remove(self._bash_safety_hook)
        self._bash_safety_hook = make_bash_safety_hook(
            mode_provider=lambda: cast("Mode", self._mode),
        )
        self._hooks.pre_tool.insert(0, self._bash_safety_hook)
        # Re-arm SessionStart so /clear feels like a fresh session.
        self._session_start_fired = False
        # Drop pending notifications + partial buffer.
        self._pending_notifications.clear()
        self._partial_assistant_text = ""
        self._loop = self._build_loop()
        journal.write("session_cleared", session=self._session_id)

    async def compact(
        self, *, source: Literal["manual", "auto", "reactive"] = "manual",
    ) -> CompactResult:
        """Summarize old history, preserve session state, rebuild Context.

        Entry point for ``/compact`` and (future) auto-compact. The heavy
        lifting lives in :func:`aura.core.compact.run_compact`; this method
        exists so callers have a stable surface and so the skill/command
        layer doesn't need to reach into the compact module directly.
        """
        return await run_compact(self, source=source)

    def record_skill_invocation(self, skill: Skill) -> None:
        """Proxy to Context — appends ``skill`` to the invoked list.

        Exposed on Agent so that :class:`SkillCommand` (which is constructed
        with an Agent, not a Context) doesn't need to reach into a private
        attribute.
        """
        self._context.record_skill_invocation(skill)

    # ------------------------------------------------------------------
    # F-01-001 / F-05-003 abort + partial-text plumbing
    # ------------------------------------------------------------------

    @property
    def current_abort(self) -> AbortController | None:
        """The live :class:`AbortController` for the running astream call.

        ``None`` between turns. Tests / Ctrl+C handlers fire
        ``abort.abort(reason)`` against whatever is in flight without
        having to thread the controller through every layer.
        """
        return self._current_abort

    @property
    def cwd(self) -> Path:
        """The Agent's logical working directory. Read-only — mutate via
        :meth:`set_cwd` so :class:`CwdChangedHook` consumers fire."""
        return self._cwd

    @property
    def team(self) -> object | None:
        """The :class:`TeamManager` this Agent is bound to, or ``None``.

        Typed loose to avoid an import cycle. Callers cast at the use
        site when they need TeamManager methods.
        """
        return self._team

    @property
    def pending_notifications(self) -> tuple[TaskNotification, ...]:
        """Snapshot of queued :class:`TaskNotification` records.

        Read-only tuple. :meth:`_drain_task_notifications` is the
        write/clear endpoint used by Context.build.
        """
        return tuple(self._pending_notifications)

    def buffer_partial_assistant_text(self, text: str) -> None:
        """Append ``text`` to the partial-assistant buffer.

        Producer hook for streaming renderers / tests so an abort
        before the final AIMessage still surfaces partial reasoning.
        Reset on every new astream call.
        """
        self._partial_assistant_text += text

    def _enqueue_task_notification(self, notif: TaskNotification) -> None:
        """External producer hook — append ``notif`` to the queue.

        Unbounded at the queue level — the build-time renderer caps the
        emitted block at 5 entries (FIFO) and collapses the tail to a
        ``(N more earlier)`` line, so the parent's prompt envelope stays
        compact while the queue itself preserves order.
        """
        self._pending_notifications.append(notif)

    def _drain_task_notifications(self) -> list[TaskNotification]:
        """Pop every queued notification and return them, oldest first."""
        drained = list(self._pending_notifications)
        self._pending_notifications.clear()
        return drained

    async def _cascade_abort_to_children(self, reason: str) -> None:
        """Fire every controller in :attr:`_running_aborts`.

        Idempotent — already-aborted controllers skip silently.
        Yields once after the fan-out so each child watchdog gets a
        scheduler tick to observe the flipped event before the
        parent's astream finally clause unwinds.
        """
        for controller in list(self._running_aborts.values()):
            if not controller.aborted:
                controller.abort(reason)
        await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Round 6L: team membership
    # ------------------------------------------------------------------

    def join_team(
        self,
        *,
        manager: object,
        member_name: str | None = None,
    ) -> None:
        """Bind this Agent to a :class:`TeamManager`.

        Stamps ``_team`` (+ optional ``_team_member_name``) and
        auto-enables ``send_message`` when the user has not pinned a
        custom ``tools.enabled`` allowlist. Idempotent on the same
        manager / member name.
        """
        if self._team is manager and (
            member_name is None or self._team_member_name == member_name
        ):
            return
        self._team = manager
        if member_name is not None:
            self._team_member_name = member_name
        self._auto_enable_send_message_for_team()

    def leave_team(self) -> None:
        """Unbind from the team and conditionally drop send_message.

        Mirrors :meth:`join_team` — when WE auto-added send_message,
        WE remove it; user-pinned allowlists keep their tool.
        """
        self._team = None
        self._team_member_name = None
        self._auto_disable_send_message_for_team()

    def _auto_enable_send_message_for_team(self) -> None:
        """Register ``send_message`` + rebind the loop's bound model.

        Skips when the user pinned ``tools.enabled`` (their choice
        wins) or when the tool is already in the registry (idempotent
        on double-join).
        """
        if self._user_pinned_tools_allowlist_value:
            return
        if "send_message" in self._registry:
            return
        from aura.tools.send_message import SendMessage
        send_tool = SendMessage(agent=self)
        self._registry.register(send_tool)
        self._available_tools["send_message"] = send_tool
        self._loop._rebind_tools(self._registry.tools())

    def _auto_disable_send_message_for_team(self) -> None:
        """Unregister ``send_message`` if WE registered it."""
        if self._user_pinned_tools_allowlist_value:
            return
        if "send_message" not in self._registry:
            return
        self._registry.unregister("send_message")
        self._available_tools.pop("send_message", None)
        self._loop._rebind_tools(self._registry.tools())

    def _user_pinned_tools_allowlist(self) -> bool:
        """True iff the user supplied a custom ``tools.enabled`` value."""
        return self._user_pinned_tools_allowlist_value

    # ------------------------------------------------------------------
    # F-04-014 lifecycle hook fire helpers
    # ------------------------------------------------------------------

    async def fire_session_start(self) -> None:
        """Fire SessionStart hooks; idempotent on repeat fire."""
        if self._session_start_fired:
            return
        self._session_start_fired = True
        runner = getattr(self._hooks, "run_session_start", None)
        if runner is None:
            return
        await runner(
            session_id=self._session_id,
            mode=self._mode,
            cwd=self._cwd,
            model_name=self._current_model_spec,
            state=self._state,
        )

    async def fire_notification(self, *, kind: str, body: str) -> None:
        """Fire Notification hooks with the given ``(kind, body)``."""
        runner = getattr(self._hooks, "run_notification", None)
        if runner is None:
            return
        await runner(
            session_id=self._session_id,
            kind=kind,
            body=body,
            state=self._state,
        )

    async def fire_stop(self, *, reason: str) -> None:
        """Fire Stop hooks with the given ``reason``."""
        runner = getattr(self._hooks, "run_stop", None)
        if runner is None:
            return
        await runner(
            session_id=self._session_id,
            reason=reason,
            turn_count=self._state.turn_count,
            state=self._state,
        )

    # ------------------------------------------------------------------
    # Round 3A: session resume
    # ------------------------------------------------------------------

    def resume_session(self, session_id: str) -> int:
        """Swap the live session_id; reset state to fresh-session feel.

        Loads ``session_id``'s history from storage, updates the live
        session_id, zeroes turn_count + token usage, drops partial
        buffers, and re-arms SessionStart so the lifecycle fires again
        on the next astream. Raises ``KeyError`` if the requested
        session has no rows. Returns the message count of the
        resumed session.
        """
        history = self._storage.load(session_id)
        if not history:
            raise KeyError(
                f"session {session_id!r} has no persisted history"
            )
        self._session_id = session_id
        self._state.reset()
        self._turn_denials = []
        self._state.custom[DENIALS_SINK_KEY] = self._turn_denials
        self._partial_assistant_text = ""
        self._session_start_fired = False
        journal.write(
            "session_resumed",
            session=session_id,
            message_count=len(history),
        )
        return len(history)

    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def mcp_manager(self) -> MCPManager | None:
        """Live MCP manager, or ``None`` before/outside ``aconnect``.

        Exposed for the CLI-layer ``@mention`` preprocessor (see
        :mod:`aura.cli.attachments`), which needs read access to the
        resources catalogue and ``read_resource`` without reaching into a
        private attribute. Always ``None`` when no servers are configured
        or when ``aconnect`` hasn't run yet — the caller short-circuits on
        that case.
        """
        return self._mcp_manager

    @property
    def current_model(self) -> str:
        """Live model spec — the string passed to ``switch_model`` most
        recently, or ``config.router["default"]`` if no switch yet."""
        return self._current_model_spec

    @property
    def mode(self) -> str:
        """Effective permission mode — one of
        ``default`` / ``accept_edits`` / ``plan`` / ``bypass``. Read-only
        mirror of what the CLI installed at construction. Surfaced here
        so the bottom status bar can display the current mode without
        re-reading the permission store each render."""
        return self._mode

    async def set_cwd(self, path: Path) -> None:
        """Move the Agent's working directory and notify ``cwd_changed``.

        V14-HOOK-CATALOG. ``path`` is resolved (so ``~`` and relative
        forms normalize), the new value lands on ``self._cwd``, a
        ``cwd_changed`` journal event is emitted, and every registered
        :class:`CwdChangedHook` consumer fires with ``(old_cwd, new_cwd)``.
        Default-shipped consumer refreshes project memory + rules from
        the new cwd (see :func:`aura.core.hooks.auto_reload
        .make_cwd_rules_reload_hook`).

        We do NOT shell out to ``os.chdir`` here — the Agent's ``_cwd``
        and the process's CWD are deliberately distinct (the Agent
        owning a logical workdir is enough for memory + rules + skill
        loading; mutating the process CWD would race against tools
        running concurrently). External ``os.chdir`` calls are out of
        scope by design.
        """
        new_cwd = Path(path).expanduser().resolve()
        old_cwd = self._cwd
        if new_cwd == old_cwd:
            # No-op — don't emit a journal event for "moved to the same
            # place". Same defensiveness as ``set_mode`` returning
            # without journalling on a same-mode call would have, if
            # we'd written it that way.
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
        """Update the current permission mode.

        Valid values: ``default`` / ``accept_edits`` / ``plan`` /
        ``bypass``. The CLI's shift+tab keybinding uses this to cycle
        among the three non-bypass modes at runtime; ``bypass`` remains
        settable programmatically (CLI entry point uses it) but is
        deliberately excluded from the interactive cycle — it can only
        be enabled via ``--bypass-permissions``.

        When the Agent was constructed with ``disable_bypass=True``
        (from ``PermissionsConfig.disable_bypass``), switching to
        ``bypass`` raises ``AuraConfigError`` — same kill switch as the
        CLI flag path, just at a different entry point.
        """
        valid = {"default", "accept_edits", "plan", "bypass"}
        if mode not in valid:
            raise ValueError(
                f"invalid mode {mode!r}; expected one of {sorted(valid)}"
            )
        if mode == "bypass" and self._disable_bypass:
            raise AuraConfigError(
                source="PermissionsConfig",
                detail=(
                    "bypass mode is disabled by config "
                    "(permissions.disable_bypass=true); "
                    "refusing set_mode('bypass')"
                ),
            )
        self._mode = mode
        journal.write("mode_changed", session=self._session_id, mode=mode)

    @property
    def context_window(self) -> int:
        """Effective context window in tokens. Honors
        ``AuraConfig.context_window`` override when set, otherwise falls
        back to ``aura.core.llm.get_context_window`` for the current
        model. Kept on Agent so the bottom bar has one clean place to
        read it from rather than re-resolving on every render."""
        if self._config.context_window is not None:
            return self._config.context_window
        from aura.core.llm import get_context_window
        return get_context_window(self.current_model)

    @property
    def pinned_tokens_estimate(self) -> int:
        """Estimated size of the pinned prompt prefix in tokens.

        Char-count / 4 approximation over: system message, project
        memory, rules, skill catalogue, and the JSON-serialized schemas
        of all currently-bound tools. Real
        ``cache_read_input_tokens`` from a provider response is
        strictly more accurate — but many providers (e.g. deepseek)
        don't support prompt caching and always return 0, so the
        status bar falls back to this estimate to give the operator
        *some* anchor for the pinned channel size. Also surfaced
        BEFORE the first turn, so the REPL opens with a meaningful
        number instead of a zero."""
        return self._pinned_tokens_estimate

    def _estimate_history_tokens(self, history: list[BaseMessage]) -> int:
        """Char-count / 4 fallback when ``usage_metadata`` is missing.

        Bug 2 fix #2 / F-01-004. Aliyun DashScope (error 1261), some
        Ollama builds, and a few self-hosted backends never populate
        the LangChain usage envelope, leaving ``total_tokens_used``
        pinned at zero — so the auto-compact trigger could never arm
        on a runaway session. The 4-chars-per-token approximation is
        good enough to cross the (large) threshold once exhaustion
        actually happens.
        """
        char_count = 0
        for msg in history:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                char_count += len(content)
            else:
                char_count += len(str(content))
        return char_count // 4

    def _estimate_pinned_tokens(self) -> int:
        # Build the pinned messages with an empty history — everything
        # from Context.build that doesn't depend on live turn state.
        import json

        char_count = 0
        for message in self._context.build([]):
            content = getattr(message, "content", "")
            if isinstance(content, str):
                char_count += len(content)
        # Tool schemas go into every request as a separate payload the
        # provider also bills against the cached prefix. Approximate via
        # name + description + JSON-serialized args schema.
        for tool in self._registry.tools():
            char_count += len(tool.name or "")
            char_count += len(tool.description or "")
            try:
                schema = json.dumps(
                    getattr(tool, "args", {}) or {},
                    default=str,
                    ensure_ascii=False,
                )
            except (TypeError, ValueError):
                schema = ""
            char_count += len(schema)
        # Standard 4-chars-per-token approximation. Not exact (provider
        # tokenizers vary) but consistently in the ballpark, which is
        # all an at-a-glance status indicator needs.
        return char_count // 4

    @property
    def router_aliases(self) -> dict[str, str]:
        """除 'default' 之外的别名 → 'provider:model' 映射。"""
        return {k: v for k, v in self._config.router.items() if k != "default"}

    @property
    def session_id(self) -> str:
        return self._session_id

    def last_turn_denials(self) -> tuple[PermissionDenial, ...]:
        """Immutable view of permission denials from the most recent turn.

        Populated by the permission hook on every non-allow decision
        (``safety_blocked`` / ``plan_mode_blocked`` / ``user_deny``).
        Cleared at the start of each ``AgentLoop.run_turn`` so a turn
        that denies zero tools opens with an empty tuple. Between turns
        (after astream returns) the list holds the just-finished turn's
        denials so SDK / plugin / UI code can inspect them without
        parsing journal JSONL.

        Returns a tuple — mutation attempts raise ``TypeError`` /
        ``AttributeError``. The underlying list (``_turn_denials``) is
        owned by the Agent and updated in-place; we return a snapshot
        tuple so the caller never gets a reference that could racily
        grow under them between turns.

        Workstream G5 — ``docs/specs/2026-04-23-aura-main-channel-parity.md``.
        """
        return tuple(self._turn_denials)

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
        return AgentLoop(
            model=self._model,
            registry=self._registry,
            context=self._context,
            hooks=self._hooks,
            state=self._state,
            retry_config=self._config.retry,
            session_id=self._session_id,
            microcompact_policy=policy,
        )

    def _build_context(
        self,
        *,
        inherited_reads: Mapping[Path, _ReadRecord] | None = None,
    ) -> Context:
        return Context(
            cwd=self._cwd,
            system_prompt=self._system_prompt,
            primary_memory=self._primary_memory,
            rules=self._rules,
            skills=self._skill_registry.list(),
            todos_provider=lambda: self._state.custom.get("todos", []),
            notifications_drainer=self._drain_task_notifications,
            inherited_reads=inherited_reads,
        )

    async def aconnect(self) -> None:
        """Establish MCP connections and register discovered tools / prompts.

        Must be called before the first turn if ``mcp_servers`` are
        configured. No-op if no servers are configured. Failures are
        journalled and swallowed — the agent starts without the failing
        servers' tools (graceful degradation is a v0.3.0 non-negotiable).
        """
        if not self._config.mcp_servers:
            return
        try:
            manager = MCPManager(self._config.mcp_servers)
            tools, commands = await manager.start_all()
        except Exception as exc:  # noqa: BLE001
            journal.write(
                "mcp_aconnect_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        self._mcp_manager = manager
        for t in tools:
            # Collisions with built-ins or a previous MCP discovery pass
            # would raise; journal + skip so a duplicate doesn't take the
            # whole aconnect down.
            try:
                self._registry.register(t)
            except ValueError as exc:
                journal.write(
                    "mcp_tool_register_skipped",
                    tool=t.name,
                    error=str(exc),
                )
        self._mcp_commands = list(commands)
        # MCP resources are exposed via the CLI-layer ``@server:uri`` mention
        # preprocessor (see :mod:`aura.cli.attachments`), NOT as an LLM tool.
        # Claude-code parity: the user attaches resources by naming them
        # inline; the preprocessor resolves + injects the body before the
        # turn hits the model. The prior ``mcp_read_resource`` auto-
        # registration was removed in v0.10.x — it inverted the control
        # direction (LLM had to invent URIs) and silently re-pulled
        # resources turn after turn. :class:`aura.tools.mcp_read_resource
        # .MCPReadResourceTool` is still importable for programmatic SDK
        # users who want LLM-driven reads; the resource surface just isn't
        # wired into the default agent anymore.
        catalogue = manager.resources_catalogue()
        self._loop._rebind_tools(self._registry.tools())
        journal.write(
            "mcp_aconnect_done",
            tool_count=len(tools),
            command_count=len(commands),
            resource_count=len(catalogue),
        )

    # ------------------------------------------------------------------
    # Shutdown (B3): timeout-bounded, cancel-on-timeout MCP teardown.
    # ------------------------------------------------------------------
    #
    # The pre-B3 ``close`` had two failure modes that cost us in dogfood:
    #
    # 1. **Fire-and-forget under an active loop.** If ``close`` was called
    #    from inside a running event loop (notebook / Tauri backend /
    #    ``asyncio.run(_entry())`` during teardown but before the loop
    #    closed), it did ``loop.create_task(stop_all())`` and returned —
    #    the task was then orphaned and a hanging MCP server kept its
    #    subprocess alive past agent teardown.
    # 2. **Swallow-all ``except Exception``** made every path look like a
    #    success in journal; operators couldn't tell a clean shutdown
    #    apart from a swallowed RuntimeError.
    #
    # New contract:
    #   * :meth:`aclose` is the canonical async entry. ``stop_all`` runs
    #     under :func:`asyncio.wait_for`; on timeout the coroutine is
    #     cancelled, ``servers_hanging`` is computed from
    #     ``manager.status()`` (whoever's still ``connected``), and a
    #     ``mcp_close_timeout`` journal event fires. Unexpected errors
    #     emit ``mcp_close_error``; the happy path emits ``mcp_stopped``.
    #   * :meth:`close` is the sync SDK/CLI wrapper. No active loop →
    #     ``asyncio.run(self.aclose(...))``. Active loop → :class:`RuntimeError`
    #     so the caller is forced onto the async path. Fire-and-forget is
    #     gone.
    def _teardown_local_tasks(self) -> None:
        """Cancel subagent tasks + kill lingering shell subprocesses.

        Split out of ``close`` / ``aclose`` so both paths share the same
        local-cleanup sequence. ``.cancel()`` is a request, not a join —
        we don't await here; the event loop will deliver the
        ``CancelledError`` next time each task is scheduled, at which
        point run_task flips the record to cancelled.
        """
        for task_id, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
            self._running_tasks.pop(task_id, None)
        # Shell tasks own a real subprocess — cancelling the watcher
        # asyncio.Task above causes it to send SIGTERM→SIGKILL in its
        # finally. Belt-and-braces: also SIGKILL any lingering handles
        # here in case the watcher task already completed but the
        # subprocess is somehow still alive (shouldn't happen, but close
        # is our last chance to not leave zombies).
        for task_id, proc in list(self._running_shells.items()):
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, Exception):
                    proc.kill()
            self._running_shells.pop(task_id, None)

    def _connected_server_names(self) -> list[str]:
        """Best-effort snapshot of servers still in ``connected`` state.

        Used to populate ``servers_hanging`` on the timeout journal
        event. ``status()`` is pure-sync and defensively written never to
        raise — if a half-torn-down manager misbehaves we degrade to
        ``[]`` rather than poisoning the shutdown path.
        """
        mgr = self._mcp_manager
        if mgr is None:
            return []
        try:
            entries = mgr.status()
        except Exception:  # noqa: BLE001
            return []
        return [e.name for e in entries if getattr(e, "state", None) == "connected"]

    async def aclose(self, *, mcp_timeout: float = 5.0) -> None:
        """Async, timeout-bounded teardown (B3).

        Contract:
        - Cancels in-flight subagent tasks + kills lingering shell
          subprocesses (same as the old sync ``close``).
        - Runs ``MCPManager.stop_all`` under ``asyncio.wait_for``. On
          timeout, the coroutine is cancelled and a ``mcp_close_timeout``
          event fires with ``{session, elapsed_sec, timeout_sec,
          servers_hanging}``.
        - Unexpected exceptions during ``stop_all`` are captured into a
          ``mcp_close_error`` event (shutdown is best-effort; a thrown
          exception must not crash the caller).
        - Normal completion emits ``mcp_stopped`` with ``elapsed_sec``.
        - ``self._mcp_manager`` is set to ``None`` on every branch so a
          subsequent ``aclose()`` / ``close()`` is an idempotent no-op.
        - Finally, ``self._storage.close()`` to flush SQLite.

        F-04-014: fires ``Stop(reason="user_exit")`` BEFORE the
        teardown so the hook sees a live state / mode / model. Hook
        exceptions are suppressed — a broken stop hook MUST NOT block
        agent shutdown.
        """
        with contextlib.suppress(Exception):
            await self.fire_stop(reason="user_exit")
        self._teardown_local_tasks()

        # Claude-code parity (gh-32730): rm -rf every team this session
        # created that wasn't explicitly /team delete'd, so an orphan
        # team dir doesn't accumulate forever. Best-effort — failures
        # journal but don't block the rest of the teardown.
        # CRITICAL: only the LEADER fires cleanup. Teammates (where
        # ``_team_member_name`` is set) inherit ``_team`` from the
        # leader's manager via ``join_team`` — calling cleanup from
        # a teammate's aclose would cancel sibling teammates'
        # runtimes mid-flight (the leader's set is shared).
        if self._team is not None and self._team_member_name is None:
            cleanup = getattr(self._team, "cleanup_session_teams", None)
            if callable(cleanup):
                with contextlib.suppress(Exception):
                    await cleanup()

        if self._mcp_manager is not None:
            mgr = self._mcp_manager
            servers_hanging = self._connected_server_names()
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            try:
                await asyncio.wait_for(mgr.stop_all(), timeout=mcp_timeout)
            except TimeoutError:
                elapsed = loop.time() - t0
                journal.write(
                    "mcp_close_timeout",
                    session=self._session_id,
                    elapsed_sec=elapsed,
                    timeout_sec=mcp_timeout,
                    servers_hanging=servers_hanging,
                )
            except Exception as exc:  # noqa: BLE001
                journal.write(
                    "mcp_close_error",
                    session=self._session_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                journal.write(
                    "mcp_stopped",
                    session=self._session_id,
                    elapsed_sec=loop.time() - t0,
                )
            finally:
                self._mcp_manager = None

        self._storage.close()

    def close(self, *, mcp_timeout: float = 5.0) -> None:
        """Sync teardown — thin wrapper around :meth:`aclose`.

        Primary entry for no-loop callers (the CLI's outer ``finally``
        after ``asyncio.run(_entry())`` has returned, legacy SDK users
        who never opened a loop, sync unit tests). When called without a
        running loop we spin one via ``asyncio.run(self.aclose(...))``.

        Inside a running event loop there are two cases:

        * **No MCP manager to tear down** (and no storage-hostile state) —
          we do the pure-sync cleanup in-place (``_teardown_local_tasks``
          + ``_storage.close()``). This keeps the historical contract for
          async unit tests that build a bare Agent and call ``close()``.
        * **MCP manager is live** — we refuse. The pre-B3 path was
          ``loop.create_task(stop_all())`` fire-and-forget, which leaked
          tasks and kept MCP subprocesses alive past exit. Async callers
          must explicitly ``await agent.aclose(...)`` to get the
          timeout-bounded shutdown contract.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No active loop — safe to run our own.
            asyncio.run(self.aclose(mcp_timeout=mcp_timeout))
            return
        # Active loop. If nothing needs the async teardown, do the sync
        # subset in-place — safe, matches pre-B3 behaviour for bare
        # agents. If an MCP manager IS live, refuse: the caller must
        # await aclose() to honour the timeout + cancel-on-timeout
        # contract.
        if self._mcp_manager is not None:
            raise RuntimeError(
                "Agent.close() called inside a running event loop with a "
                "live MCP manager. Use `await agent.aclose(mcp_timeout=...)` "
                "instead — the old fire-and-forget close path was removed "
                "in v0.11 (B3)."
            )
        self._teardown_local_tasks()
        self._storage.close()

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        await self.aclose()


def build_agent(
    config: AuraConfig,
    *,
    hooks: HookChain | None = None,
    available_tools: dict[str, BaseTool] | None = None,
    session_id: str = _DEFAULT_SESSION,
    session_rules: SessionRuleSet | None = None,
    question_asker: QuestionAsker | None = None,
    mode: str = "default",
    disable_bypass: bool = False,
    ruleset: RuleSet | None = None,
    safety: SafetyPolicy | None = None,
) -> Agent:
    # 生产便利工厂：自动解析 model + storage；Agent 构造器保持 DI 注入以便测试替换。
    provider, model_name = llm.resolve(config.router["default"], cfg=config)
    model = llm.create(provider, model_name)
    storage = SessionStorage(config.resolved_storage_path())
    return Agent(
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
        safety=safety,
    )
