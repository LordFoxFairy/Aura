"""Agent facade — config + model + storage + hooks 组装成一条对话的入口层。"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
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
from aura.core.compact.constants import (
    AUTO_COMPACT_THRESHOLD,
    auto_compact_threshold_for,
)
from aura.core.compact.legacy_adapter import LegacyCompactor
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
from aura.core.permissions.denials import PermissionDenial
from aura.core.permissions.mode import Mode
from aura.core.permissions.safety import SafetyPolicy
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.registry import ToolRegistry
from aura.core.runtime.session import SessionRuntime
from aura.core.runtime.tool_factory import (
    STATEFUL_TOOL_FACTORIES,
    ToolRuntime,
)
from aura.core.skills import Skill, SkillRegistry, load_skills
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.store import TasksStore
from aura.core.tokens import estimate_message_tokens, estimate_text_tokens
from aura.schemas.events import AgentEvent, AssistantDelta, Final
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolError
from aura.tools import BUILTIN_STATEFUL_TOOLS, BUILTIN_TOOLS
from aura.tools.ask_user import QuestionAsker

# Substring signatures that identify provider-level "context length exceeded"
# errors. We match on stringified message — not exception type — so we don't
# have to import openai / anthropic SDK types and so custom/wrapper models
# keep working. Lowercased at compare time; covers OpenAI, Anthropic,
# Google, Aliyun DashScope, Ollama variants. Add new entries when a
# provider surfaces a novel phrasing — bug-watch the journal for
# ``llm_invoke_failed`` events whose error doesn't trigger reactive compact.
_CONTEXT_OVERFLOW_PHRASES: tuple[str, ...] = (
    # OpenAI / OpenAI-compatible.
    "context length",
    "context_length_exceeded",
    "maximum context",
    # Anthropic.
    "prompt is too long",
    # Aliyun DashScope (the OpenAI-compat endpoint surfaces this exact
    # phrasing for both 1261 and the bare-text variants Qwen returns).
    "prompt exceeds max length",
    "input too long",
    "exceeds max length",
    # Google Gemini.
    "request payload size exceeds",
    # Generic — last-resort matchers.
    "too many tokens",
    "max_tokens exceeded",
)

# Aliyun DashScope error codes that map to "context too long". Code-level
# detection complements phrase matching: even when the SDK localizes the
# message into another language, the structured code field stays stable.
_CONTEXT_OVERFLOW_CODES: tuple[str, ...] = (
    "1261",  # DashScope: "Prompt exceeds max length"
)


def _is_context_overflow(exc: BaseException) -> bool:
    """True iff ``exc``'s message matches a known context-overflow signature.

    Checks both the lowercased error text (cheap substring match against
    :data:`_CONTEXT_OVERFLOW_PHRASES`) and the structured error code
    field (``'code': '1261'`` style — DashScope and similar surfaces).
    """
    msg = str(exc).lower()
    if any(phrase in msg for phrase in _CONTEXT_OVERFLOW_PHRASES):
        return True
    # Fallback: structured code in the rendered exception. DashScope's
    # OpenAI-compat client renders ``Error code: 400 - {'error':
    # {'code': '1261', ...}}`` so the literal ``'code': '1261'`` substring
    # is stable across SDK locales.
    for code in _CONTEXT_OVERFLOW_CODES:
        if f"'code': '{code}'" in msg or f'"code": "{code}"' in msg:
            return True
    return False


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
        deny_ruleset: RuleSet | None = None,
        ask_ruleset: RuleSet | None = None,
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
        # Phase 1 Task 13: lifecycle / persistence / streaming-buffer state
        # lives on a peer SessionRuntime so Agent stays focused on the loop
        # / model / hooks wiring. The runtime owns: session_id, storage,
        # session_log_path, session_rules snapshot, partial-assistant
        # buffer, SessionStart re-arm flag, pending notifications queue,
        # and the inherited_reads carry-over for subagents. ``Agent``
        # forwards user-facing methods (clear_session, aclose,
        # resume_session) so the public API is unchanged.
        self._session_runtime = SessionRuntime(
            storage=storage,
            session_id=session_id,
            session_log_dir=session_log_dir,
            session_rules=session_rules,
            inherited_reads=inherited_reads,
        )
        self._hooks = hooks or HookChain()
        self._state = LoopState()
        # G5 / Phase 1 Task 4: per-turn deny records live on the typed
        # ``state.slots.turn_denials`` slot. The permission + bash safety
        # hooks append to that list, ``Loop.run_turn`` clears it at the
        # start of every astream call, and ``last_turn_denials()`` reads
        # it back through ``self._state`` (Loop and Agent share the same
        # ``LoopState``, so no Agent-side alias is needed).
        # F-0910-002: auto-compact circuit breaker — three consecutive failed
        # auto-compact attempts disable subsequent auto-firings for this
        # session. Manual ``/compact`` bypasses this counter (different code
        # path), and a successful auto-compact resets it to 0. Lives on
        # the typed ``state.slots.consecutive_compact_failures`` slot
        # (Phase 1 Task 6). The default LoopSlots() already has this at
        # 0, so no explicit seed is needed — kept as an explicit
        # ``replace`` for parity with the old reset-on-construct semantics
        # in case a future refactor reuses an existing LoopSlots.
        self._state.slots = dataclasses.replace(
            self._state.slots, consecutive_compact_failures=0,
        )
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
            # F-0910-011 — bundled skills (verify / simplify / code-review)
            # ship with Aura and load alongside user + project layers.
            self._skill_registry = load_skills(
                cwd=self._cwd, include_bundled=True,
            )
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
            parent_deny_rules=deny_ruleset,
            parent_ask_rules=ask_ruleset,
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
        # F-05-003 partial-text buffer, F-04-014 SessionStart re-arm
        # flag, and Round 4F notification queue all live on
        # :class:`SessionRuntime` (Phase 1 Task 13). Agent property
        # forwards keep the historical attribute names so external
        # callers (tests, commands, streaming renderers) see no API
        # break.

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
        # Phase 2 Task 6: factory-driven wiring for the 7 stateful tools
        # whose deps live on :class:`ToolRuntime`. The 6 remaining
        # tools (task_output, web_search, enter_plan_mode, exit_plan_mode,
        # bash_background, skill) close over ``self``-bound methods and
        # stay wired inline below.
        tool_runtime = ToolRuntime(
            state=self._state,
            asker=question_asker or _unavailable_question_asker,
            tasks_store=self._tasks_store,
            subagent_factory=self._subagent_factory,
            running_tasks=self._running_tasks,
            running_shells=self._running_shells,
            transcript_storage=self._storage,
            agent=self,
        )
        for factory in STATEFUL_TOOL_FACTORIES:
            self._available_tools[factory.name] = factory.build(tool_runtime)
        # Residual stateful tools — Agent-method closures only.
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
            session_id_provider=(
                lambda: getattr(self, "_session_id", _DEFAULT_SESSION)
            ),
            session_rules_provider=lambda: self._session_rules,
            loop_state_provider=lambda: self._state,
        )
        # ``session_id``, ``session_log_path`` (per-session JSONL routing
        # for ``journal.session_scope``), and ``session_rules`` all live
        # on :class:`SessionRuntime`. Property forwards below preserve
        # the historical ``self._session_id`` / ``self._session_log_path``
        # / ``self._session_rules`` access patterns used by tests + the
        # commands layer.
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
        # ``inherited_reads`` (Workstream G8) only flows into the FIRST
        # Context construction — /clear and /compact build their own fresh
        # Contexts and must NOT resurrect a long-gone parent's read
        # fingerprints, so we do NOT store this on self. Subagent spawn
        # re-snapshots the parent at each ``SubagentFactory.spawn`` call.
        self._context = self._build_context(inherited_reads=inherited_reads)
        # Bash safety — Tier A shell attacks (zsh builtins, CR
        # parser differential, malformed+separator, cd+git compound). Inserted
        # at pre_tool[0] so it precedes any caller-supplied permission hook —
        # safety is a separate axis from permission and cannot be overridden
        # by allow/deny/ask rules. Bypass mode intentionally skips this hook,
        # matching the product contract that bypass is an operator opt-in to
        # run commands without policy prompts. Stateless; tracked as a field
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
            finally:
                self._current_abort = None
            # F-01-012 — reactive recompact lives INSIDE AgentLoop now
            # (turn_count + per-turn audit state survive the recompact).
            # The Agent simply hands a callback into the loop via
            # ``_build_loop``; on context-overflow, the loop calls back,
            # we compact + reload ``history`` in place, and the SAME
            # turn retries. No outer try/retry block here.
            self._storage.save(self._session_id, history)
            journal.write(
                "astream_end",
                session=self._session_id,
                history_len=len(history),
                total_tokens=self._state.total_tokens_used,
            )
            # Auto-compact post-turn. Deliberately AFTER save + astream_end
            # so the summary turn sees a stable, already-persisted history
            # and we don't interleave compact I/O with the caller's yield
            # stream. Phase 1 §3.3: routed through the Compactor Protocol
            # so threshold check, circuit breaker, and run_compact all live
            # on one named call site. The adapter mirrors the pre-Phase-1
            # journal events exactly; behavior is unchanged.
            await self._compactor.auto(
                history,
                self._state.slots,
                model=self._current_model_spec,
            )

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

    # ------------------------------------------------------------------
    # Phase 1 Task 13 — SessionRuntime forwards
    #
    # The session lifecycle / persistence / streaming-buffer state lives
    # on :class:`SessionRuntime` (see ``aura/core/runtime/session.py``).
    # These property forwards preserve the historical attribute names
    # (``self._storage`` / ``self._session_id`` / etc.) so the wide tail
    # of internal call sites — tests, command handlers, compact, hooks,
    # auto_reload — keeps working without a sweep.
    # ------------------------------------------------------------------

    @property
    def _storage(self) -> SessionStorage:
        return self._session_runtime.storage

    @property
    def _session_id(self) -> str:
        return self._session_runtime.session_id

    @_session_id.setter
    def _session_id(self, value: str) -> None:
        # ``resume_session`` is the only legitimate writer (the runtime's
        # :meth:`resume` flips it as part of the resume contract).
        # Direct re-assignment is preserved for parity with the pre-Phase-1
        # surface in case a test fixture flips it manually.
        self._session_runtime._session_id = value

    @property
    def _session_log_path(self) -> Path | None:
        return self._session_runtime.session_log_path

    @property
    def _session_rules(self) -> SessionRuleSet | None:
        return self._session_runtime.session_rules

    @property
    def _partial_assistant_text(self) -> str:
        return self._session_runtime.partial_assistant_text

    @_partial_assistant_text.setter
    def _partial_assistant_text(self, value: str) -> None:
        # Used by astream's reset (``= ""``) and the abort flush path.
        # The runtime's :meth:`reset_partial_assistant_text` handles the
        # empty-string case explicitly; for any other rebind we go through
        # the underlying field so the ``+=`` accumulator path still works
        # via the property descriptor.
        self._session_runtime._partial_assistant_text = value

    @property
    def _session_start_fired(self) -> bool:
        return self._session_runtime.session_start_fired

    @_session_start_fired.setter
    def _session_start_fired(self, value: bool) -> None:
        self._session_runtime._session_start_fired = value

    @property
    def _pending_notifications(self) -> list[TaskNotification]:
        # Returning the live list is intentional — call sites use
        # ``.append`` and ``.clear`` directly, and the runtime IS the
        # single source of truth for the queue.
        return self._session_runtime._pending_notifications

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
        # Phase 1 Task 13: lifecycle (storage.clear, session_rules drop,
        # buffers + queue + SessionStart re-arm) lives on the runtime.
        # Agent retains ownership of LoopState slot resets, hook chain
        # rewiring, memory/rules cache invalidation, and Context/Loop
        # rebuild — those need model + hook + skill wiring outside the
        # session lifecycle scope.
        self._session_runtime.clear()
        self._state.reset()
        # ``LoopState.reset`` only zeros the counters; slots live across
        # sessions for legitimate carry-over (token-stats etc.) and are
        # explicitly reset here per slot, by their owners. Clear the G5
        # denials list in place so ``Agent.last_turn_denials()`` returns
        # ``()`` immediately after /clear.
        self._state.slots.turn_denials.clear()
        # Clear the typed ``state.slots.todos`` list in place so /clear
        # starts the next session with no stale plan items.
        self._state.slots.todos.clear()
        # Per-slot resets — scratchpad state that must reset on /clear.
        # Mutable lists/dicts mutate in place; scalars + the buddy state
        # rebind via ``dataclasses.replace`` (LoopSlots is frozen at the
        # attribute level).
        from aura.schemas.state import BuddyState as _BuddyState
        self._state.slots.perm_dedup_cache.clear()
        self._state.slots.invoked_skills.clear()
        self._state.slots.preserved_invoked_skills.clear()
        self._state.slots.skill_restrict_leases.clear()
        self._state.slots = dataclasses.replace(
            self._state.slots,
            active_team=None,
            ask_pending=False,
            consecutive_compact_failures=0,
            buddy=_BuddyState(),
        )
        # Drop any captured prior mode — /clear starts a fresh session so
        # a leftover "accept_edits" from a previous plan cycle shouldn't
        # bleed into the next one.
        self._prior_mode = None
        # /clear 语义：同时 invalidate memory/rules caches + 重建 Context。
        # progressive 状态（nested fragments / matched rules）随新实例自然清空 ——
        # 不做原地 reset，避免遗漏字段。
        project_memory.clear_cache(self._cwd)
        rules.clear_cache(self._cwd)
        self._primary_memory = project_memory.load_project_memory(
            self._cwd,
            auto_memory_dir=self._storage.memory_dir(cwd=self._cwd),
        )
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
        self._loop = self._build_loop()
        journal.write("session_cleared", session=self._session_id)

    def _effective_auto_compact_threshold(self) -> int:
        """Resolve the live auto-compact threshold.

        F-0910-001: ``-1`` means "derive from the live model's context
        window". An explicit positive override (constructor kwarg) wins
        over the model-aware computation; ``0`` keeps disabling the
        feature entirely. Recomputed every call so ``switch_model`` is
        honored without re-wiring.
        """
        if self._auto_compact_threshold == -1:
            return auto_compact_threshold_for(self._current_model_spec)
        return self._auto_compact_threshold

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
        return self._session_runtime.pending_notifications

    def buffer_partial_assistant_text(self, text: str) -> None:
        """Append ``text`` to the partial-assistant buffer.

        Producer hook for streaming renderers / tests so an abort
        before the final AIMessage still surfaces partial reasoning.
        Reset on every new astream call.
        """
        self._session_runtime.buffer_partial_assistant_text(text)

    def _enqueue_task_notification(self, notif: TaskNotification) -> None:
        """External producer hook — append ``notif`` to the queue.

        Unbounded at the queue level — the build-time renderer caps the
        emitted block at 5 entries (FIFO) and collapses the tail to a
        ``(N more earlier)`` line, so the parent's prompt envelope stays
        compact while the queue itself preserves order.
        """
        self._session_runtime.enqueue_task_notification(notif)

    def _drain_task_notifications(self) -> list[TaskNotification]:
        """Pop every queued notification and return them, oldest first."""
        return self._session_runtime.drain_task_notifications()

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

        Raises ``RuntimeError`` when the teams feature gate
        (``teams.enabled``) is False — claude-code parity with
        ``isAgentSwarmsEnabled()``. The caller should set
        ``teams.enabled=true`` in ``.aura/config.json`` to opt in.
        """
        if not self._config.teams.enabled:
            raise RuntimeError(
                "teams disabled — set teams.enabled=true in "
                ".aura/config.json to enable the multi-agent swarm "
                "subsystem (mirrors claude-code's "
                "isAgentSwarmsEnabled() flag)"
            )
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
        on double-join). Also skips when the teams feature gate
        (``teams.enabled``) is False — claude-code parity with
        ``isAgentSwarmsEnabled()``: a disabled teams subsystem must
        not grow the LLM's tool schema with an inert tool.
        """
        if not self._config.teams.enabled:
            return
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

        Phase 1 Task 13 — the storage swap + buffer reset + log-path
        retarget + ``session_resumed`` journal event live on the
        :class:`SessionRuntime`. Agent retains LoopState reset (slot
        bookkeeping for the new session).
        """
        count = self._session_runtime.resume(session_id)
        self._state.reset()
        # Phase 1 Task 4: drop any captured denials so the resumed
        # session opens with an empty ``last_turn_denials()`` view
        # (parity with the pre-migration re-seed).
        self._state.slots.turn_denials.clear()
        return count

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
        on a runaway session.

        Counts the FULL prompt the model sees per turn, not just
        ``history``: pinned prefix (system + memory + skills + tool
        schemas) plus the live history. Prior versions only counted
        history, which underestimated by ~5–10k tokens on a real
        config and let DashScope sessions overflow before the trigger
        ever armed. The 4-chars-per-token approximation is consistent
        in-aggregate and cheap to compute.
        """
        history_tokens = sum(estimate_message_tokens(msg) for msg in history)
        return history_tokens + self._estimate_pinned_tokens()

    def _estimate_pinned_tokens(self) -> int:
        # Build the pinned messages with an empty history — everything
        # from Context.build that doesn't depend on live turn state.
        import json

        tokens = 0
        for message in self._context.build([]):
            content = getattr(message, "content", "")
            if isinstance(content, str):
                tokens += estimate_text_tokens(content)
        # Tool schemas go into every request as a separate payload the
        # provider also bills against the cached prefix. Approximate via
        # name + description + JSON-serialized args schema.
        for tool in self._registry.tools():
            tokens += estimate_text_tokens(tool.name or "")
            tokens += estimate_text_tokens(tool.description or "")
            try:
                schema = json.dumps(
                    getattr(tool, "args", {}) or {},
                    default=str,
                    ensure_ascii=False,
                )
            except (TypeError, ValueError):
                schema = ""
            tokens += estimate_text_tokens(schema)
        return tokens

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
        ``AttributeError``. The underlying list lives on
        ``self._state.slots.turn_denials`` (Phase 1 Task 4 migration);
        the snapshot tuple isolates the caller from a racy in-place
        grow between turns.

        Workstream G5 — ``docs/specs/2026-04-23-aura-main-channel-parity.md``.
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
        # Phase 1 §3.3: build a fresh LegacyCompactor every time so a
        # ``switch_model`` (which rebuilds the loop) gets a compactor
        # whose microcompact policy reflects the current Agent config.
        # Stored on ``self._compactor`` so the post-turn auto-compact
        # site in :meth:`astream` can reach it without going through
        # the loop.
        self._compactor = LegacyCompactor(
            self,
            microcompact_policy=policy,
            session_id=self._session_id,
            turn_provider=lambda: self._state.turn_count,
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
        """F-01-012 — compact + reload history in place for the loop.

        Called by ``AgentLoop._invoke_model`` when ``ainvoke`` raises a
        context-overflow. Compacts (which writes the summary back to
        storage), then refreshes the in-memory history list in place
        so the loop's local reference sees the new state without a
        re-assignment.
        """
        await self.compact(source="reactive")
        history[:] = self._storage.load(self._session_id)

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
            todos_provider=lambda: self._state.slots.todos,
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
        # F-02-031 — route the merge through ``assemble_tool_pool`` so
        # builtin-vs-MCP collisions resolve with builtin precedence + a
        # ``mcp_tool_shadowed`` journal event instead of a silent skip.
        from aura.core.registry import assemble_tool_pool  # noqa: PLC0415
        merged = assemble_tool_pool(self._registry.tools(), tools)
        # Replace registry contents with the merged pool. Clear-then-add
        # keeps the existing ToolRegistry instance + its callers (the
        # loop's tool binding, send_message register/unregister, etc.).
        for name in list(self._registry):
            self._registry.unregister(name)
        for t in merged.values():
            self._registry.register(t)
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

        # Phase 1 Task 13 — storage close lives on SessionRuntime so
        # other lifecycle exit points (future graceful-shutdown hooks)
        # can route through one named call.
        self._session_runtime.close_storage()

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
        self._session_runtime.close_storage()

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
    deny_ruleset: RuleSet | None = None,
    ask_ruleset: RuleSet | None = None,
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
        deny_ruleset=deny_ruleset,
        ask_ruleset=ask_ruleset,
        safety=safety,
    )
