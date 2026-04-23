"""Agent facade — config + model + storage + hooks 组装成一条对话的入口层。"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from aura.config.schema import AuraConfig, AuraConfigError
from aura.core import llm
from aura.core.compact import CompactResult, run_compact
from aura.core.compact.constants import AUTO_COMPACT_THRESHOLD
from aura.core.hooks import HookChain
from aura.core.hooks.bash_safety import make_bash_safety_hook
from aura.core.hooks.budget import default_hooks
from aura.core.hooks.must_read_first import make_must_read_first_hook
from aura.core.loop import AgentLoop
from aura.core.mcp import MCPManager
from aura.core.memory import project_memory, rules
from aura.core.memory.context import Context, _ReadRecord
from aura.core.memory.system_prompt import build_system_prompt
from aura.core.permissions.session import SessionRuleSet
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.registry import ToolRegistry
from aura.core.skills import Skill, SkillRegistry, load_skills
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.store import TasksStore
from aura.schemas.events import AgentEvent, Final
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolError
from aura.tools import BUILTIN_STATEFUL_TOOLS, BUILTIN_TOOLS
from aura.tools.ask_user import QuestionAsker

_DEFAULT_SESSION = "default"

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
        session_log_dir: Path | None = None,
        pre_loaded_skills: SkillRegistry | None = None,
        mode: str = "default",
        system_prompt_suffix: str = "",
        disable_bypass: bool = False,
        inherited_reads: Mapping[Path, _ReadRecord] | None = None,
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
        self._subagent_factory = SubagentFactory(
            parent_config=self._config,
            parent_model_spec=self._config.router.get("default", ""),
            parent_skills=self._skill_registry,
            parent_read_records_provider=lambda: self._context._read_records,
        )
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
                # ``parent_hooks=self._hooks`` threads the parent's live
                # HookChain into run_task so post_subagent fires on each
                # terminal transition. Passing the bound chain (not a
                # copy) keeps the wire-up in sync with runtime hook
                # installs (e.g. clear_session swap of must_read_first).
                self._available_tools[name] = cls(
                    store=self._tasks_store,
                    factory=self._subagent_factory,
                    running=self._running_tasks,
                    parent_hooks=self._hooks,
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
                )
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
        self._bash_safety_hook = make_bash_safety_hook()
        self._hooks.pre_tool.insert(0, self._bash_safety_hook)
        # Tool-intrinsic invariant (matches claude-code FileEditTool): edit_file
        # rejects before any user-supplied gate would run. Appended AFTER the
        # caller's hooks so permission (CLI-installed) runs first — if the user
        # denies the tool, we don't also yell about the missing read. Tracked as
        # a field so clear_session can swap it when Context is rebuilt.
        self._must_read_first_hook = make_must_read_first_hook(self._context)
        self._hooks.pre_tool.append(self._must_read_first_hook)
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

    async def astream(
        self,
        prompt: str,
        *,
        attachments: list[HumanMessage] | None = None,
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
            history = self._storage.load(self._session_id)
            # pre_user_prompt fires at the very entry — BEFORE attachments +
            # user HumanMessage land in history — so hooks see the prompt
            # exactly as the user typed it. Observational only; buggy hooks
            # get journaled + swallowed so a broken observer can't block
            # user turns.
            if self._hooks.pre_user_prompt:
                try:
                    await self._hooks.run_pre_user_prompt(
                        prompt=prompt, state=self._state,
                    )
                except Exception as exc:  # noqa: BLE001
                    journal.write(
                        "pre_user_prompt_hook_failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
            # Attachments land BEFORE the user's HumanMessage so the model
            # sees the injected context first (matching how claude-code
            # prepends ``<attachment>`` blocks ahead of the user turn).
            if attachments:
                history.extend(attachments)
            history.append(HumanMessage(content=prompt))
            # Save BEFORE the first model call — crash / kill / Ctrl-C after
            # this point still leaves the user turn on disk for resume.
            self._storage.save(self._session_id, history)

            # Reactive recompact: if the model signals a context-length
            # overflow on this turn, run compact(source="reactive") and
            # retry the turn ONCE against the compacted history. Only the
            # first failure triggers the recovery; a second overflow on the
            # retry is re-raised so persistent-overflow bugs (e.g. system
            # prompt already too long) surface instead of looping forever.
            retry_exc: BaseException | None = None
            try:
                async for event in self._loop.run_turn(history=history):
                    yield event
            except asyncio.CancelledError:
                journal.write("astream_cancelled", session=self._session_id)
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
                # Stash and retry OUTSIDE the except — avoids nested-exception
                # chaining if compact or the retry turn itself raises.
                retry_exc = exc

            if retry_exc is not None:
                await self.compact(source="reactive")
                # compact replaced history with [summary, *recent_files, *tail];
                # the retry reads the already-persisted history (which contains
                # the user turn + attachments from the pre-invoke save above).
                # No need to re-pass attachments — they're durably in history.
                history = self._storage.load(self._session_id)
                async for event in self._loop.run_turn(history=history):
                    yield event

            # Second save: captures tool results + assistant message that
            # landed during run_turn. Not guarded by a success branch any
            # more — the pre-invoke save already gave us the durability floor;
            # this save is additive (final full state). Cancellation still
            # skips this line via the raise above, which is fine — the user
            # turn is already on disk from the pre-invoke save.
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
            # stream. Zero threshold disables; any positive value arms it.
            if (
                self._auto_compact_threshold > 0
                and self._state.total_tokens_used > self._auto_compact_threshold
            ):
                journal.write(
                    "auto_compact_triggered",
                    session=self._session_id,
                    tokens=self._state.total_tokens_used,
                    threshold=self._auto_compact_threshold,
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
        self._storage.clear(self._session_id)
        self._state.reset()
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
        self._bash_safety_hook = make_bash_safety_hook()
        self._hooks.pre_tool.insert(0, self._bash_safety_hook)
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

    def _build_loop(self) -> AgentLoop:
        return AgentLoop(
            model=self._model,
            registry=self._registry,
            context=self._context,
            hooks=self._hooks,
            state=self._state,
            retry_config=self._config.retry,
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
            inherited_reads=inherited_reads,
        )

    async def bootstrap(self) -> None:
        """Fire the ``pre_session`` hook chain — session-start signal.

        The sibling to :meth:`shutdown`. Kept separate from ``__init__``
        (which is sync) because hooks are async by contract and we don't
        want to spin up a throwaway event loop just to fire a best-effort
        signal. Cleanest separation: callers that want pre_session
        delivered call ``bootstrap()`` after construction, mirroring the
        existing ``aconnect()`` invocation in the CLI entry point.

        Exceptions raised inside a hook are journaled and swallowed —
        these are observational signals; a buggy hook must NOT abort the
        session it's supposed to be announcing. No-op when no
        pre_session hooks are registered.
        """
        if not self._hooks.pre_session:
            return
        try:
            await self._hooks.run_pre_session(
                session_id=self._session_id, cwd=self._cwd,
            )
        except Exception as exc:  # noqa: BLE001
            journal.write(
                "pre_session_hook_failed",
                session=self._session_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def shutdown(self, *, mcp_timeout: float = 5.0) -> None:
        """Fire ``post_session`` then tear down asynchronously.

        Symmetric with :meth:`bootstrap`. The sync :meth:`close` path
        stays available for legacy SDK callers that never opened a loop,
        and deliberately does NOT fire post_session: firing an async
        hook from a sync context would either block the event loop or
        schedule a best-effort task that never gets awaited. Cleaner
        contract: async consumers call ``shutdown()`` (which delegates
        to :meth:`aclose`, preserving the timeout-bounded MCP teardown
        from B3); sync consumers call :meth:`close` and miss the
        post_session signal — no surprise semantics either way.
        """
        if self._hooks.post_session:
            try:
                await self._hooks.run_post_session(
                    session_id=self._session_id, cwd=self._cwd,
                )
            except Exception as exc:  # noqa: BLE001
                journal.write(
                    "post_session_hook_failed",
                    session=self._session_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
        # Must NOT call self.close() here — that's the sync entry point,
        # which refuses to run inside an active event loop since B3 (no
        # fire-and-forget). shutdown() is always invoked under a running
        # loop (REPL context), so go directly through aclose.
        await self.aclose(mcp_timeout=mcp_timeout)

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
        """
        self._teardown_local_tasks()

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
    )
