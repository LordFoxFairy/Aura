"""Agent facade — config + model + storage + hooks 组装成一条对话的入口层。"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
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
from aura.core.memory.context import Context
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
    ) -> None:
        # ``session_rules``: CLI hands in the same SessionRuleSet that was used
        # to build the permission hook; Agent.clear_session drops its runtime
        # rules alongside history and state so /clear is coherent.
        self._config = config
        self._model = model
        self._storage = storage
        self._hooks = hooks or HookChain()
        self._state = LoopState()
        self._session_rules = session_rules
        # Permission mode — the CLI resolves the effective mode (config +
        # --bypass-permissions flag) and hands it in. Stored here so the
        # status bar can surface it without reaching back into the store
        # each render. Valid values: "default" / "accept_edits" / "plan" /
        # "bypass"; enforcement still happens in the permission hook.
        self._mode = mode
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
        self._subagent_factory = SubagentFactory(
            parent_config=self._config,
            parent_model_spec=self._config.router.get("default", ""),
            parent_skills=self._skill_registry,
        )
        # Map: task_id -> the detached asyncio.Task handle. Shared with the
        # ``task_create`` tool so Agent.close() can cancel still-running
        # subagents without reaching back into the tool's internals.
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
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
                )
            elif name == "task_output" or name == "task_get" or name == "task_list":
                self._available_tools[name] = cls(store=self._tasks_store)
            elif name == "task_stop":
                self._available_tools[name] = cls(
                    store=self._tasks_store,
                    running=self._running_tasks,
                )
            elif name == "web_search":
                # web_search takes an optional WebSearchConfig; when the user
                # did not declare ``web_search:`` in config, the tool falls
                # back to its own defaults (DuckDuckGo, max_results=5).
                self._available_tools[name] = cls(config=self._config.web_search)
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
        self._system_prompt = build_system_prompt()
        self._primary_memory = project_memory.load_project_memory(self._cwd)
        self._rules = rules.load_rules(self._cwd)
        self._context = self._build_context()
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

    async def astream(self, prompt: str) -> AsyncIterator[AgentEvent]:
        # 事务性：history 只在 turn 正常完成才 save（else 分支）。
        #   CancelledError → yield Final + re-raise → 跳过 else → 下次从 pre-turn 状态恢复
        # max_turns 由 AgentLoop 直接 yield Final(reason="max_turns") 表示，走正常 save 路径。
        # 保证：存储里永远不会出现半截 turn（AI tool_call 缺对应 tool result 等）。
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
            # Reactive recompact: if the model signals a context-length
            # overflow on this turn, run compact(source="reactive") and
            # retry the turn ONCE against the compacted history. Only the
            # first failure triggers the recovery; a second overflow on the
            # retry is re-raised so persistent-overflow bugs (e.g. system
            # prompt already too long) surface instead of looping forever.
            retry_exc: BaseException | None = None
            try:
                async for event in self._loop.run_turn(user_prompt=prompt, history=history):
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
                # the retry must see the new messages.
                history = self._storage.load(self._session_id)
                async for event in self._loop.run_turn(user_prompt=prompt, history=history):
                    yield event

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
        journal.write("model_switch_attempt", spec=spec)
        provider, model_name = llm.resolve(spec, cfg=self._config)
        self._model = llm.create(provider, model_name)
        self._loop = self._build_loop()
        journal.write(
            "model_switched",
            spec=spec,
            provider=provider.name,
            model=model_name,
        )

    def clear_session(self) -> None:
        self._storage.clear(self._session_id)
        self._state.reset()
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
    def current_model(self) -> str:
        """当前 default 对应的 'provider:model' 字符串。"""
        return self._config.router.get("default", "")

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
        """
        valid = {"default", "accept_edits", "plan", "bypass"}
        if mode not in valid:
            raise ValueError(
                f"invalid mode {mode!r}; expected one of {sorted(valid)}"
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
        )

    def _build_context(self) -> Context:
        return Context(
            cwd=self._cwd,
            system_prompt=self._system_prompt,
            primary_memory=self._primary_memory,
            rules=self._rules,
            skills=self._skill_registry.list(),
            todos_provider=lambda: self._state.custom.get("todos", []),
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
        self._loop._rebind_tools(self._registry.tools())
        journal.write(
            "mcp_aconnect_done",
            tool_count=len(tools),
            command_count=len(commands),
        )

    def close(self) -> None:
        # Cancel any still-running subagent tasks first so they don't keep
        # scribbling into a half-torn-down agent. ``cancel()`` is a request,
        # not a join — we don't await here (close is sync), but the event
        # loop will deliver the CancelledError next time the task is
        # scheduled, at which point run_task flips the record to cancelled.
        for task_id, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
            self._running_tasks.pop(task_id, None)
        # stop_all() is async; Agent.close() is sync (legacy SDK API).
        # Run a small event loop if none is running; otherwise schedule +
        # best-effort-detach. This mirrors the storage close path which is
        # pure-sync.
        if self._mcp_manager is not None:
            import asyncio as _asyncio

            try:
                loop = _asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            try:
                if loop is None:
                    _asyncio.run(self._mcp_manager.stop_all())
                else:
                    loop.create_task(self._mcp_manager.stop_all())
            except Exception as exc:  # noqa: BLE001
                journal.write(
                    "mcp_stop_all_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._mcp_manager = None
        self._storage.close()

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self.close()


def build_agent(
    config: AuraConfig,
    *,
    hooks: HookChain | None = None,
    available_tools: dict[str, BaseTool] | None = None,
    session_id: str = _DEFAULT_SESSION,
    session_rules: SessionRuleSet | None = None,
    question_asker: QuestionAsker | None = None,
    mode: str = "default",
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
    )
