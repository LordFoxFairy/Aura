"""Agent facade — config + model + storage + hooks 组装成一条对话的入口层。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from aura.config.schema import AuraConfig, AuraConfigError
from aura.core import llm
from aura.core.compact import CompactResult, run_compact
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
from aura.core.skills import Skill, load_skills
from aura.schemas.events import AgentEvent, Final
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolError
from aura.tools import BUILTIN_STATEFUL_TOOLS, BUILTIN_TOOLS
from aura.tools.ask_user import QuestionAsker

_DEFAULT_SESSION = "default"


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
            else:  # pragma: no cover — guardrail for future additions
                raise RuntimeError(f"unwired stateful tool: {name}")
        self._session_id = session_id
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
        self._cwd = Path.cwd()
        self._system_prompt = build_system_prompt()
        self._primary_memory = project_memory.load_project_memory(self._cwd)
        self._rules = rules.load_rules(self._cwd)
        # Skills: user-layer (~/.aura/skills/) + project-layer (<cwd>/.aura/skills/).
        # Loaded once at Agent init; not re-scanned on /clear (v0.2.0 MVP — no
        # hot reload). Collision resolution inside the loader logs to journal.
        self._skill_registry = load_skills(cwd=self._cwd)
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

    async def astream(self, prompt: str) -> AsyncIterator[AgentEvent]:
        # 事务性：history 只在 turn 正常完成才 save（else 分支）。
        #   CancelledError → yield Final + re-raise → 跳过 else → 下次从 pre-turn 状态恢复
        # max_turns 由 AgentLoop 直接 yield Final(reason="max_turns") 表示，走正常 save 路径。
        # 保证：存储里永远不会出现半截 turn（AI tool_call 缺对应 tool result 等）。
        journal.write(
            "astream_begin",
            session=self._session_id,
            prompt_preview=prompt[:200],
        )
        history = self._storage.load(self._session_id)
        try:
            async for event in self._loop.run_turn(user_prompt=prompt, history=history):
                yield event
        except asyncio.CancelledError:
            journal.write("astream_cancelled", session=self._session_id)
            yield Final(message="(cancelled)")
            raise
        else:
            self._storage.save(self._session_id, history)
            journal.write(
                "astream_end",
                session=self._session_id,
                history_len=len(history),
                total_tokens=self._state.total_tokens_used,
            )

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
        self, *, source: Literal["manual", "auto"] = "manual",
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
    )
