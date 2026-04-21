"""Agent facade — config + model + storage + hooks 组装成一条对话的入口层。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from aura.config.schema import AuraConfig, AuraConfigError
from aura.core import llm
from aura.core.hooks import HookChain
from aura.core.hooks.budget import MaxTurnsExceeded, default_hooks
from aura.core.hooks.must_read_first import make_must_read_first_hook
from aura.core.loop import AgentLoop
from aura.core.memory import project_memory, rules
from aura.core.memory.context import Context
from aura.core.memory.system_prompt import build_system_prompt
from aura.core.permissions.session import SessionRuleSet
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.registry import ToolRegistry
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
        self._context = self._build_context()
        # Tool-intrinsic invariant (matches claude-code FileEditTool): edit_file
        # rejects before any user-supplied gate would run. Appended AFTER the
        # caller's hooks so permission (CLI-installed) runs first — if the user
        # denies the tool, we don't also yell about the missing read. Tracked as
        # a field so clear_session can swap it when Context is rebuilt.
        self._must_read_first_hook = make_must_read_first_hook(self._context)
        self._hooks.pre_tool.append(self._must_read_first_hook)
        self._loop = self._build_loop()

    async def astream(self, prompt: str) -> AsyncIterator[AgentEvent]:
        # 事务性：history 只在 turn 正常完成才 save（else 分支）。
        #   CancelledError → yield Final + re-raise → 跳过 else → 下次从 pre-turn 状态恢复
        #   MaxTurnsExceeded → yield Final + return → 同样跳过 save
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
        except MaxTurnsExceeded as exc:
            journal.write(
                "astream_max_turns",
                session=self._session_id,
                detail=str(exc),
            )
            yield Final(message=f"({exc})")
            return
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
        self._loop = self._build_loop()
        journal.write("session_cleared", session=self._session_id)

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
            todos_provider=lambda: self._state.custom.get("todos", []),
        )

    def close(self) -> None:
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
