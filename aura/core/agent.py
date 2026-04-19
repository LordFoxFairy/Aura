"""Agent facade — config + model + storage + hooks 组装成一条对话的入口层。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from aura.config.schema import AuraConfig, AuraConfigError
from aura.core.events import AgentEvent, Final
from aura.core.hooks import HookChain
from aura.core.hooks.budget import MaxTurnsExceeded, default_hooks
from aura.core.llm import ModelFactory
from aura.core.loop import AgentLoop
from aura.core.memory import project_memory, rules
from aura.core.memory.context import Context
from aura.core.memory.system_prompt import build_system_prompt
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.registry import ToolRegistry
from aura.core.state import LoopState
from aura.tools import BUILTIN_TOOLS
from aura.tools.todo_write import make_todo_write_tool

_DEFAULT_SESSION = "default"


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
    ) -> None:
        self._config = config
        self._model = model
        self._storage = storage
        self._hooks = hooks or HookChain()
        self._state = LoopState()
        self._available_tools = (
            dict(available_tools) if available_tools is not None else dict(BUILTIN_TOOLS)
        )
        # Stateful tools (factory-bound to `self._state`) must be merged in
        # BEFORE `_build_registry` resolves `config.tools.enabled`.
        self._available_tools["todo_write"] = make_todo_write_tool(self._state)
        self._session_id = session_id
        # Inline registry construction: config.tools.enabled → lookup → ToolRegistry.
        # 只在 __init__ 被构造一次；无需抽方法。
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
        provider, model_name = ModelFactory.resolve(spec, cfg=self._config)
        self._model, _protocol = ModelFactory.create(provider, model_name)
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
        # /clear 语义：同时 invalidate memory/rules caches + 重建 Context。
        # progressive 状态（nested fragments / matched rules）随新实例自然清空 ——
        # 不做原地 reset，避免遗漏字段。
        project_memory.clear_cache(self._cwd)
        rules.clear_cache(self._cwd)
        self._primary_memory = project_memory.load_project_memory(self._cwd)
        self._rules = rules.load_rules(self._cwd)
        self._context = self._build_context()
        self._loop = self._build_loop()
        journal.write("session_cleared", session=self._session_id)

    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def config(self) -> AuraConfig:
        """只读视图；CLI 层用来查 router / providers，避免摸 `_config`。"""
        return self._config

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
) -> Agent:
    # 生产便利工厂：自动解析 model + storage；Agent 构造器保持 DI 注入以便测试替换。
    provider, model_name = ModelFactory.resolve(config.router["default"], cfg=config)
    model, _protocol = ModelFactory.create(provider, model_name)
    storage = SessionStorage(config.resolved_storage_path())
    return Agent(
        config=config,
        model=model,
        storage=storage,
        hooks=default_hooks().merge(hooks or HookChain()),
        available_tools=available_tools,
        session_id=session_id,
    )
