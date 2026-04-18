"""Agent facade — config + model + storage + hooks 组装成一条对话的入口层。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from langchain_core.language_models import BaseChatModel

from aura.config.schema import AuraConfig, AuraConfigError
from aura.core.events import AgentEvent, Final
from aura.core.hooks import HookChain
from aura.core.hooks.budget import MaxTurnsExceeded, default_hooks
from aura.core.llm import ModelFactory
from aura.core.loop import AgentLoop
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.registry import ToolRegistry
from aura.core.state import LoopState
from aura.core.system_prompt import build_system_prompt
from aura.tools.base import AuraTool
from aura.tools.bash import bash
from aura.tools.edit_file import edit_file
from aura.tools.glob import glob
from aura.tools.grep import grep
from aura.tools.read_file import read_file
from aura.tools.web_fetch import web_fetch
from aura.tools.write_file import write_file

_BUILTIN_TOOLS: dict[str, AuraTool] = {
    "bash": bash,
    "edit_file": edit_file,
    "glob": glob,
    "grep": grep,
    "read_file": read_file,
    "web_fetch": web_fetch,
    "write_file": write_file,
}

_DEFAULT_SESSION = "default"


class Agent:
    def __init__(
        self,
        config: AuraConfig,
        *,
        model: BaseChatModel,
        storage: SessionStorage,
        hooks: HookChain | None = None,
        available_tools: dict[str, AuraTool] | None = None,
        session_id: str = _DEFAULT_SESSION,
    ) -> None:
        self._config = config
        self._model = model
        self._storage = storage
        self._hooks = hooks or HookChain()
        self._state = LoopState()
        self._available_tools = (
            dict(available_tools) if available_tools is not None else dict(_BUILTIN_TOOLS)
        )
        self._session_id = session_id
        self._registry = self._build_registry()
        self._system_prompt = build_system_prompt(registry=self._registry)
        self._loop = self._build_loop()

    async def astream(self, prompt: str) -> AsyncIterator[AgentEvent]:
        # Invariant 2（§4.2）的事务性在此层实现：history 仅当 turn 正常完成才 save。
        # CancelledError → yield Final + re-raise → 跳过 else 分支 → 下次从 pre-turn 状态恢复。
        # MaxTurnsExceeded → yield Final + return → 同样跳过 save，history 保持上次持久化状态。
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
            hooks=self._hooks,
            state=self._state,
            system_prompt=self._system_prompt,
        )

    def _build_registry(self) -> ToolRegistry:
        tools: list[AuraTool] = []
        for name in self._config.tools.enabled:
            tool = self._available_tools.get(name)
            if tool is None:
                raise AuraConfigError(
                    source="tools.enabled",
                    detail=f"unknown tool name: {name!r}",
                )
            tools.append(tool)
        return ToolRegistry(tools)

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
    available_tools: dict[str, AuraTool] | None = None,
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
        hooks=default_hooks().merge(hooks) if hooks is not None else default_hooks(),
        available_tools=available_tools,
        session_id=session_id,
    )
