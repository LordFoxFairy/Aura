"""Agent facade — composes the loop with storage and config into the public API."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from langchain_core.language_models import BaseChatModel

from aura.config.schema import AuraConfig, AuraConfigError
from aura.core.events import AgentEvent, Final
from aura.core.hooks import HookChain
from aura.core.llm import ModelFactory
from aura.core.loop import AgentLoop
from aura.core.registry import ToolRegistry
from aura.core.state import LoopState
from aura.core.storage import SessionStorage
from aura.tools.base import AuraTool
from aura.tools.bash import bash
from aura.tools.read_file import read_file
from aura.tools.write_file import write_file

_BUILTIN_TOOLS: dict[str, AuraTool] = {
    "read_file": read_file,
    "write_file": write_file,
    "bash": bash,
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
    ) -> None:
        self._config = config
        self._model = model
        self._storage = storage
        self._hooks = hooks or HookChain()
        self._state = LoopState()
        self._loop = self._build_loop()

    async def astream(self, prompt: str) -> AsyncIterator[AgentEvent]:
        history = self._storage.load(_DEFAULT_SESSION)
        try:
            async for event in self._loop.run_turn(user_prompt=prompt, history=history):
                yield event
        except asyncio.CancelledError:
            yield Final(message="(cancelled)")
            raise
        else:
            self._storage.save(_DEFAULT_SESSION, history)

    def switch_model(self, spec: str) -> None:
        provider, model_name = ModelFactory.resolve(spec, cfg=self._config)
        self._model, _protocol = ModelFactory.create(provider, model_name)
        self._loop = self._build_loop()

    def clear_session(self) -> None:
        self._storage.clear(_DEFAULT_SESSION)

    @property
    def state(self) -> LoopState:
        return self._state

    def _build_loop(self) -> AgentLoop:
        return AgentLoop(
            model=self._model,
            registry=self._build_registry(),
            hooks=self._hooks,
            state=self._state,
        )

    def _build_registry(self) -> ToolRegistry:
        tools: list[AuraTool] = []
        for name in self._config.tools.enabled:
            tool = _BUILTIN_TOOLS.get(name)
            if tool is None:
                raise AuraConfigError(
                    source="tools.enabled",
                    detail=f"unknown tool name: {name!r}",
                )
            tools.append(tool)
        return ToolRegistry(tools)


def build_agent(
    config: AuraConfig, *, hooks: HookChain | None = None
) -> Agent:
    provider, model_name = ModelFactory.resolve(config.router["default"], cfg=config)
    model, _protocol = ModelFactory.create(provider, model_name)
    storage = SessionStorage(config.resolved_storage_path())
    return Agent(config=config, model=model, storage=storage, hooks=hooks)
