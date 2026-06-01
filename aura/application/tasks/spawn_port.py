"""Structural ports for subagent spawning — leaf module, no session import."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel

from aura.infrastructure.persistence.storage import SessionStorage


class SpawnedAgent(Protocol):
    """Runnable child-agent surface a task runner depends on."""

    @property
    def config(self) -> Any: ...

    @property
    def model(self) -> BaseChatModel: ...

    @property
    def storage(self) -> SessionStorage: ...

    @property
    def session_id(self) -> str: ...

    @property
    def hooks(self) -> Any: ...

    def astream(self, prompt: str) -> AsyncIterator[Any]: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class SpawnPort(Protocol):
    """The spawn surface a dispatcher tool depends on — no concrete agent type."""

    @property
    def parent_model_spec(self) -> str: ...

    @property
    def abort_event(self) -> asyncio.Event | None: ...

    def validate_model_spec(self, spec: str) -> None: ...

    def spawn(
        self,
        prompt: str,
        allowed_tools: list[str] | None = None,
        *,
        agent_type: str = "general-purpose",
        task_id: str | None = None,
        model_spec: str | None = None,
    ) -> SpawnedAgent: ...
