"""Hook protocols composed by HookChain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from aura.domain.tool import ToolResult
from aura.schemas.permissions import Outcome
from aura.schemas.state import LoopState

FileChangeKind = Literal["created", "modified", "deleted"]


class PreModelHook(Protocol):
    async def __call__(
        self,
        *,
        history: list[BaseMessage],
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


class PostModelHook(Protocol):
    async def __call__(
        self,
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


class PreToolHook(Protocol):
    # Merge precedence: first Block > first Ask > first Replace > last Allow.

    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        **kwargs: Any,
    ) -> Outcome: ...


class PostToolHook(Protocol):
    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        result: ToolResult,
        state: LoopState,
        **kwargs: Any,
    ) -> ToolResult: ...


class FileChangedHook(Protocol):
    async def __call__(
        self,
        *,
        path: Path,
        kind: FileChangeKind,
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


class CwdChangedHook(Protocol):
    async def __call__(
        self,
        *,
        old_cwd: Path,
        new_cwd: Path,
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


__all__ = [
    "CwdChangedHook",
    "FileChangeKind",
    "FileChangedHook",
    "PostModelHook",
    "PostToolHook",
    "PreModelHook",
    "PreToolHook",
]
