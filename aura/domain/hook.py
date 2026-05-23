"""Hook protocols + outcome value objects.

Pure-type contracts the orchestration layer composes; implementation
(``HookChain``) lives in :mod:`aura.application.hooks`. Zero deps
into other ``aura.*`` runtime layers — only ``aura.schemas`` (wire
shapes) and ``langchain_core`` (message / tool types).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from aura.schemas.permissions import Outcome
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult

FileChangeKind = Literal["created", "modified", "deleted"]
NotificationKind = Literal["permission_prompt", "ask_user", "error"]
StopReason = Literal["user_exit", "clear", "max_turns", "error"]


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
    """Gate one tool call; returns an :class:`Outcome` variant.

    Merge precedence (spec §3.2): first ``Block`` wins → first ``Ask``
    wins → first ``Replace`` wins → last ``Allow`` wins.
    """

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


@dataclass(frozen=True)
class UserPromptSubmitOutcome:
    """``prompt=None`` passes through; a string rewrites the prompt."""

    prompt: str | None = None


class SessionStartHook(Protocol):
    async def __call__(
        self,
        *,
        session_id: str,
        mode: str,
        cwd: Path,
        model_name: str,
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


class UserPromptSubmitHook(Protocol):
    async def __call__(
        self,
        *,
        session_id: str,
        turn_count: int,
        user_text: str,
        state: LoopState,
        **kwargs: Any,
    ) -> UserPromptSubmitOutcome: ...


class NotificationHook(Protocol):
    async def __call__(
        self,
        *,
        session_id: str,
        kind: NotificationKind,
        body: str,
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


class StopHook(Protocol):
    async def __call__(
        self,
        *,
        session_id: str,
        reason: StopReason,
        turn_count: int,
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


__all__ = [
    "CwdChangedHook",
    "FileChangeKind",
    "FileChangedHook",
    "NotificationHook",
    "NotificationKind",
    "PostModelHook",
    "PostToolHook",
    "PreModelHook",
    "PreToolHook",
    "SessionStartHook",
    "StopHook",
    "StopReason",
    "UserPromptSubmitHook",
    "UserPromptSubmitOutcome",
]
