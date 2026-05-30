"""HookChain composes Protocol-typed hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from aura.application.hooks.protocols import (
    CwdChangedHook,
    FileChangedHook,
    FileChangeKind,
    PostModelHook,
    PostToolHook,
    PreModelHook,
    PreToolHook,
)
from aura.application.loop_state import LoopState
from aura.domain.permission.decision import Decision
from aura.domain.permission.outcome import Allow, Ask, Block, Outcome, Replace
from aura.domain.tool import ToolResult

_ASK_RESOLVED_REASONS = frozenset({"user_accept", "user_always"})


# Hook closures expose __qualname__; arbitrary __call__ instances may not.
@runtime_checkable
class _QualNamed(Protocol):
    __qualname__: str


def _hook_name(hook: PreToolHook) -> str:
    module = hook.__module__
    qualname = hook.__qualname__ if isinstance(hook, _QualNamed) else repr(hook)
    return f"{module}.{qualname}".lstrip(".")


def _merge_outcomes(outcomes: list[Outcome], ask_requested: bool) -> Outcome:
    """Merge pre_tool outcomes per spec §3.2: Block > Replace > Allow/Ask precedence."""
    for o in outcomes:
        if isinstance(o, Block):
            return o
    for o in outcomes:
        if isinstance(o, Replace):
            return o
    first_authoritative: Allow | None = None
    last_allow: Allow | None = None
    for o in outcomes:
        if isinstance(o, Allow):
            last_allow = o
            if first_authoritative is None and o.decision.reason != "mode_bypass":
                first_authoritative = o
    winner_allow = first_authoritative if first_authoritative is not None else last_allow
    if winner_allow is not None:
        if ask_requested and winner_allow.decision.reason not in _ASK_RESOLVED_REASONS:
            for o in outcomes:
                if isinstance(o, Ask):
                    return o
            return Ask(reason="pending escalation")
        return winner_allow
    for o in outcomes:
        if isinstance(o, Ask):
            return o
    raise AssertionError("_merge_outcomes called with empty outcomes")


@dataclass
class HookChain:
    pre_model: list[PreModelHook] = field(default_factory=list)
    post_model: list[PostModelHook] = field(default_factory=list)
    pre_tool: list[PreToolHook] = field(default_factory=list)
    post_tool: list[PostToolHook] = field(default_factory=list)
    file_changed: list[FileChangedHook] = field(default_factory=list)
    cwd_changed: list[CwdChangedHook] = field(default_factory=list)

    async def run_pre_model(
        self, *, history: list[BaseMessage], state: LoopState,
    ) -> None:
        for hook in self.pre_model:
            await hook(history=history, state=state)

    async def run_post_model(
        self,
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],
        state: LoopState,
    ) -> None:
        for hook in self.post_model:
            await hook(ai_message=ai_message, history=history, state=state)

    async def run_pre_tool(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        **kwargs: Any,
    ) -> Outcome:
        # ask_pending: once any hook returns Ask, downstream auto-allow demotes to asker.
        from aura.infrastructure.persistence import journal

        outcomes: list[Outcome] = []
        ask_pending = False
        for hook in self.pre_tool:
            raw = await hook(
                tool=tool, args=args, state=state,
                ask_pending=ask_pending, **kwargs,
            )
            outcomes.append(raw)

            if isinstance(raw, Ask):
                ask_pending = True

            if isinstance(raw, Block | Ask | Replace):
                # Ask carries no Decision; Block/Replace do.
                decision = None if isinstance(raw, Ask) else raw.decision
                journal.write(
                    "pre_tool_hook_decision",
                    hook=_hook_name(hook),
                    tool=tool.name,
                    allow=False if decision is None else decision.allow,
                    reason="" if decision is None else decision.reason,
                )

            if isinstance(raw, Block):
                return raw

        if outcomes:
            return _merge_outcomes(outcomes, ask_pending)
        return Allow(decision=Decision(allow=True, reason="chain_empty"))

    async def run_post_tool(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        result: ToolResult,
        state: LoopState,
    ) -> ToolResult:
        for hook in self.post_tool:
            result = await hook(
                tool=tool, args=args, result=result, state=state,
            )
        return result

    async def run_file_changed(
        self,
        *,
        path: Path,
        kind: FileChangeKind,
        state: LoopState,
    ) -> None:
        for hook in self.file_changed:
            await hook(path=path, kind=kind, state=state)

    async def run_cwd_changed(
        self,
        *,
        old_cwd: Path,
        new_cwd: Path,
        state: LoopState,
    ) -> None:
        for hook in self.cwd_changed:
            await hook(old_cwd=old_cwd, new_cwd=new_cwd, state=state)

    def merge(self, other: HookChain) -> HookChain:
        return HookChain(
            pre_model=[*self.pre_model, *other.pre_model],
            post_model=[*self.post_model, *other.post_model],
            pre_tool=[*self.pre_tool, *other.pre_tool],
            post_tool=[*self.post_tool, *other.post_tool],
            file_changed=[*self.file_changed, *other.file_changed],
            cwd_changed=[*self.cwd_changed, *other.cwd_changed],
        )


__all__ = [
    "CwdChangedHook",
    "FileChangeKind",
    "FileChangedHook",
    "HookChain",
    "PostModelHook",
    "PostToolHook",
    "PreModelHook",
    "PreToolHook",
]
