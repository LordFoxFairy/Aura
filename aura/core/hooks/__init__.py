"""9 个生命周期 hook Protocol + HookChain — **kwargs: Any 保证向前兼容。

4 turn-cycle hooks (pre_model / post_model / pre_tool / post_tool) + 5
session-cycle hooks (pre_session / post_session / post_subagent /
pre_compact / pre_user_prompt). Session-cycle hooks are non-blocking
signals — they receive **kwargs and return None; they cannot short-circuit
or mutate pipeline state (unlike pre_tool / post_tool which DO gate / shape
the tool call).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult


class PreModelHook(Protocol):
    # 可原地 mutate history（compact / inject system message 等场景）；无返回值。
    async def __call__(
        self,
        *,
        history: list[BaseMessage],
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


class PostModelHook(Protocol):
    # 只读观察（usage 累计 / audit log）；不得修改 history 或 ai_message。
    async def __call__(
        self,
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


class PreToolHook(Protocol):
    # 返回 ToolResult = 短路，记入 history 但不调 ainvoke；返回 None = 放行。
    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        **kwargs: Any,
    ) -> ToolResult | None: ...


class PostToolHook(Protocol):
    # 链式调用：上一个 hook 的输出作为下一个 hook 的 result 输入（变换而非观察）。
    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        result: ToolResult,
        state: LoopState,
        **kwargs: Any,
    ) -> ToolResult: ...


# ---------------------------------------------------------------------------
# Session-cycle hooks — lifecycle signals the harness fires at well-known
# transitions. All are non-blocking: return None, side-effect only. The
# harness never consults their output, and an exception raised inside one
# does NOT abort the triggering operation (callers swallow + journal — see
# each call site in agent.py / loop.py / compact.py / tasks/run.py).
# ---------------------------------------------------------------------------


class PreSessionHook(Protocol):
    # Fires once per Agent at ``bootstrap()``, after __init__ wiring has
    # completed. Typical use: start a session-scoped resource (tracing span,
    # usage counter, file handle). ``session_id`` + ``cwd`` identify the
    # session; more kwargs may be added in the future.
    async def __call__(
        self,
        *,
        session_id: str,
        cwd: object,
        **kwargs: Any,
    ) -> None: ...


class PostSessionHook(Protocol):
    # Fires once per Agent at ``shutdown()``, BEFORE teardown (MCP stop,
    # storage close). Symmetric with PreSessionHook. ``close()`` itself is
    # sync and does NOT fire this — shutdown() is the async pair.
    async def __call__(
        self,
        *,
        session_id: str,
        cwd: object,
        **kwargs: Any,
    ) -> None: ...


class PostSubagentHook(Protocol):
    # Fires inside ``run_task`` at each terminal transition (completed /
    # failed / cancelled). Non-blocking: the subagent's record is already
    # written before this fires.
    async def __call__(
        self,
        *,
        task_id: str,
        status: str,
        final_text: str,
        error: str | None,
        **kwargs: Any,
    ) -> None: ...


class PreCompactHook(Protocol):
    # Fires inside ``run_compact`` BEFORE the summary LLM turn. ``trigger``
    # is one of ``"manual"`` / ``"auto"`` / ``"reactive"`` — matches the
    # ``source`` param passed to ``Agent.compact``. ``state`` is the live
    # LoopState (read-only reference; hook should treat as such).
    async def __call__(
        self,
        *,
        state: LoopState,
        trigger: str,
        **kwargs: Any,
    ) -> None: ...


class PreUserPromptHook(Protocol):
    # Fires inside ``AgentLoop.run_turn`` right before the user's
    # HumanMessage lands in history. ``prompt`` is the raw user string.
    # Observational only — the hook cannot mutate the prompt or short-circuit
    # the turn (use pre_model for that).
    async def __call__(
        self,
        *,
        prompt: str,
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


@dataclass
class HookChain:
    pre_model: list[PreModelHook] = field(default_factory=list)
    post_model: list[PostModelHook] = field(default_factory=list)
    pre_tool: list[PreToolHook] = field(default_factory=list)
    post_tool: list[PostToolHook] = field(default_factory=list)
    # Session-cycle slots. All default empty so existing code paths
    # (hooks=None, default_hooks(), caller-supplied chains) keep identical
    # behavior — adding these slots is a pure extension.
    pre_session: list[PreSessionHook] = field(default_factory=list)
    post_session: list[PostSessionHook] = field(default_factory=list)
    post_subagent: list[PostSubagentHook] = field(default_factory=list)
    pre_compact: list[PreCompactHook] = field(default_factory=list)
    pre_user_prompt: list[PreUserPromptHook] = field(default_factory=list)

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
        self, *, tool: BaseTool, args: dict[str, Any], state: LoopState,
    ) -> ToolResult | None:
        for hook in self.pre_tool:
            decision = await hook(tool=tool, args=args, state=state)
            if decision is not None:
                return decision
        return None

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

    async def run_pre_session(
        self, *, session_id: str, cwd: object,
    ) -> None:
        for hook in self.pre_session:
            await hook(session_id=session_id, cwd=cwd)

    async def run_post_session(
        self, *, session_id: str, cwd: object,
    ) -> None:
        for hook in self.post_session:
            await hook(session_id=session_id, cwd=cwd)

    async def run_post_subagent(
        self,
        *,
        task_id: str,
        status: str,
        final_text: str,
        error: str | None,
    ) -> None:
        for hook in self.post_subagent:
            await hook(
                task_id=task_id,
                status=status,
                final_text=final_text,
                error=error,
            )

    async def run_pre_compact(
        self, *, state: LoopState, trigger: str,
    ) -> None:
        for hook in self.pre_compact:
            await hook(state=state, trigger=trigger)

    async def run_pre_user_prompt(
        self, *, prompt: str, state: LoopState,
    ) -> None:
        for hook in self.pre_user_prompt:
            await hook(prompt=prompt, state=state)

    def merge(self, other: HookChain) -> HookChain:
        # 非破坏性拼接：self 优先 other 后；不修改任何一方的原始列表。
        return HookChain(
            pre_model=[*self.pre_model, *other.pre_model],
            post_model=[*self.post_model, *other.post_model],
            pre_tool=[*self.pre_tool, *other.pre_tool],
            post_tool=[*self.post_tool, *other.post_tool],
            pre_session=[*self.pre_session, *other.pre_session],
            post_session=[*self.post_session, *other.post_session],
            post_subagent=[*self.post_subagent, *other.post_subagent],
            pre_compact=[*self.pre_compact, *other.pre_compact],
            pre_user_prompt=[*self.pre_user_prompt, *other.pre_user_prompt],
        )
