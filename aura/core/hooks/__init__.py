"""4 turn-cycle hook Protocols + HookChain — **kwargs: Any stays for forward compat.

4 turn-cycle hooks: ``pre_model`` / ``post_model`` / ``pre_tool`` /
``post_tool``. ``pre_tool`` / ``post_tool`` gate or shape the tool call;
``pre_model`` may mutate history (compact, inject system message);
``post_model`` is observational.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from aura.core.permissions.decision import Decision
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult


@dataclass(frozen=True)
class PreToolOutcome:
    """What a :class:`PreToolHook` returns — two independent channels.

    - ``short_circuit``: non-None replaces the tool's own execution with
      this :class:`ToolResult`. The tool is NOT invoked; the result lands
      in history as a ToolMessage. Used by safety / budget / must-read
      hooks to block a call without consulting the model.
    - ``decision``: non-None carries a permission :class:`Decision` up to
      :class:`aura.core.loop.AgentLoop._plan_tool_calls` which stamps it
      onto the :class:`aura.core.loop.ToolStep` so a PermissionAudit can
      be emitted between ToolCallStarted and ToolCallCompleted. A hook
      that has nothing to say about permission returns ``decision=None``.

    Both fields default to ``None`` — a hook that neither short-circuits
    nor emits a decision returns :obj:`PreToolOutcome.passthrough` (or
    equivalently ``PreToolOutcome()``). Returning an implicit ``None``
    from a PreToolHook is a type error by design — the contract is strict
    so future readers can rely on "every hook returns an outcome".

    Minimal usage — a hook that only short-circuits::

        async def my_hook(*, tool, args, state, **_):
            if _should_block(tool, args):
                return PreToolOutcome(
                    short_circuit=ToolResult(ok=False, error="blocked"),
                    decision=None,
                )
            return PRE_TOOL_PASSTHROUGH
    """

    short_circuit: ToolResult | None = None
    decision: Decision | None = None


# Shared sentinel for the "nothing to do" case so common hooks don't each
# allocate a fresh PreToolOutcome on every call. Frozen dataclass = safe
# to share by reference.
PRE_TOOL_PASSTHROUGH: PreToolOutcome = PreToolOutcome()


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
    """Pre-tool hook — gates / observes one tool call, returns PreToolOutcome.

    The return value is strict (:class:`PreToolOutcome`, not
    ``PreToolOutcome | None``): a hook that has nothing to do MUST
    return :data:`PRE_TOOL_PASSTHROUGH` (or equivalently
    ``PreToolOutcome()``). This catches "I forgot to return anything"
    bugs at the type-checker, and keeps the
    :meth:`HookChain.run_pre_tool` merge loop simple.

    Channels:

    - ``outcome.short_circuit`` non-None → tool is NOT invoked; this
      :class:`ToolResult` becomes the tool's result.
    - ``outcome.decision`` non-None → permission :class:`Decision`
      surfaces on :class:`aura.core.loop.ToolStep.permission_decision`
      for the auditor to emit a PermissionAudit event.

    Both channels are independent. A permission hook typically sets
    BOTH (allow-with-reason: ``decision=allow, short_circuit=None``;
    deny: ``decision=deny, short_circuit=ToolResult(ok=False,...)``).
    """

    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        **kwargs: Any,
    ) -> PreToolOutcome: ...


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


@dataclass
class HookChain:
    pre_model: list[PreModelHook] = field(default_factory=list)
    post_model: list[PostModelHook] = field(default_factory=list)
    pre_tool: list[PreToolHook] = field(default_factory=list)
    post_tool: list[PostToolHook] = field(default_factory=list)

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
    ) -> PreToolOutcome:
        """Merge pre_tool hook outcomes across the chain.

        Merge semantics (intentional asymmetry — the two channels model
        different intents):

        - ``short_circuit`` is **first-wins**. The first hook that emits
          a non-None ``short_circuit`` stops the chain immediately —
          subsequent hooks are NOT called. Matches the pre-G4 contract:
          safety / budget denials take precedence over anything further
          down the chain, and once a ToolResult is decided there's
          nothing for later hooks to add.
        - ``decision`` is **last-wins** across hooks the chain actually
          ran. Permission hooks typically sit at the end of the chain
          (permission is the final gate before a tool executes), so the
          last non-None decision is the authoritative one. A hook
          earlier in the chain emitting a decision does not block a
          later hook from overriding it — if an early hook
          short-circuits, the later hook never runs so its (potential)
          decision is naturally not considered.

        ``**kwargs`` forwards per-call metadata to hooks. Today
        ``tool_call_id`` flows through (G5 uses it to stamp
        PermissionDenial records); additional keys land as more
        workstreams plumb data into the hook surface without having to
        revise every call site. :class:`PreToolHook` accepts
        ``**kwargs`` by protocol so unchanged hooks silently ignore
        new keys.
        """
        merged_decision: Decision | None = None
        for hook in self.pre_tool:
            outcome = await hook(tool=tool, args=args, state=state, **kwargs)
            # Last-wins: any non-None decision updates the running merge.
            if outcome.decision is not None:
                merged_decision = outcome.decision
            # First-wins: stop immediately on the first short-circuit.
            if outcome.short_circuit is not None:
                return PreToolOutcome(
                    short_circuit=outcome.short_circuit,
                    decision=merged_decision,
                )
        return PreToolOutcome(short_circuit=None, decision=merged_decision)

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

    def merge(self, other: HookChain) -> HookChain:
        # 非破坏性拼接：self 优先 other 后；不修改任何一方的原始列表。
        return HookChain(
            pre_model=[*self.pre_model, *other.pre_model],
            post_model=[*self.post_model, *other.post_model],
            pre_tool=[*self.pre_tool, *other.pre_tool],
            post_tool=[*self.post_tool, *other.post_tool],
        )
