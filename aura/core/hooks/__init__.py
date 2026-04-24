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
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from aura.core.permissions.decision import Decision
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult

#: Values accepted for the ``trigger`` field of the pre-compact hook. Mirrors
#: the ``source`` literal on ``Agent.compact`` — kept here so hook authors get
#: a real type (not ``str``) and typos are caught at call-site type-check time.
CompactTrigger = Literal["manual", "auto", "reactive"]


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
    # matches the ``source`` param passed to ``Agent.compact``. ``state`` is
    # the live LoopState (read-only reference; hook should treat as such).
    async def __call__(
        self,
        *,
        state: LoopState,
        trigger: CompactTrigger,
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
        self, *, state: LoopState, trigger: CompactTrigger,
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
