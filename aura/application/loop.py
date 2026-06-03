"""AgentLoop — 协调一次对话的 turn 循环，驱动 model → tool → model 的迭代。"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from aura.application.compact import Compactor, MicrocompactPolicy, apply_microcompact
from aura.application.hooks import HookChain
from aura.application.loop_state import LoopState
from aura.application.memory.context import Context
from aura.config.schema import RetryConfig
from aura.domain.abort import AbortController, AbortException, current_abort_signal
from aura.domain.context_overflow import is_context_overflow
from aura.domain.events import (
    AgentEvent,
    AssistantDelta,
    Final,
    PermissionAudit,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)
from aura.domain.permission.decision import Decision
from aura.domain.permission.outcome import Allow, Ask, Block, Replace
from aura.domain.tool import ToolError, ToolResult
from aura.domain.tool_meta_access import meta_dict
from aura.domain.tool_registry import ToolRegistry
from aura.infrastructure.persistence import journal
from aura.infrastructure.retry import with_retry
from aura.tools.errors import hint_for_error
from aura.tools.progress import (
    ProgressCallback,
    reset_progress_callback,
    set_progress_callback,
)

# Shared with aura.application.session so AgentLoop and AgentSession defaults never drift.
DEFAULT_SESSION = "default"

# Auto-decisions whose dim "auto-allowed: <reason>" the renderer surfaces.
_AUTO_ALLOW_REASONS: frozenset[str] = frozenset(
    {"rule_allow", "mode_bypass"},
)

# Resolved at __init__ so a mid-session env flip can't race the outer wait.
_BATCH_TIMEOUT_ENV_VAR = "AURA_BATCH_TIMEOUT_SEC"
_DEFAULT_BATCH_TIMEOUT_SEC: float = 60.0

# Cap retries so a stuck "length" finish reason can't loop forever.
_MAX_LENGTH_RETRY: int = 3

# OpenAI → ``length``; Anthropic → ``max_tokens``. Lowercased before compare.
_LENGTH_FINISH_REASONS: frozenset[str] = frozenset({"length", "max_tokens"})

_LENGTH_RESUME_PROMPT: str = (
    "(Output token limit hit. Resume directly from where you stopped, "
    "without repeating prior content.)"
)


# Anthropic surfaces stop_reason as a top-level attr; OpenAI omits it.
@runtime_checkable
class _HasStopReason(Protocol):
    stop_reason: object


def _length_truncated(ai: AIMessage) -> bool:
    """True iff ``ai`` was cut short by a provider's max-output-tokens cap."""
    meta = ai.response_metadata
    candidates: list[object] = [
        meta.get("finish_reason"),
        meta.get("stop_reason"),
        ai.stop_reason if isinstance(ai, _HasStopReason) else None,
    ]
    return any(
        isinstance(raw, str) and raw.lower() in _LENGTH_FINISH_REASONS
        for raw in candidates
    )


def _resolve_batch_timeout(override: float | None) -> float:
    """Effective batch deadline in seconds; ``0.0`` disables the feature.

    Precedence: ``override`` kwarg > ``AURA_BATCH_TIMEOUT_SEC`` env >
    :data:`_DEFAULT_BATCH_TIMEOUT_SEC`. Malformed env strings fall through
    rather than raising — the loop constructor is a hot path.
    """
    if override is not None:
        return override if override > 0 else 0.0
    raw = os.environ.get(_BATCH_TIMEOUT_ENV_VAR)
    if raw is not None:
        try:
            parsed = float(raw)
        except ValueError:
            return _DEFAULT_BATCH_TIMEOUT_SEC
        return parsed if parsed > 0 else 0.0
    return _DEFAULT_BATCH_TIMEOUT_SEC


# Tools whose successful invocation feeds a path back into Context state.
# bash (shell semantics vary) and web_fetch (URLs) are deliberately excluded.
PATH_TRIGGER_TOOLS: dict[str, str] = {
    "read_file": "path",
    "write_file": "path",
    "edit_file": "path",
    "grep": "path",
    "glob": "path",
}

ProgressItem = tuple[str, str, Literal["stdout", "stderr"], str]


def _validated_args_model(schema: object) -> type[BaseModel] | None:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    return None


def _serialize(result: ToolResult, *, tool_name: str = "") -> str:
    # default=str avoids exceptions on datetime/Path/bytes — a raise here
    # would drop the ToolMessage and break tool_call.id alignment.
    if result.ok:
        return json.dumps(result.output, default=str, ensure_ascii=False)
    # Embed the UI hint so the model gets the same recovery guidance.
    error = result.error or "tool failed"
    hint = hint_for_error(tool_name, error)
    if hint is not None:
        return f"{error}\n\nHint: {hint}"
    return error


@dataclass(frozen=True)
class ToolStep:
    tool_call: ToolCall
    tool: BaseTool | None
    args: dict[str, object] | None
    decision: ToolResult | None
    # Captured from pre_tool merge so the loop can emit PermissionAudit.
    permission_decision: Decision | None = None


def partition_batches(steps: list[ToolStep]) -> list[list[ToolStep]]:
    """Group ``steps`` into ordered batches; concurrency-safe runs merge."""
    batches: list[list[ToolStep]] = []
    current: list[ToolStep] = []
    for step in steps:
        tool = step.tool
        safe = (
            tool is not None
            and meta_dict(tool).get("is_concurrency_safe", False)
            and step.decision is None
        )
        if safe:
            current.append(step)
            continue
        if current:
            batches.append(current)
            current = []
        batches.append([step])
    if current:
        batches.append(current)
    return batches


class AgentLoop:
    _DEFAULT_MAX_TURNS: int | None = None
    # One retry only: if one compact didn't fit, repeats won't either.
    _MAX_REACTIVE_COMPACT: int = 1

    def __init__(
        self,
        *,
        model: BaseChatModel,
        registry: ToolRegistry,
        context: Context,
        hooks: HookChain | None = None,
        state: LoopState | None = None,
        max_turns: int | None = _DEFAULT_MAX_TURNS,
        retry_config: RetryConfig | None = None,
        session_id: str = DEFAULT_SESSION,
        microcompact_policy: MicrocompactPolicy | None = None,
        batch_timeout_sec: float | None = None,
        compact_callback: Callable[
            [list[BaseMessage]], Awaitable[None]
        ] | None = None,
        compactor: Compactor | None = None,
    ) -> None:
        self._registry = registry
        self._hooks = hooks or HookChain()
        self._state = state or LoopState()
        self._context = context
        self._retry_config = retry_config or RetryConfig()
        self._session_id = session_id
        self._microcompact_policy = microcompact_policy
        self._max_turns = max_turns
        # Raw model kept so rebind_tools works — some providers reject re-bind.
        self._model = model
        # tools=[] is inconsistent across providers; skip bind on empty.
        self._bound = model.bind_tools(registry.tools()) if len(registry) > 0 else model
        # 0.0 sentinel disables the per-batch deadline.
        self._batch_timeout_sec = _resolve_batch_timeout(batch_timeout_sec)
        self._compact_callback = compact_callback
        self._compactor = compactor

    def rebind_tools(self, tools: list[BaseTool]) -> None:
        """Rebind the loop's model with an updated tool set."""
        self._bound = self._model.bind_tools(tools) if tools else self._model

    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def microcompact_policy(self) -> MicrocompactPolicy | None:
        return self._microcompact_policy

    @property
    def max_turns(self) -> int | None:
        return self._max_turns

    async def run_turn(
        self,
        *,
        history: list[BaseMessage],
        abort: AbortController | None = None,
    ) -> AsyncIterator[AgentEvent]:
        # Caller owns the user HumanMessage append + persist so a mid-turn
        # crash can't erase the user's input.
        # Per-turn sinks reset at the turn boundary; list identity is stable
        # so AgentSession.last_turn_denials keeps pointing at the live bucket.
        self._state.slots.turn_denials.clear()
        self._state.slots.perm_dedup_cache.clear()
        # contextvar lets tools and spawned subagents inherit the signal.
        ctx_token = None
        if abort is not None:
            ctx_token = current_abort_signal.set(abort)
        try:
            while True:
                journal.write("turn_begin", turn=self._state.turn_count + 1)
                # Gate between turns so cancel doesn't race the next ainvoke.
                if abort is not None and abort.aborted:
                    self._synthesise_missing_tool_messages(history, set())
                    raise AbortException(abort.reason or "aborted")
                ai = await self._invoke_model_with_abort(history, abort)

                if ai.content:
                    yield AssistantDelta(text=str(ai.content))
                if not ai.tool_calls:
                    if (
                        ai.response_metadata.get("finish_reason")
                        == "length_recovery_exhausted"
                    ):
                        journal.write(
                            "turn_end",
                            turn=self._state.turn_count,
                            ended_with="length_recovery_exhausted",
                        )
                        yield Final(
                            message=str(ai.content),
                            reason="length_recovery_exhausted",
                        )
                        return
                    journal.write(
                        "turn_end",
                        turn=self._state.turn_count, ended_with="final",
                    )
                    yield Final(message=str(ai.content))
                    return

                # On cancel, synthesise one ToolMessage per unanswered call
                # so providers don't 400 on the trailing tool_use AIMessage.
                try:
                    async for event in self._dispatch_tool_calls(
                        ai.tool_calls, history,
                    ):
                        yield event
                except (AbortException, asyncio.CancelledError):
                    answered = {
                        m.tool_call_id for m in history
                        if isinstance(m, ToolMessage)
                    }
                    self._synthesise_missing_tool_messages(history, answered)
                    raise
                journal.write(
                    "turn_end",
                    turn=self._state.turn_count,
                    ended_with="tool_loop",
                    tool_count=len(ai.tool_calls),
                )

                if (
                    self._max_turns is not None
                    and self._state.turn_count >= self._max_turns
                ):
                    journal.write(
                        "turn_end",
                        turn=self._state.turn_count,
                        ended_with="max_turns_reached",
                        max_turns=self._max_turns,
                    )
                    yield Final(
                        message=f"max turns reached ({self._max_turns})",
                        reason="max_turns",
                    )
                    return
        finally:
            if ctx_token is not None:
                current_abort_signal.reset(ctx_token)

    def _synthesise_missing_tool_messages(
        self,
        history: list[BaseMessage],
        answered_ids: set[str],
    ) -> None:
        """Pair every trailing tool_use with an "(aborted)" ToolMessage.

        Providers reject unmatched tool_use/tool_result pairs; this keeps
        the next astream's history valid. Idempotent on answered ids.
        """
        for msg in reversed(history):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id")
                    if not tc_id or tc_id in answered_ids:
                        continue
                    history.append(
                        ToolMessage(
                            content="(aborted by user)",
                            tool_call_id=tc_id,
                            name=tc.get("name", "?"),
                            status="error",
                        ),
                    )
                    answered_ids.add(tc_id)
                return

    async def _invoke_model_with_abort(
        self,
        history: list[BaseMessage],
        abort: AbortController | None,
    ) -> AIMessage:
        """Race ``_invoke_model`` against ``abort.signal``; cancel the loser."""
        if abort is None:
            return await self._invoke_model(history)
        invoke_task: asyncio.Task[AIMessage] = asyncio.ensure_future(
            self._invoke_model(history),
        )
        abort_task: asyncio.Task[bool] = asyncio.ensure_future(
            abort.signal.wait(),
        )
        try:
            done, _ = await asyncio.wait(
                {invoke_task, abort_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            invoke_task.cancel()
            abort_task.cancel()
            raise
        if abort_task in done and invoke_task not in done:
            invoke_task.cancel()
            with contextlib.suppress(BaseException):
                await invoke_task
            raise AbortException(abort.reason or "aborted")
        abort_task.cancel()
        return invoke_task.result()

    async def _invoke_with_retry(
        self, messages: list[BaseMessage],
    ) -> AIMessage:
        """Wrap just the SDK ainvoke through retry policy; tool semantics untouched."""
        async def _do_invoke() -> AIMessage:
            return await self._bound.ainvoke(messages)

        return await with_retry(
            _do_invoke,
            max_attempts=self._retry_config.max_attempts,
            base_delay_s=self._retry_config.base_delay_s,
            max_delay_s=self._retry_config.max_delay_s,
        )

    async def _invoke_model(self, history: list[BaseMessage]) -> AIMessage:
        # Bump turn_count first so pre_model hooks see "the Nth turn about to run".
        self._state.turn_count += 1
        await self._hooks.run_pre_model(history=history, state=self._state)
        recompact_attempts = 0
        messages: list[BaseMessage]
        ai: AIMessage
        while True:
            messages = await self._build_view(history)
            try:
                ai = await self._invoke_with_retry(messages)
                break
            except Exception as exc:
                has_reactive_path = (
                    self._compactor is not None
                    or self._compact_callback is not None
                )
                if (
                    not has_reactive_path
                    or not is_context_overflow(exc)
                    or recompact_attempts >= self._MAX_REACTIVE_COMPACT
                ):
                    raise
                recompact_attempts += 1
                journal.write(
                    "reactive_compact_triggered",
                    session=self._session_id,
                    turn=self._state.turn_count,
                    attempt=recompact_attempts,
                    error=str(exc),
                )
                if self._compactor is not None:
                    await self._compactor.reactive(
                        history, self._state.slots,
                    )
                else:
                    assert self._compact_callback is not None
                    await self._compact_callback(history)

        ai = await self._retry_on_length_truncation(ai, messages)
        history.append(ai)
        await self._hooks.run_post_model(
            ai_message=ai, history=history, state=self._state,
        )
        return ai

    async def _build_view(
        self, history: list[BaseMessage],
    ) -> list[BaseMessage]:
        # Context.build is the only message-assembly site; microcompact is
        # view-only — stored history stays full, only outgoing messages shrink.
        messages = self._context.build(history)
        if self._compactor is not None:
            return await self._compactor.microcompact(
                messages, self._state.slots,
            )
        if self._microcompact_policy is not None:
            mc_result = apply_microcompact(
                messages, self._microcompact_policy,
            )
            if mc_result.cleared_pair_count > 0:
                journal.write(
                    "microcompact_applied",
                    session=self._session_id,
                    turn=self._state.turn_count,
                    cleared_pair_count=mc_result.cleared_pair_count,
                    cleared_tool_call_ids=list(
                        mc_result.cleared_tool_call_ids,
                    ),
                    cleared_positions=[
                        [p.ai_idx, p.tool_idx]
                        for p in mc_result.cleared_pairs
                    ],
                )
                return mc_result.messages
        return messages

    async def _retry_on_length_truncation(
        self,
        ai: AIMessage,
        messages: list[BaseMessage],
    ) -> AIMessage:
        # Resume prompts append to the local view; the caller appends final ai.
        length_retries = 0
        while length_retries < _MAX_LENGTH_RETRY and _length_truncated(ai):
            length_retries += 1
            journal.write(
                "length_recovery",
                session=self._session_id,
                turn=self._state.turn_count,
                attempt=length_retries,
                max_attempts=_MAX_LENGTH_RETRY,
            )
            messages.append(ai)
            messages.append(HumanMessage(content=_LENGTH_RESUME_PROMPT))
            ai = await self._invoke_with_retry(messages)
        if not _length_truncated(ai):
            return ai
        journal.write(
            "length_recovery_exhausted",
            session=self._session_id,
            turn=self._state.turn_count,
            attempts=length_retries,
        )
        # Preserve partial content; strip tool_calls (they reference half-built
        # args and MUST NOT be dispatched). Sentinel finish_reason routes
        # ``run_turn`` to the length-exhaustion Final branch.
        return AIMessage(
            content=ai.content,
            response_metadata={
                **ai.response_metadata,
                "finish_reason": "length_recovery_exhausted",
            },
        )

    async def _dispatch_tool_calls(
        self, tool_calls: list[ToolCall], history: list[BaseMessage],
    ) -> AsyncIterator[AgentEvent]:
        # Three stages: plan → partition batches → execute.
        steps = await self._plan_tool_calls(tool_calls)
        journal.write(
            "tool_plan_built",
            turn=self._state.turn_count,
            steps=[
                {
                    "tool": s.tool_call.get("name"),
                    "short_circuited": s.decision is not None,
                }
                for s in steps
            ],
        )

        for batch in partition_batches(steps):
            async for event in self._run_batch(batch, history):
                yield event

    async def _run_batch(
        self, batch: list[ToolStep], history: list[BaseMessage],
    ) -> AsyncIterator[AgentEvent]:
        # Strict ordering: tool_call.id → ToolMessage alignment is load-bearing.
        for event in self._emit_started(batch):
            yield event

        progress_queue: asyncio.Queue[ProgressItem | None] = asyncio.Queue()
        batch_deadline = (
            self._batch_timeout_sec if self._batch_timeout_sec > 0 else 0.0
        )
        gather_task = asyncio.create_task(
            self._gather_all_with_progress(batch, progress_queue, batch_deadline),
        )
        # Sentinel via done_callback so drain exits without polling gather.done().
        gather_task.add_done_callback(lambda _: progress_queue.put_nowait(None))

        abort_signal = current_abort_signal.get()
        watchdog_task: asyncio.Task[None] | None = None
        if abort_signal is not None and not abort_signal.aborted:
            watchdog_task = asyncio.ensure_future(
                self._abort_watchdog(abort_signal, gather_task),
            )

        try:
            async for event in self._drain_progress(progress_queue):
                yield event
            results: list[ToolResult] = await gather_task
        finally:
            if not gather_task.done():
                gather_task.cancel()
                await asyncio.gather(gather_task, return_exceptions=True)
            if watchdog_task is not None and not watchdog_task.done():
                watchdog_task.cancel()

        for event in self._emit_completed(batch, results, history):
            yield event
        journal.write("tool_batch_end", turn=self._state.turn_count, size=len(batch))

    def _emit_started(
        self, batch: list[ToolStep],
    ) -> Iterator[AgentEvent]:
        journal.write(
            "tool_batch_begin",
            turn=self._state.turn_count,
            size=len(batch),
            tools=[s.tool_call.get("name") for s in batch],
        )
        for step in batch:
            tc = step.tool_call
            tc_id = str(tc.get("id") or "")
            yield ToolCallStarted(
                name=tc["name"],
                input=dict(tc["args"]),
                id=tc_id,
            )
            pd = step.permission_decision
            if pd is not None and pd.reason in _AUTO_ALLOW_REASONS:
                yield PermissionAudit(tool=tc["name"], text=pd.audit_line())
            journal.write(
                "tool_execute_begin", tool=tc["name"], tool_call_id=tc["id"],
            )

    def _emit_completed(
        self,
        batch: list[ToolStep],
        results: list[ToolResult],
        history: list[BaseMessage],
    ) -> Iterator[AgentEvent]:
        for step, result in zip(batch, results, strict=True):
            tc = step.tool_call
            tc_id = str(tc.get("id") or "")
            journal.write(
                "tool_execute_end",
                tool=tc["name"], tool_call_id=tc["id"],
                ok=result.ok, error=result.error,
            )
            status: Literal["success", "error"] = "success" if result.ok else "error"
            history.append(
                ToolMessage(
                    content=_serialize(result, tool_name=tc["name"]),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                    status=status,
                )
            )
            yield ToolCallCompleted(
                name=tc["name"],
                output=result.output,
                error=result.error,
                id=tc_id,
            )

    async def _drain_progress(
        self,
        progress_queue: asyncio.Queue[ProgressItem | None],
    ) -> AsyncIterator[AgentEvent]:
        while True:
            item = await progress_queue.get()
            if item is None:
                return
            tool_call_id, tool_name, stream_name, chunk = item
            assert stream_name in ("stdout", "stderr")
            yield ToolCallProgress(
                name=tool_name,
                stream=stream_name,
                chunk=chunk,
                id=tool_call_id,
            )

    async def _gather_all_with_progress(
        self,
        batch: list[ToolStep],
        progress_queue: asyncio.Queue[ProgressItem | None],
        batch_deadline: float,
    ) -> list[ToolResult]:
        def make_cb(tool_call_id: str, tool_name: str) -> ProgressCallback:
            def _cb(stream: Literal["stdout", "stderr"], chunk: str) -> None:
                progress_queue.put_nowait((tool_call_id, tool_name, stream, chunk))
            return _cb

        async def execute_one(step: ToolStep) -> ToolResult:
            # contextvars are task-local; each step owns its own set/reset.
            tool_call_id = str(step.tool_call.get("id") or "")
            token = set_progress_callback(
                make_cb(tool_call_id, step.tool_call["name"]),
            )
            try:
                return await self._execute_step(step)
            finally:
                reset_progress_callback(token)

        if batch_deadline <= 0:
            return list(await asyncio.gather(
                *(execute_one(s) for s in batch),
            ))
        # Per-step Task: cancel slow ones individually, keep finished results.
        # Position-aligned with ``batch`` — callers use zip(..., strict=True).
        tasks: list[asyncio.Task[ToolResult]] = [
            asyncio.create_task(execute_one(s)) for s in batch
        ]
        try:
            _, pending = await asyncio.wait(tasks, timeout=batch_deadline)
        except asyncio.CancelledError:
            # asyncio.wait doesn't auto-cancel waitees (unlike gather).
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if pending:
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        return await self._collect_batch_results(
            batch, tasks, pending, batch_deadline,
        )

    async def _collect_batch_results(
        self,
        batch: list[ToolStep],
        tasks: list[asyncio.Task[ToolResult]],
        pending: set[asyncio.Task[ToolResult]],
        batch_deadline: float,
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        cancelled_ids: list[str] = []
        completed_ids: list[str] = []
        for step, t in zip(batch, tasks, strict=True):
            tc_id = step.tool_call["id"] or ""
            if t in pending:
                synthesised = ToolResult(
                    ok=False,
                    error=f"batch timeout after {batch_deadline}s",
                )
                # Cancel path still runs post_tool so all consumers see a
                # uniform ToolResult shape.
                final: ToolResult
                if step.tool is not None and step.args is not None:
                    final = await self._hooks.run_post_tool(
                        tool=step.tool,
                        args=step.args,
                        result=synthesised,
                        state=self._state,
                    )
                else:
                    final = synthesised
                results.append(final)
                cancelled_ids.append(tc_id)
            else:
                results.append(t.result())
                completed_ids.append(tc_id)
        if cancelled_ids:
            journal.write(
                "batch_timeout",
                session=self._session_id,
                turn=self._state.turn_count,
                size=len(batch),
                timeout_sec=batch_deadline,
                cancelled_count=len(cancelled_ids),
                cancelled_tool_call_ids=cancelled_ids,
                completed_tool_call_ids=completed_ids,
            )
        return results

    async def _abort_watchdog(
        self,
        abort_signal: AbortController,
        gather_task: asyncio.Task[list[ToolResult]],
    ) -> None:
        # Cancel gather so run_turn synthesises tool messages on the way out.
        await abort_signal.signal.wait()
        if not gather_task.done():
            gather_task.cancel()

    async def _plan_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolStep]:
        steps: list[ToolStep] = []
        for tc in tool_calls:
            tool = self._registry.get(tc["name"])
            if tool is None:
                steps.append(ToolStep(
                    tool_call=tc, tool=None, args=None,
                    decision=ToolResult(ok=False, error=f"unknown tool: {tc['name']!r}"),
                ))
                continue

            # Validate args before pre_tool so hooks don't decide on bad input.
            raw_args = dict(tc["args"])
            schema = _validated_args_model(tool.args_schema)
            if schema is not None:
                try:
                    schema.model_validate(raw_args)
                except ValidationError as exc:
                    steps.append(ToolStep(
                        tool_call=tc, tool=tool, args=None,
                        decision=ToolResult(ok=False, error=f"invalid args: {exc}"),
                    ))
                    continue

            outcome = await self._hooks.run_pre_tool(
                tool=tool,
                args=raw_args,
                state=self._state,
                tool_call_id=tc["id"],
            )
            match outcome:
                case Allow(decision=perm_decision):
                    # Hook allowed; tool will run. Carry the decision for
                    # the PermissionAudit event.
                    steps.append(ToolStep(
                        tool_call=tc, tool=tool, args=raw_args,
                        decision=None,
                        permission_decision=perm_decision,
                    ))
                case Block(decision=perm_decision):
                    # Hook denied; inject synthetic error from audit_line().
                    steps.append(ToolStep(
                        tool_call=tc, tool=tool, args=raw_args,
                        decision=ToolResult(
                            ok=False, error=perm_decision.audit_line(),
                        ),
                        permission_decision=perm_decision,
                    ))
                case Ask():
                    # Well-formed chains resolve Ask in the hook; any leak = deny.
                    steps.append(ToolStep(
                        tool_call=tc, tool=tool, args=raw_args,
                        decision=ToolResult(
                            ok=False,
                            error="permission escalation unresolved — no asker hook present",
                        ),
                        permission_decision=None,
                    ))
                case Replace(result=sc_result, decision=perm_decision):
                    # Hook injected a synthetic result; tool is NOT invoked.
                    steps.append(ToolStep(
                        tool_call=tc, tool=tool, args=raw_args,
                        decision=sc_result,
                        permission_decision=perm_decision,
                    ))
        return steps

    async def _execute_step(self, step: ToolStep) -> ToolResult:
        if step.decision is not None:
            return step.decision
        assert step.tool is not None
        assert step.args is not None
        # Tools with their own timeout ladder (bash) set timeout_sec=None.
        timeout: float | None = meta_dict(step.tool).get("timeout_sec")
        try:
            if timeout is not None:
                try:
                    output = await asyncio.wait_for(
                        step.tool.ainvoke(step.args), timeout=timeout,
                    )
                except TimeoutError as exc:
                    raise ToolError(
                        f"tool {step.tool.name!r} timed out after {timeout}s"
                    ) from exc
            else:
                output = await step.tool.ainvoke(step.args)
            result = ToolResult(ok=True, output=output)
            self._maybe_trigger_path(step, result)
        except Exception as exc:  # noqa: BLE001  # tool boundary; CancelledError inherits BaseException so it skips this clause
            # ToolError messages stay verbatim; other types get a type prefix
            # so the model can distinguish programmer errors from user-facing ones.
            text = str(exc) if isinstance(exc, ToolError) else f"{type(exc).__name__}: {exc}"
            result = ToolResult(ok=False, error=text)
        return await self._hooks.run_post_tool(
            tool=step.tool, args=step.args, result=result, state=self._state
        )

    def _maybe_trigger_path(self, step: ToolStep, result: ToolResult) -> None:
        """Feed the touched path back into Context after a successful tool run.

        Uses ``result.output`` (not args) so read_file's ``partial`` flag is
        honoured even when ``limit >= total_lines``.
        """
        tool = step.tool
        args = step.args
        if tool is None or args is None:
            return
        arg_name = PATH_TRIGGER_TOOLS.get(tool.name)
        if arg_name is None:
            return
        raw = args.get(arg_name)
        if not isinstance(raw, str) or not raw:
            return
        try:
            resolved = Path(raw).resolve()
        except OSError:
            return
        self._context.on_tool_touched_path(resolved)
        # Record read_file targets so edit_file's must-read-first hook can verify.
        if tool.name == "read_file":
            partial = False
            if isinstance(result.output, dict):
                partial = bool(result.output.get("partial", False))
            self._context.record_read(resolved, partial=partial)
