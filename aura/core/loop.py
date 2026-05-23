"""AgentLoop — 协调一次对话的 turn 循环，驱动 model → tool → model 的迭代。"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from aura.application.memory.context import Context
from aura.application.permission.decision import Decision
from aura.config.schema import RetryConfig
from aura.domain.abort import AbortController, AbortException, current_abort_signal
from aura.domain.tool_registry import ToolRegistry
from aura.infrastructure.persistence import journal
from aura.infrastructure.retry import with_retry
from aura.schemas.events import (
    AgentEvent,
    AssistantDelta,
    Final,
    PermissionAudit,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)
from aura.schemas.permissions import Allow, Ask, Block, Replace
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolError, ToolResult
from aura.schemas.tool_meta_access import meta_dict
from aura.tools.errors import hint_for_error
from aura.tools.progress import (
    ProgressCallback,
    reset_progress_callback,
    set_progress_callback,
)

# Canonical default session name. Shared with ``aura.core.agent`` — keep a
# single source of truth so ``AgentLoop(session_id=)`` default and
# ``Agent(session_id=)`` default never drift. Any string literal ``"default"``
# elsewhere in the session_id pipeline is a bug.
DEFAULT_SESSION = "default"

# Reasons for which a permission prompt was NOT shown — the renderer surfaces
# an "auto-allowed: <reason>" dim line after ToolCallStarted. User-prompted
# allows/denies skip the audit (the prompt itself was the audit).
_AUTO_ALLOW_REASONS: frozenset[str] = frozenset(
    {"rule_allow", "mode_bypass"},
)

# Resolved once in __init__ so env flip mid-session does not race the outer wait.
_BATCH_TIMEOUT_ENV_VAR = "AURA_BATCH_TIMEOUT_SEC"
_DEFAULT_BATCH_TIMEOUT_SEC: float = 60.0

# Cap so a misbehaving provider that always returns length cannot loop forever.
_MAX_LENGTH_RETRY: int = 3

# ``length`` = OpenAI; ``max_tokens`` = Anthropic. Lowercased before compare.
_LENGTH_FINISH_REASONS: frozenset[str] = frozenset({"length", "max_tokens"})

_LENGTH_RESUME_PROMPT: str = (
    "(Output token limit hit. Resume directly from where you stopped, "
    "without repeating prior content.)"
)


def _length_truncated(ai: AIMessage) -> bool:
    """Detect a length-cutoff AIMessage across provider shapes.

    OpenAI surfaces ``finish_reason='length'`` inside ``response_metadata``;
    Anthropic surfaces ``stop_reason='max_tokens'`` either at the top
    level or inside ``response_metadata``. We probe all three and
    case-fold so SDK-version drift doesn't silently disable recovery.
    """
    meta = getattr(ai, "response_metadata", None) or {}
    candidates: list[object] = [
        meta.get("finish_reason"),
        meta.get("stop_reason"),
        getattr(ai, "stop_reason", None),
    ]
    return any(
        isinstance(raw, str) and raw.lower() in _LENGTH_FINISH_REASONS
        for raw in candidates
    )


def _resolve_batch_timeout(override: float | None) -> float:
    """Pick the effective batch wallclock deadline, in seconds.

    Precedence (highest first):

    1. Explicit ``override`` kwarg to :class:`AgentLoop` — the test suite
       injects sub-second deadlines so it can observe the cancel branch.
    2. ``AURA_BATCH_TIMEOUT_SEC`` environment variable.
    3. :data:`_DEFAULT_BATCH_TIMEOUT_SEC` (60s).

    Returns ``0.0`` when the resolved value is ``<= 0`` — the dispatch path
    treats "value <= 0" as "feature disabled" (parity with the ``0``
    disables pattern used elsewhere, e.g. ``auto_compact_threshold=0``).
    Malformed env strings fall through to the default rather than raising.
    """
    if override is not None:
        return override if override > 0 else 0.0
    raw = os.environ.get(_BATCH_TIMEOUT_ENV_VAR)
    if raw is not None:
        try:
            parsed = float(raw)
        except ValueError:
            # Malformed env → default. The loop constructor is a hot path;
            # raising here would break unrelated sessions on a typo.
            return _DEFAULT_BATCH_TIMEOUT_SEC
        return parsed if parsed > 0 else 0.0
    return _DEFAULT_BATCH_TIMEOUT_SEC


# 成功调用后需把路径反馈给 Context progressive 状态的工具 → 其 path 参数名。
# bash（shell 语义不固定）和 web_fetch（URL 而非文件系统）刻意排除。
PATH_TRIGGER_TOOLS: dict[str, str] = {
    "read_file": "path",
    "write_file": "path",
    "edit_file": "path",
    "grep": "path",
    "glob": "path",
}


def _serialize(result: ToolResult, *, tool_name: str = "") -> str:
    # `default=str` + `ensure_ascii=False`：遇到非 JSON-native 值
    # （datetime / Path / bytes）降级为字符串而非抛异常。
    # 这里抛出会导致那一条 ToolMessage 漏 append —— 破坏 tool_call.id 与
    # ToolMessage 的严格对齐。
    if result.ok:
        return json.dumps(result.output, default=str, ensure_ascii=False)
    # On error, append the SAME hint the UI renders so the model sees the
    # recovery guidance in its tool-result message — not just the user.
    # Without this, a failing grep/read/edit leaves the model guessing how
    # to recover while a red panel blinks at the human.
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
    # Permission decision captured directly from the pre_tool hook chain's
    # merged Outcome decision. Only populated when a permission
    # hook is installed AND the hook ran to a Decision; None otherwise.
    # Used to emit PermissionAudit after ToolCallStarted.
    permission_decision: Decision | None = None


def partition_batches(steps: list[ToolStep]) -> list[list[ToolStep]]:
    """将 steps 按并发安全性分批（保序，不重排）。

    1. 连续的 is_concurrency_safe 且 decision=None 的 step 合并成一个并行 batch，
       批内用 gather 一次并发执行并保序拿回结果。
    2. 非 safe 或已被 pre_tool 短路（decision 非 None）的 step 单独成 batch。
    3. 维持原 tool_call 顺序 —— 并发只发生在 batch 内，不跨 batch。
    """
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
    # One retry: a single compact didn't fit means repeated compacts won't either.
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
        # Raw model kept so _rebind_tools can rebind after MCP registers tools
        # — double-binding an already-bound model is unsupported on some providers.
        self._model = model
        # 空 registry 跳过 bind_tools：某些 provider 对 tools=[] 行为不一致。
        self._bound = model.bind_tools(registry.tools()) if len(registry) > 0 else model
        # 0.0 sentinel = disabled; otherwise _run_batch enforces deadline regardless of size.
        self._batch_timeout_sec = _resolve_batch_timeout(batch_timeout_sec)
        self._compact_callback = compact_callback
        self._compactor = compactor

    def _rebind_tools(self, tools: list[BaseTool]) -> None:
        """Rebind the loop's model with an updated tool set.

        Called by :meth:`Agent.aconnect` after MCP registers tools
        dynamically. Empty ``tools`` falls back to the raw model (matches
        the constructor's behaviour for an empty registry).
        """
        self._bound = self._model.bind_tools(tools) if tools else self._model

    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def max_turns(self) -> int | None:
        return self._max_turns

    async def run_turn(
        self,
        *,
        history: list[BaseMessage],
        abort: AbortController | None = None,
    ) -> AsyncIterator[AgentEvent]:
        # Contract: caller appends + persists the user's HumanMessage BEFORE
        # invoking — transcript ownership lives one layer up so a crash
        # mid-turn cannot erase the user's input.
        #
        # Per-turn sinks reset at the turn boundary, NOT inside the while loop:
        # multiple model rounds within the same user turn share one bucket.
        # List identity is stable so Agent.last_turn_denials sees the same object.
        self._state.slots.turn_denials.clear()
        self._state.slots.perm_dedup_cache.clear()
        # Install controller into contextvar so tools (and spawned subagents
        # on the same task tree) inherit the same signal.
        ctx_token = None
        if abort is not None:
            ctx_token = current_abort_signal.set(abort)
        try:
            while True:
                journal.write("turn_begin", turn=self._state.turn_count + 1)
                # Pre-invoke abort gate: a cancellation between turns
                # synthesises one consistent shutdown path instead of
                # racing the next ainvoke against the contextvar.
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
                        # Surface exhaustion as its own Final reason carrying
                        # the partial assistant text.
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

                # Mid-batch abort: synthesise one ToolMessage per unanswered
                # tool_call_id — provider 400's on a tool_use without matching
                # tool_result, so balance the trailing AIMessage on cancel.
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
        """Append a synthetic ToolMessage for every unanswered tool_call_id.

        On cancel, every ``tool_use`` block in the trailing AIMessage
        must be paired with a ``tool_result`` for the next provider
        request to validate. We append ``status="error"`` ToolMessages
        with a short "(aborted by user)" body so a subsequent astream
        sees a balanced history. Idempotent on already-answered ids.
        """
        for msg in reversed(history):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
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
        """Race ``_invoke_model`` against ``abort.signal``.

        - ``abort is None`` → passthrough (legacy / SDK call sites).
        - Abort fires first → cancel the in-flight invoke, raise
          :class:`AbortException` so the outer loop balances history.
        - Invoke completes first → cancel the watcher and return.
        """
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
        """Run ``self._bound.ainvoke(messages)`` through the retry policy.

        Pulled out so the call site doesn't define a closure over a loop
        variable (ruff B023). Wraps ONLY the SDK call — not tool dispatch,
        not hook run, not history.append — so retries are surgical and
        tool semantics stay untouched.
        """
        async def _do_invoke() -> AIMessage:
            return await self._bound.ainvoke(messages)

        return await with_retry(
            _do_invoke,
            max_attempts=self._retry_config.max_attempts,
            base_delay_s=self._retry_config.base_delay_s,
            max_delay_s=self._retry_config.max_delay_s,
        )

    async def _invoke_model(self, history: list[BaseMessage]) -> AIMessage:
        # turn_count 先于 pre_model hook 递增，hook 看到的是"即将开始的第 N 轮"。
        self._state.turn_count += 1
        await self._hooks.run_pre_model(history=history, state=self._state)
        from aura.core.agent import (
            _is_context_overflow,  # noqa: PLC0415  避免循环导入  # deferred import is intentional
        )

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
                    or not _is_context_overflow(exc)
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
        # Context.build 是组装 messages 的唯一构造点；microcompact 是 view-only：
        # stored history 保留全量，只压缩出参 messages。
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
        # messages 是 local view（含 microcompact）—— resume prompts append 在这里
        # 让下一次 ainvoke 看到 partial AIMessage；最终返回的 ai 由 caller append 到 history。
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
                **(getattr(ai, "response_metadata", None) or {}),
                "finish_reason": "length_recovery_exhausted",
            },
        )

    async def _dispatch_tool_calls(
        self, tool_calls: list[ToolCall], history: list[BaseMessage],
    ) -> AsyncIterator[AgentEvent]:
        # plan（解析 + pre_tool hook）→ partition batches → execute：三段式分离关注点。
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
        # 保序：Started 在 gather 之前同步 yield，Completed 按 batch 顺序 yield ——
        # tool_call.id 与 ToolMessage 严格一一对齐才能让 provider 正确串联。
        for event in self._emit_started(batch):
            yield event

        progress_queue: asyncio.Queue[
            tuple[str, str, str, str] | None
        ] = asyncio.Queue()
        batch_deadline = (
            self._batch_timeout_sec if self._batch_timeout_sec > 0 else 0.0
        )
        gather_task = asyncio.create_task(
            self._gather_all_with_progress(batch, progress_queue, batch_deadline),
        )
        # gather 完成时 done_callback 推 None 哨兵，drain 据此退出，不竞争 gather.done()。
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
        progress_queue: asyncio.Queue[tuple[str, str, str, str] | None],
    ) -> AsyncIterator[AgentEvent]:
        while True:
            item = await progress_queue.get()
            if item is None:
                return
            tool_call_id, tool_name, stream_name, chunk = item
            assert stream_name in ("stdout", "stderr")
            yield ToolCallProgress(
                name=tool_name,
                stream=stream_name,  # type: ignore[arg-type]  # deliberately off-type arg to exercise path
                chunk=chunk,
                id=tool_call_id,
            )

    async def _gather_all_with_progress(
        self,
        batch: list[ToolStep],
        progress_queue: asyncio.Queue[tuple[str, str, str, str] | None],
        batch_deadline: float,
    ) -> list[ToolResult]:
        def make_cb(tool_call_id: str, tool_name: str) -> ProgressCallback:
            def _cb(stream: Literal["stdout", "stderr"], chunk: str) -> None:
                progress_queue.put_nowait((tool_call_id, tool_name, stream, chunk))
            return _cb

        async def execute_one(step: ToolStep) -> ToolResult:
            # contextvars 是 task-local：每个 step 独立设置/重置 callback。
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
        # Per-step Task 便于单独取消慢任务并保留已完成结果；batch 顺序 load-bearing：
        # 调用方 ``zip(..., strict=True)`` 依赖位置对齐。
        tasks: list[asyncio.Task[ToolResult]] = [
            asyncio.create_task(execute_one(s)) for s in batch
        ]
        try:
            _, pending = await asyncio.wait(tasks, timeout=batch_deadline)
        except asyncio.CancelledError:
            # asyncio.wait 不会自动取消 waitee（与 gather 不同），需手动传播
            # 取消信号，否则后台任务连同它们 park 的 sleep 会继续跑。
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
                # 取消路径同样走 post_tool，保证 size-budget / logger 等
                # consumer 看到统一的 ToolResult 形状。
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
        # AbortController fire 时取消 gather_task：让 CancelledError 抛到 run_turn，
        # 由 run_turn 用合成 ToolMessage 平衡掉孤立的 trailing AIMessage。
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

            # 早校验：在派发前把 pydantic 错误转成 decision 短路 —— pre_tool hook
            # 不该看到 invalid args（否则 permission/budget 基于错误假设做决策）。
            raw_args = dict(tc["args"])
            schema = tool.args_schema
            if isinstance(schema, type) and issubclass(schema, BaseModel):
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
            # run_pre_tool 总返回 Outcome 的四种变体之一。
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
                    # Defensive：well-formed permission chain 应在 hook 里就 resolve Ask；
                    # 漏到 loop 视为 deny。
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
        # Tools 拥有自己 timeout ladder（bash）时把 timeout_sec 设 None 避免叠加。
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
        except Exception as exc:  # noqa: BLE001  # swallowed at boundary; failure must not propagate
            # ToolError 是工具作者主动抛的用户态消息（保留原文）；其余异常 type-prefix
            # 让模型能区分编程错误与用户态错误。CancelledError 继承 BaseException 不会进来。
            text = str(exc) if isinstance(exc, ToolError) else f"{type(exc).__name__}: {exc}"
            result = ToolResult(ok=False, error=text)
        return await self._hooks.run_post_tool(
            tool=step.tool, args=step.args, result=result, state=self._state
        )

    def _maybe_trigger_path(self, step: ToolStep, result: ToolResult) -> None:
        """成功的 path-aware tool 调用后，把路径反馈给 Context 的 progressive 状态。

        仅在 `_execute_step` 的成功分支（decision 为 None + ainvoke 未抛）被调用，
        因此 `step.tool` 与 `step.args` 必非 None —— 由 `_plan_tool_calls` 保证。

        `result.output` 传进来是为了拿到 read_file 返回的 `partial` 标志 ——
        必须用工具返回值而非 args（`limit >= total_lines` 也可能是 full read）。
        """
        # narrowed by assert above; mypy keeps union
        arg_name = PATH_TRIGGER_TOOLS.get(step.tool.name)  # type: ignore[union-attr]
        if arg_name is None:
            return
        raw = step.args.get(arg_name)  # type: ignore[union-attr]  # narrowed by assert above; mypy keeps union
        if not isinstance(raw, str) or not raw:
            return
        try:
            resolved = Path(raw).resolve()
        except OSError:
            return
        self._context.on_tool_touched_path(resolved)
        # Must-read-first invariant: record successful read_file targets so
        # edit_file (via make_must_read_first_hook) can verify a prior read.
        # `partial` fallbacks to False for robustness — a custom read_file
        # tool without the key is treated as a full read.
        if step.tool.name == "read_file":  # type: ignore[union-attr]
            partial = False
            if isinstance(result.output, dict):
                partial = bool(result.output.get("partial", False))
            self._context.record_read(resolved, partial=partial)
