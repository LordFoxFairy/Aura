"""4 turn-cycle hook Protocols + HookChain — **kwargs: Any stays for forward compat.

4 turn-cycle hooks: ``pre_model`` / ``post_model`` / ``pre_tool`` /
``post_tool``. ``pre_tool`` / ``post_tool`` gate or shape the tool call;
``pre_model`` may mutate history (compact, inject system message);
``post_model`` is observational.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from aura.core.permissions.decision import Decision
from aura.schemas.permissions import Allow, Ask, Block, Outcome, Replace
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult

FileChangeKind = Literal["created", "modified", "deleted"]

# F-04-014 (Round 5H): lifecycle event discriminators.
NotificationKind = Literal["permission_prompt", "ask_user", "error"]
StopReason = Literal["user_exit", "clear", "max_turns", "error"]


@dataclass(frozen=True)
class PreToolOutcome:
    """What a :class:`PreToolHook` returns — three independent channels.

    - ``short_circuit``: non-None replaces the tool's own execution with
      this :class:`ToolResult`. The tool is NOT invoked; the result lands
      in history as a ToolMessage. Used by safety / budget / must-read
      hooks to block a call without consulting the model.
    - ``decision``: non-None carries a permission :class:`Decision` up to
      :class:`aura.core.loop.AgentLoop._plan_tool_calls` which stamps it
      onto the :class:`aura.core.loop.ToolStep` so a PermissionAudit can
      be emitted between ToolCallStarted and ToolCallCompleted. A hook
      that has nothing to say about permission returns ``decision=None``.
    - ``ask``: when True, demote any pending auto-allow back to the
      user-prompt path. Lets a PreToolHook (e.g. a custom audit hook)
      override an upstream auto-allow rule and force a fresh user
      confirmation. Merge precedence is ``deny > ask > allow``: a deny
      cannot be promoted back to a prompt, but an auto-allow can be.

    All three fields default to passthrough — a hook that has nothing to
    do returns :obj:`PRE_TOOL_PASSTHROUGH` (or equivalently
    ``PreToolOutcome()``). Returning an implicit ``None`` from a
    PreToolHook is a type error by design — the contract is strict so
    future readers can rely on "every hook returns an outcome".

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
    ask: bool = False


# Shared sentinel for the "nothing to do" case so common hooks don't each
# allocate a fresh PreToolOutcome on every call. Frozen dataclass = safe
# to share by reference.
PRE_TOOL_PASSTHROUGH: PreToolOutcome = PreToolOutcome()

# The ``ask`` channel signal lives on the typed
# ``state.slots.ask_pending`` slot (Phase 1 Task 6). Set to True by
# ``HookChain.run_pre_tool`` while the chain has an unresolved ask
# request, so a downstream permission hook can detect "an earlier hook
# said ask" and route the call into its asker rather than an auto-allow
# branch. Restored to the prior value at the end of every
# ``run_pre_tool`` — never leaks across tool calls.
#
# The legacy constant is kept as a back-compat re-export for tests that
# imported it directly (``from aura.core.hooks import
# PRE_TOOL_ASK_PENDING_KEY``); it's now a documentation string that
# points at the slot, not a live key.
PRE_TOOL_ASK_PENDING_KEY = "_pre_tool_ask_pending"


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
    """Pre-tool hook — gates / observes one tool call, returns
    PreToolOutcome (legacy) OR an :class:`Outcome` variant (Phase 1).

    The Protocol's declared return type stays :class:`PreToolOutcome`
    for the Task 8-10 migration window — this keeps every legacy call
    site mypy-clean. New hook authors targeting the Phase-1 tagged
    union (``Allow`` / ``Block`` / ``Ask`` / ``Replace`` from
    :mod:`aura.schemas.permissions`) declare their own return type as
    :class:`aura.schemas.permissions.Outcome` and pass the function
    through ``cast(PreToolHook, fn)`` (or use ``Any``-typed module-
    level lists) when registering. :meth:`HookChain.run_pre_tool`
    accepts both shapes at runtime via :func:`isinstance` dispatch
    on the four Outcome variants. Task 10 will flip the Protocol
    return type to ``Outcome`` outright once every built-in hook
    has migrated.

    Returning ``None`` from a PreToolHook is a type error by design —
    a passthrough hook MUST return :data:`PRE_TOOL_PASSTHROUGH` (or
    equivalently ``PreToolOutcome()``) so the merge contract stays
    explicit.

    Legacy channels (PreToolOutcome):

    - ``outcome.short_circuit`` non-None → tool is NOT invoked; this
      :class:`ToolResult` becomes the tool's result.
    - ``outcome.decision`` non-None → permission :class:`Decision`
      surfaces on :class:`aura.core.loop.ToolStep.permission_decision`
      for the auditor to emit a PermissionAudit event.

    Both channels are independent. A permission hook typically sets
    BOTH (allow-with-reason: ``decision=allow, short_circuit=None``;
    deny: ``decision=deny, short_circuit=ToolResult(ok=False,...)``).

    Outcome merge precedence: when every hook in a single chain run
    returns an :class:`Outcome` variant, :meth:`HookChain.run_pre_tool`
    applies spec §3.2 — first ``Block`` wins → first ``Ask`` wins →
    first ``Replace`` wins → last ``Allow`` wins. Mixed chains fall
    back to legacy merge logic so existing built-in hooks keep
    working unchanged.
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


class FileChangedHook(Protocol):
    """Fires when a file Aura is watching changes on disk.

    Producer: :class:`aura.core.hooks.file_watcher.FileWatcher`.
    Live-reload of project memory / rules / skills happens via consumers
    of this hook (see :mod:`aura.core.hooks.auto_reload`); the hook
    surface lets SDK callers add their own watchers without forking
    the agent's wiring.
    """

    async def __call__(
        self,
        *,
        path: Path,
        kind: FileChangeKind,
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


class CwdChangedHook(Protocol):
    """Fires when the Agent's working directory changes mid-session.

    Producer: :meth:`aura.core.agent.Agent.set_cwd`. External
    ``os.chdir`` calls are NOT observed — the contract is "the Agent
    knows because the Agent moved itself". Consumers refresh project
    memory + rules from the new cwd (see
    :mod:`aura.core.hooks.auto_reload`).
    """

    async def __call__(
        self,
        *,
        old_cwd: Path,
        new_cwd: Path,
        state: LoopState,
        **kwargs: Any,
    ) -> None: ...


# F-04-014 (Round 5H) lifecycle Protocols.

@dataclass(frozen=True)
class UserPromptSubmitOutcome:
    """Return value of a :class:`UserPromptSubmitHook`.

    ``prompt`` is the rewritten user message. ``None`` (default) is
    passthrough; ``str`` replaces the prompt for downstream hooks AND
    for the model. The chain is left-to-right composing.
    """
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


def _outcome_to_pretooloutcome(outcome: Outcome) -> PreToolOutcome:
    """Translate an :class:`Outcome` variant into the equivalent
    legacy :class:`PreToolOutcome` shape.

    Used during the Task 8-10 migration window so the existing legacy
    merge inside :meth:`HookChain.run_pre_tool` (first-deny-wins on
    decisions, first-wins on short-circuit, ask side-channel) can
    absorb new-style hook returns alongside legacy ones — without
    forcing a big-bang switchover. After Task 10 deletes
    :class:`PreToolOutcome` this helper goes with it.

    Mapping:

    - :class:`Allow` → ``PreToolOutcome(decision=allow_decision)``
    - :class:`Block` → ``PreToolOutcome(decision=deny_decision)`` —
      Block carries a deny decision; the loop turns the deny into a
      synthetic ToolMessage downstream, so we don't pre-fill
      ``short_circuit`` here.
    - :class:`Ask`   → ``PreToolOutcome(ask=True)`` — the asker
      escalation runs on the ``ask`` side-channel exactly like today.
    - :class:`Replace` → ``PreToolOutcome(short_circuit=result,
      decision=deny_decision)`` — Replace IS the canonical short-
      circuit shape (canned ToolResult + deny audit), straight 1:1.
    """
    match outcome:
        case Allow(decision=d):
            return PreToolOutcome(short_circuit=None, decision=d, ask=False)
        case Block(decision=d):
            return PreToolOutcome(short_circuit=None, decision=d, ask=False)
        case Ask():
            return PreToolOutcome(short_circuit=None, decision=None, ask=True)
        case Replace(result=r, decision=d):
            return PreToolOutcome(short_circuit=r, decision=d, ask=False)
    # Defensive — should be unreachable since Outcome is a closed union.
    raise TypeError(f"unknown Outcome variant: {type(outcome).__name__}")


def _merge_outcomes_to_pretooloutcome(
    outcomes: list[Outcome],
    ask_requested: bool,
) -> PreToolOutcome:
    """Apply spec §3.2 precedence to a list of :class:`Outcome` returns.

    Precedence (registration order matters):

    1. **First Block wins.** Returned as ``PreToolOutcome(decision=
       block.decision)`` — the loop turns the deny into a synthetic
       ToolMessage; no short-circuit ToolResult is pre-baked.
    2. **First Ask wins.** Returned as ``PreToolOutcome(ask=True)`` —
       loop escalates to the asker, which produces the real Decision.
    3. **First Replace wins.** Returned with ``short_circuit=result``
       and ``decision=replace.decision`` — Replace is the canonical
       short-circuit shape.
    4. **Last Allow wins.** Lets a refining hook (e.g. permission)
       overwrite an upstream allow's reason for the audit trail.

    ``ask_requested`` is the merged ask side-channel (which is True
    iff at least one :class:`Ask` was seen earlier in iteration —
    threaded through so downstream consumers reading
    ``PreToolOutcome.ask`` see the same flag whether or not Ask
    happened to be the precedence winner).
    """
    # 1. First Block.
    for o in outcomes:
        if isinstance(o, Block):
            return PreToolOutcome(
                short_circuit=None,
                decision=o.decision,
                ask=ask_requested,
            )
    # 2. First Ask.
    for o in outcomes:
        if isinstance(o, Ask):
            return PreToolOutcome(
                short_circuit=None,
                decision=None,
                ask=True,
            )
    # 3. First Replace.
    for o in outcomes:
        if isinstance(o, Replace):
            return PreToolOutcome(
                short_circuit=o.result,
                decision=o.decision,
                ask=ask_requested,
            )
    # 4. Last Allow.
    last_allow: Allow | None = None
    for o in outcomes:
        if isinstance(o, Allow):
            last_allow = o
    if last_allow is not None:
        return PreToolOutcome(
            short_circuit=None,
            decision=last_allow.decision,
            ask=ask_requested,
        )
    # Empty list — caller should have skipped this path; treat as
    # passthrough rather than raise so a future refactor can't
    # accidentally crash a live REPL.
    return PreToolOutcome(
        short_circuit=None,
        decision=None,
        ask=ask_requested,
    )


@dataclass
class HookChain:
    pre_model: list[PreModelHook] = field(default_factory=list)
    post_model: list[PostModelHook] = field(default_factory=list)
    pre_tool: list[PreToolHook] = field(default_factory=list)
    post_tool: list[PostToolHook] = field(default_factory=list)
    # v0.14 V14-HOOK-CATALOG: out-of-band hooks that don't sit on the
    # turn cycle. ``file_changed`` is fired by FileWatcher; ``cwd_changed``
    # by Agent.set_cwd. Distinct lists so registration / merge stays
    # symmetric with the four turn-cycle slots.
    file_changed: list[FileChangedHook] = field(default_factory=list)
    cwd_changed: list[CwdChangedHook] = field(default_factory=list)
    # F-04-014 (Round 5H) lifecycle slots.
    session_start: list[SessionStartHook] = field(default_factory=list)
    user_prompt_submit: list[UserPromptSubmitHook] = field(default_factory=list)
    notification: list[NotificationHook] = field(default_factory=list)
    stop: list[StopHook] = field(default_factory=list)

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

        A hook MAY return either the legacy three-channel
        :class:`PreToolOutcome` OR a Phase-1 :class:`Outcome` variant
        (``Allow`` / ``Block`` / ``Ask`` / ``Replace`` from
        :mod:`aura.schemas.permissions`). Both are accepted during the
        Task 8-10 migration window. Once every hook in a single chain
        run returns :class:`Outcome`, the new merge precedence applies
        (spec §3.2): first ``Block`` wins → first ``Ask`` wins → first
        ``Replace`` wins → last ``Allow`` wins. As soon as one hook
        returns a legacy :class:`PreToolOutcome`, the chain falls back
        to the legacy merge logic below so existing built-in hooks
        (Task 9 will migrate them) keep working unchanged.

        Legacy merge semantics (three channels, three intents):

        - ``short_circuit`` is **first-wins**. The first hook that emits
          a non-None ``short_circuit`` stops the chain immediately —
          subsequent hooks are NOT called. Matches the pre-G4 contract:
          safety / budget denials take precedence over anything further
          down the chain, and once a ToolResult is decided there's
          nothing for later hooks to add.
        - ``decision`` is **first-deny-wins**. As soon as any hook
          returns a deny decision (``allow=False``), that decision is
          locked in: later hooks may continue to run (e.g. to populate
          their own short-circuits) but cannot promote the merged
          decision back to allow. If no hook ever denies, the **last
          non-None allow** wins — matching the typical chain shape
          where the permission hook is last and gets to stamp its
          ``rule_allow`` / ``mode_bypass`` reason as the authoritative
          allow audit trail.
        - ``ask`` is **deny > ask > allow**. Once any hook sets
          ``ask=True``, downstream hooks see
          ``state.slots.ask_pending=True`` (Phase 1 Task 6) so a
          permission hook can force-prompt instead of auto-allowing.
          A subsequent deny still wins over the ask; an allow CANNOT
          erase a pending ask.

        Audit trail: every hook that emits a non-None decision also
        triggers a ``pre_tool_hook_decision`` journal event (with the
        hook's qualified name + tool + reason + allow flag) BEFORE
        merging. The downstream ``permission_decision`` event in the
        loop continues to carry the merged outcome — together they let
        the auditor reconstruct the full chain, not just the final
        result.

        ``**kwargs`` forwards per-call metadata to hooks. Today
        ``tool_call_id`` flows through (G5 uses it to stamp
        PermissionDenial records); additional keys land as more
        workstreams plumb data into the hook surface without having to
        revise every call site. :class:`PreToolHook` accepts
        ``**kwargs`` by protocol so unchanged hooks silently ignore
        new keys.
        """
        # Lazy-imported: hook chain is core-loaded; the journal module
        # pulls in storage/config and shouldn't sit on the import path
        # of the hook protocols themselves.
        from aura.core.persistence import journal

        merged_decision: Decision | None = None
        merged_decision_locked = False  # set True once a deny is recorded
        ask_requested = False
        # Collect every Outcome variant a hook returned this run so the
        # spec-§3.2 precedence merge can fire at the end. ``None`` means
        # the chain saw at least one legacy ``PreToolOutcome`` and the
        # collected list is no longer authoritative — fall through to
        # the legacy merge result instead. The list is kept in
        # registration order so "first Block/Ask/Replace wins" is
        # well-defined.
        outcomes: list[Outcome] | None = []
        # Snapshot the prior value so we can restore it after this run
        # — state.slots is shared across turns and we must not leak
        # ``ask_pending`` into the next tool call. The signal lives on
        # the typed ``slots.ask_pending`` (Phase 1 Task 6).
        prior_ask_pending = state.slots.ask_pending
        try:
            for hook in self.pre_tool:
                raw = await hook(tool=tool, args=args, state=state, **kwargs)
                # Normalize: an Outcome variant is converted into the
                # equivalent PreToolOutcome shape so the legacy merge
                # below can absorb it uniformly. The pure-Outcome chain
                # path (``outcomes is not None``) records the original
                # variant separately so it can apply spec-§3.2
                # precedence at the end.
                if isinstance(raw, Allow | Block | Ask | Replace):
                    if outcomes is not None:
                        outcomes.append(raw)
                    outcome = _outcome_to_pretooloutcome(raw)
                else:
                    # Legacy PreToolOutcome (or anything else that
                    # quacks like one) — downgrade the chain to the
                    # legacy merge path by clearing the Outcome bag.
                    outcomes = None
                    outcome = raw
                if outcome.ask and not ask_requested:
                    ask_requested = True
                    # Make the ask flag visible to downstream hooks (the
                    # permission hook is typically last) so it can demote
                    # an auto-allow to the asker path within the SAME
                    # chain run, not on a follow-up turn.
                    state.slots = dataclasses.replace(
                        state.slots, ask_pending=True,
                    )
                if outcome.decision is not None:
                    # Per-hook audit event — fires for every non-None
                    # decision the chain produces, regardless of merge
                    # outcome, so audit readers can reconstruct what each
                    # hook said even when one was overridden by a later
                    # hook in the merge.
                    hook_name = (
                        f"{getattr(hook, '__module__', '')}."
                        f"{getattr(hook, '__qualname__', repr(hook))}"
                    ).lstrip(".")
                    journal.write(
                        "pre_tool_hook_decision",
                        hook=hook_name,
                        tool=tool.name,
                        allow=outcome.decision.allow,
                        reason=outcome.decision.reason,
                    )
                    # First-deny-wins: a deny locks the merged decision so
                    # a later hook's allow cannot silently override it.
                    # Among allows (no deny seen yet) we keep last-wins
                    # behavior so the permission hook (typically last) gets
                    # to stamp its reason.
                    if not merged_decision_locked:
                        merged_decision = outcome.decision
                        if not outcome.decision.allow:
                            merged_decision_locked = True
                # Pure-Outcome chains short-circuit only on the first
                # :class:`Block` per spec §3.2: Block is the highest-
                # precedence variant so once seen no later hook can
                # override it. :class:`Replace` / :class:`Ask` /
                # :class:`Allow` do NOT stop the chain — a downstream
                # Block must still be allowed to win, so we keep
                # iterating and let the precedence merge run at the
                # end of the chain. (Legacy short-circuit semantics
                # are restored below for non-Outcome chains.)
                if outcomes is not None and isinstance(raw, Block):
                    return _merge_outcomes_to_pretooloutcome(
                        outcomes, ask_requested,
                    )
                # Legacy first-wins: stop immediately on the first
                # short-circuit when the chain is in legacy mode.
                if outcome.short_circuit is not None and outcomes is None:
                    return PreToolOutcome(
                        short_circuit=outcome.short_circuit,
                        decision=merged_decision,
                        ask=ask_requested,
                    )
            if outcomes is not None and outcomes:
                # Pure-Outcome chain — apply spec §3.2 precedence:
                # first Block wins → first Ask wins → first Replace
                # wins → last Allow wins.
                return _merge_outcomes_to_pretooloutcome(
                    outcomes, ask_requested,
                )
            return PreToolOutcome(
                short_circuit=None,
                decision=merged_decision,
                ask=ask_requested,
            )
        finally:
            # Always restore the prior value so the ask flag never leaks
            # across run_pre_tool invocations.
            if state.slots.ask_pending != prior_ask_pending:
                state.slots = dataclasses.replace(
                    state.slots, ask_pending=prior_ask_pending,
                )

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
        """Fan out a FileChanged event to every registered consumer.

        Each consumer is awaited sequentially; any exception is allowed
        to surface so a buggy reload hook is loud rather than silent.
        Order = registration order, matching the turn-cycle runners.
        """
        for hook in self.file_changed:
            await hook(path=path, kind=kind, state=state)

    async def run_cwd_changed(
        self,
        *,
        old_cwd: Path,
        new_cwd: Path,
        state: LoopState,
    ) -> None:
        """Fan out a CwdChanged event to every registered consumer."""
        for hook in self.cwd_changed:
            await hook(old_cwd=old_cwd, new_cwd=new_cwd, state=state)

    # ------------------------------------------------------------------
    # F-04-014 lifecycle runners — exception-isolated so a broken hook
    # MUST NOT take down the lifecycle event.
    # ------------------------------------------------------------------

    async def run_session_start(
        self,
        *,
        session_id: str,
        mode: str,
        cwd: Path,
        model_name: str,
        state: LoopState,
    ) -> None:
        from aura.core.persistence import journal
        for hook in self.session_start:
            try:
                await hook(
                    session_id=session_id,
                    mode=mode,
                    cwd=cwd,
                    model_name=model_name,
                    state=state,
                )
            except Exception as exc:  # noqa: BLE001
                journal.write(
                    "lifecycle_hook_error",
                    slot="session_start",
                    detail=f"{type(exc).__name__}: {exc}",
                )

    async def run_user_prompt_submit(
        self,
        *,
        session_id: str,
        turn_count: int,
        user_text: str,
        state: LoopState,
    ) -> str:
        """Compose the user_prompt_submit chain left-to-right.

        Each hook sees the previous hook's output as ``user_text``. A
        non-None ``UserPromptSubmitOutcome.prompt`` rewrites; ``None``
        passes through. A hook that raises has its outcome discarded —
        the previous prompt survives.
        """
        from aura.core.persistence import journal
        current = user_text
        for hook in self.user_prompt_submit:
            try:
                outcome = await hook(
                    session_id=session_id,
                    turn_count=turn_count,
                    user_text=current,
                    state=state,
                )
            except Exception as exc:  # noqa: BLE001
                journal.write(
                    "lifecycle_hook_error",
                    slot="user_prompt_submit",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                continue
            if outcome is not None and outcome.prompt is not None:
                current = outcome.prompt
        return current

    async def run_notification(
        self,
        *,
        session_id: str,
        kind: NotificationKind,
        body: str,
        state: LoopState,
    ) -> None:
        from aura.core.persistence import journal
        for hook in self.notification:
            try:
                await hook(
                    session_id=session_id,
                    kind=kind,
                    body=body,
                    state=state,
                )
            except Exception as exc:  # noqa: BLE001
                journal.write(
                    "lifecycle_hook_error",
                    slot="notification",
                    detail=f"{type(exc).__name__}: {exc}",
                )

    async def run_stop(
        self,
        *,
        session_id: str,
        reason: StopReason,
        turn_count: int,
        state: LoopState,
    ) -> None:
        from aura.core.persistence import journal
        for hook in self.stop:
            try:
                await hook(
                    session_id=session_id,
                    reason=reason,
                    turn_count=turn_count,
                    state=state,
                )
            except Exception as exc:  # noqa: BLE001
                journal.write(
                    "lifecycle_hook_error",
                    slot="stop",
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def merge(self, other: HookChain) -> HookChain:
        # 非破坏性拼接：self 优先 other 后；不修改任何一方的原始列表。
        return HookChain(
            pre_model=[*self.pre_model, *other.pre_model],
            post_model=[*self.post_model, *other.post_model],
            pre_tool=[*self.pre_tool, *other.pre_tool],
            post_tool=[*self.post_tool, *other.post_tool],
            file_changed=[*self.file_changed, *other.file_changed],
            cwd_changed=[*self.cwd_changed, *other.cwd_changed],
            session_start=[*self.session_start, *other.session_start],
            user_prompt_submit=[
                *self.user_prompt_submit, *other.user_prompt_submit,
            ],
            notification=[*self.notification, *other.notification],
            stop=[*self.stop, *other.stop],
        )
