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
    """Pre-tool hook — gates / observes one tool call, returns an
    :class:`Outcome` variant (``Allow`` / ``Block`` / ``Ask`` /
    ``Replace`` from :mod:`aura.schemas.permissions`).

    Returning ``None`` from a PreToolHook is a type error by design —
    every hook must return an :class:`Outcome` variant.

    Outcome variants and their semantics:

    - :class:`Allow` → tool runs; carries a :class:`Decision` for
      audit (``permission_decision`` on :class:`ToolStep`).
    - :class:`Block` → tool is denied; loop injects a synthetic
      ToolMessage with ``decision.audit_line()``.
    - :class:`Ask` → escalate to the asker; the permission hook
      resolves this by interacting with the user.
    - :class:`Replace` → tool is NOT invoked; a canned
      :class:`ToolResult` is injected directly.

    Merge precedence (spec §3.2): first ``Block`` wins → first
    ``Ask`` wins → first ``Replace`` wins → last ``Allow`` wins.
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


def _merge_outcomes(
    outcomes: list[Outcome],
    ask_requested: bool,
) -> Outcome:
    """Apply spec §3.2 precedence to a pure-Outcome chain, returning an
    :class:`Outcome` variant directly.

    Merge precedence:
      1. First Block wins (hard deny, short-circuits on sight in run_pre_tool).
      2. First Replace wins (synthetic result injection).
      3. First authoritative Allow wins (reason != ``"mode_bypass"``).
         Passthrough hooks return ``mode_bypass`` to signal "no opinion";
         the first hook with a real verdict (rule_allow, user_accept, …)
         is the canonical audit line. If all Allows are mode_bypass, the
         last one wins.

    ``Ask`` outcomes are NOT included in the precedence ladder here — they
    are handled as side-channel signals by ``run_pre_tool`` (setting
    ``ask_pending`` on ``LoopSlots``). A well-formed chain always has a
    permission hook that consumes the Ask signal and returns Allow/Block.
    If no hook resolved the Ask, the ``ask_requested`` flag causes the
    winning Allow to be wrapped in ``Ask(reason="pending escalation")``
    so the loop can detect the unresolved escalation.
    """
    # 1. First Block.
    for o in outcomes:
        if isinstance(o, Block):
            return o
    # 2. First Replace.
    for o in outcomes:
        if isinstance(o, Replace):
            return o
    # 3. First authoritative Allow (reason != "mode_bypass") wins.
    #    Passthrough hooks return Allow(reason="mode_bypass") to signal
    #    "no opinion". The first hook that carries a real decision
    #    (rule_allow, user_accept, …) is the authoritative verdict.
    #    If every Allow is mode_bypass, fall through to the last one.
    first_authoritative: Allow | None = None
    last_allow: Allow | None = None
    for o in outcomes:
        if isinstance(o, Allow):
            last_allow = o
            if first_authoritative is None:
                reason = getattr(getattr(o, "decision", None), "reason", "mode_bypass")
                if reason != "mode_bypass":
                    first_authoritative = o
    winner_allow = first_authoritative if first_authoritative is not None else last_allow
    if winner_allow is not None:
        if ask_requested:
            # Check whether the Ask was resolved by the asker. A well-formed
            # chain has a permission hook that sees ``ask_pending=True``,
            # calls the asker, and returns Allow with a user-driven reason
            # (``user_accept`` / ``user_always``). Only these reasons confirm
            # the asker was actually invoked; any mode- or rule-based reason
            # means the Ask was NOT resolved and should escalate.
            winner_reason = getattr(
                getattr(winner_allow, "decision", None), "reason", "mode_bypass"
            )
            _ASK_RESOLVED_REASONS = frozenset({"user_accept", "user_always"})
            if winner_reason not in _ASK_RESOLVED_REASONS:
                # Ask was NOT resolved — escalate so the loop detects it.
                # Return the first Ask from the chain to preserve its reason.
                for o in outcomes:
                    if isinstance(o, Ask):
                        return o
                return Ask(reason="pending escalation")
        return winner_allow
    # Fallback: only Ask outcomes (no Allow/Block/Replace). Return the
    # first Ask from the chain so its original reason is preserved.
    for o in outcomes:
        if isinstance(o, Ask):
            return o
    raise AssertionError(
        "_merge_outcomes called with empty outcomes list — "
        "caller should guard against this"
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
    ) -> Outcome:
        """Merge pre_tool hook outcomes across the chain (spec §3.2).

        Every hook MUST return an :class:`Outcome` variant. Merge
        precedence: first ``Block`` wins → first ``Ask`` wins → first
        ``Replace`` wins → last ``Allow`` wins.

        When the chain is empty or all hooks return :class:`Allow`, the
        last :class:`Allow` wins. When the chain is empty, the sentinel
        ``Allow(decision=Decision(allow=True, reason="mode_bypass"))``
        is returned so the loop always has an :class:`Outcome` to
        pattern-match on.

        Audit trail: every hook that carries a non-None ``decision``
        field triggers a ``pre_tool_hook_decision`` journal event (hook
        qualified name + tool + reason + allow flag) before merging.

        ``**kwargs`` forwards per-call metadata to hooks (e.g.
        ``tool_call_id`` for PermissionDenial records).
        """
        from aura.core.persistence import journal

        outcomes: list[Outcome] = []
        ask_requested = False
        prior_ask_pending = state.slots.ask_pending
        try:
            for hook in self.pre_tool:
                raw = await hook(tool=tool, args=args, state=state, **kwargs)
                outcomes.append(raw)

                # Track ask_pending so downstream hooks (permission) see it.
                if isinstance(raw, Ask) and not ask_requested:
                    ask_requested = True
                    state.slots = dataclasses.replace(
                        state.slots, ask_pending=True,
                    )

                # Emit per-hook audit event for Block, Ask, and Replace decisions —
                # these are non-trivial hook verdicts that need audit trail
                # reconstruction. Allow is the implicit default; the authoritative
                # allow decision is captured by the ``permission_decision`` event
                # in the loop after all hooks have run.
                if isinstance(raw, Block | Ask | Replace):
                    decision_attr = getattr(raw, "decision", None)
                    hook_name = (
                        f"{getattr(hook, '__module__', '')}."
                        f"{getattr(hook, '__qualname__', repr(hook))}"
                    ).lstrip(".")
                    journal.write(
                        "pre_tool_hook_decision",
                        hook=hook_name,
                        tool=tool.name,
                        allow=False if decision_attr is None else decision_attr.allow,
                        reason="" if decision_attr is None else decision_attr.reason,
                    )

                # Block is highest-priority; short-circuit immediately
                # so no later hook can override the first Block.
                if isinstance(raw, Block):
                    return raw

            if outcomes:
                return _merge_outcomes(outcomes, ask_requested)
            # Empty chain — return a sentinel that signals "no hook opinion".
            # ``chain_empty`` is an allow reason but is NOT in _AUTO_ALLOW_REASONS,
            # so the loop does not emit a PermissionAudit event when there is no
            # permission hook at all.
            return Allow(decision=Decision(allow=True, reason="chain_empty"))
        finally:
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
