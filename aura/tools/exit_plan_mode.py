"""exit_plan_mode — leave plan mode after the user approves the plan.

Companion to ``enter_plan_mode``. Matches claude-code's
``ExitPlanModeV2Tool`` semantics: the model signals "planning is done;
resume executing", the USER sees the plan and either approves (→ mode
flips) or rejects (→ mode stays ``plan`` so the model can iterate).

The user-approval gate is the whole point of plan mode — without it the
LLM could call ``exit_plan_mode`` with any plan and walk itself out of
the dry-run sandbox unilaterally. The tool therefore:

1. Raises ``ToolError`` if not currently in plan mode (category error).
2. Calls the injected ``asker`` ("Exit plan mode and accept this plan?"
   + rendered plan) BEFORE mutating state.
3. Flips the mode only on explicit approval; on denial, returns a
   ``ToolError`` so the model sees the rejection and can revise.

Not destructive, not read-only (meta-tool). Deliberately exempted from
plan-mode enforcement in ``aura.core.hooks.permission`` — the tool is
*callable* from plan mode; its OWN user-confirmation step is the gate.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.schemas.tool import ToolError, tool_metadata
from aura.tools.ask_user import QuestionAsker
from aura.tools.enter_plan_mode import ModeGetter, ModeSetter

# Modes the tool is allowed to transition *into*. Deliberately narrow:
# ``bypass`` is intentionally excluded — plan → bypass would be a
# security surprise, and ``bypass`` can only be entered via the CLI
# ``--bypass-permissions`` flag. ``plan`` itself is also excluded (would
# be a no-op; use ``enter_plan_mode`` instead).
ExitTarget = Literal["default", "accept_edits"]

# Approval prompt — concise, echoes claude-code's "Exit plan mode?". The
# rendered plan is appended to the question text so the user sees WHAT
# they're approving, not just a yes/no in a vacuum.
_APPROVAL_PROMPT = "Exit plan mode and accept this plan?"

# Two-choice picker: "Yes" approves, "No" rejects. Default "No" so a
# stray Enter keeps the model in plan mode (fail-safe). Matches the
# "explicit consent required" contract.
_APPROVAL_OPTIONS = ["Yes", "No"]
_APPROVAL_DEFAULT = "No"


class ExitPlanModeParams(BaseModel):
    plan: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description=(
            "The plan the model proposes for user approval. Markdown OK. "
            "Shown to the user in the approval prompt; echoed back in the "
            "tool result on approval so downstream turns can reference it."
        ),
    )
    to_mode: ExitTarget = Field(
        default="default",
        description=(
            "Which mode to switch to after approval. ``default`` (the "
            "normal rule / ask gate) or ``accept_edits`` (auto-allow file "
            "edits, prompt for everything else)."
        ),
    )


def _preview(args: dict[str, Any]) -> str:
    plan = args.get("plan", "")
    head = plan.splitlines()[0] if plan else ""
    return f"exit plan -> {args.get('to_mode', 'default')}: {head[:40]}"


def _render_prompt(plan: str) -> str:
    """Compose the approval-prompt text. Single source of truth so tests
    and the CLI see identical framing."""
    return f"{_APPROVAL_PROMPT}\n\n{plan}"


class ExitPlanMode(BaseTool):
    """Exit plan mode — ASKS the user first. ToolError if user denies
    or if called outside plan mode."""

    # ``ModeSetter`` / ``ModeGetter`` / ``QuestionAsker`` are bare Callable
    # aliases — not pydantic models — so we need permission to store them
    # without validation. Same rationale as AskUserQuestion's ``asker`` slot.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "exit_plan_mode"
    description: str = (
        "Present the plan to the user for approval and — if approved — "
        "leave plan mode. Pass to_mode='accept_edits' to land in "
        "auto-allow-for-edits instead of the default gate. The user sees "
        "the plan verbatim and chooses Yes/No; on No the tool returns an "
        "error and mode stays 'plan' so you can revise. Only valid from "
        "plan mode — call this after enter_plan_mode."
    )
    args_schema: type[BaseModel] = ExitPlanModeParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_concurrency_safe=False,
        max_result_size_chars=8000,
        args_preview=_preview,
    )
    _set_mode: ModeSetter = PrivateAttr()
    _get_mode: ModeGetter = PrivateAttr()
    _asker: QuestionAsker = PrivateAttr()

    def __init__(
        self,
        *,
        mode_setter: ModeSetter,
        mode_getter: ModeGetter,
        asker: QuestionAsker,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._set_mode = mode_setter
        self._get_mode = mode_getter
        self._asker = asker

    def _run(
        self, plan: str, to_mode: ExitTarget = "default",
    ) -> dict[str, Any]:
        # BaseTool marks ``_run`` abstract; we cannot ask the user from a
        # sync context (the CLI asker awaits a prompt_toolkit Application).
        # Force callers through the async path — the agent loop always
        # uses ainvoke. Same pattern as AskUserQuestion.
        raise NotImplementedError("exit_plan_mode is async-only; use ainvoke")

    async def _arun(
        self, plan: str, to_mode: ExitTarget = "default",
    ) -> dict[str, Any]:
        previous = self._get_mode()
        if previous != "plan":
            raise ToolError(
                f"exit_plan_mode called from mode {previous!r}; "
                "only valid when currently in 'plan' mode"
            )
        # Ask BEFORE mutating. On denial we want the mode unchanged so
        # the model can iterate on the plan. On approval we flip, then
        # return the envelope the LLM uses to drive the next turn.
        answer = await self._asker(
            _render_prompt(plan), _APPROVAL_OPTIONS, _APPROVAL_DEFAULT,
        )
        # Normalize: any answer that isn't "Yes" (case-insensitive) is a
        # denial. Empty string (CLI cancel / Ctrl+C) is a denial too — the
        # fail-safe default, NOT an accidental approval.
        if answer.strip().lower() != "yes":
            raise ToolError(
                "user rejected the plan; staying in plan mode — "
                "revise the plan and call exit_plan_mode again"
            )
        self._set_mode(to_mode)
        return {
            "previous_mode": "plan",
            "new_mode": to_mode,
            "plan": plan,
            "approved": True,
        }
