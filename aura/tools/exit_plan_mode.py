"""exit_plan_mode — ask user, then leave plan mode on approval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.schemas.tool import ToolError, ToolMetadata
from aura.tools.ask_user import FormQuestionDict, UserAsker
from aura.tools.enter_plan_mode import ModeGetter, ModeSetter

# bypass / plan are not valid exit targets.
ExitTarget = Literal["default", "accept_edits"]

PriorModeGetter = Callable[[], str | None]

_APPROVAL_QUESTION = "Exit plan mode and accept this plan?"
_APPROVAL_HEADER = "Approve plan"


class ExitPlanModeParams(BaseModel):
    plan: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The plan to present for user approval. Markdown OK.",
    )
    to_mode: ExitTarget | None = Field(
        default=None,
        description=(
            "Mode to switch to after approval. Omit to restore the pre-plan "
            "mode; set 'default' or 'accept_edits' to override."
        ),
    )


def _preview(args: dict[str, Any]) -> str:
    plan = args.get("plan", "")
    head = plan.splitlines()[0] if plan else ""
    target = args.get("to_mode") or "default"
    return f"exit plan -> {target}: {head[:40]}"


def _build_question(plan: str) -> FormQuestionDict:
    return {
        "question": f"{_APPROVAL_QUESTION}\n\n{plan}",
        "header": _APPROVAL_HEADER,
        "multi_select": False,
        "options": [
            {"label": "Yes", "description": "Exit plan mode and proceed."},
            {"label": "No", "description": "Stay in plan mode; revise the plan."},
        ],
    }


class ExitPlanMode(BaseTool):
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
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=False,
        is_concurrency_safe=False,
        rule_matcher=None,
        args_preview=_preview,
        timeout_sec=None,
    )
    _set_mode: ModeSetter = PrivateAttr()
    _get_mode: ModeGetter = PrivateAttr()
    _asker: UserAsker = PrivateAttr()
    _get_prior_mode: PriorModeGetter | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        mode_setter: ModeSetter,
        mode_getter: ModeGetter,
        asker: UserAsker,
        get_prior_mode: PriorModeGetter | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._set_mode = mode_setter
        self._get_mode = mode_getter
        self._asker = asker
        self._get_prior_mode = get_prior_mode

    def _resolve_target(self, to_mode: ExitTarget | None) -> ExitTarget:
        # bypass/plan are never valid restore targets even if saved.
        if to_mode is not None:
            return to_mode
        if self._get_prior_mode is not None:
            prior = self._get_prior_mode()
            if prior in ("default", "accept_edits"):
                return prior  # type: ignore[return-value]
        return "default"

    def _run(
        self, plan: str, to_mode: ExitTarget | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("exit_plan_mode is async-only; use ainvoke")

    async def _arun(
        self, plan: str, to_mode: ExitTarget | None = None,
    ) -> dict[str, Any]:
        previous = self._get_mode()
        if previous != "plan":
            raise ToolError(
                f"exit_plan_mode called from mode {previous!r}; "
                "only valid when currently in 'plan' mode"
            )
        question = _build_question(plan)
        answers = await self._asker([question])
        # Empty / cancel maps to denial (fail-safe).
        answer = answers.get(question["question"], "")
        if answer.strip().lower() != "yes":
            raise ToolError(
                "user rejected the plan; staying in plan mode — "
                "revise the plan and call exit_plan_mode again"
            )
        target = self._resolve_target(to_mode)
        self._set_mode(target)
        return {
            "previous_mode": "plan",
            "new_mode": target,
            "plan": plan,
            "approved": True,
        }
