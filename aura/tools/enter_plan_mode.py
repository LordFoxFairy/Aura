"""enter_plan_mode — flip the agent into plan mode."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.domain.tool import ToolMetadata

ModeSetter = Callable[[str], None]
ModeGetter = Callable[[], str]
PriorModeSaver = Callable[[str], None]


class EnterPlanModeParams(BaseModel):
    plan: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The plan to propose to the user. Markdown OK.",
    )


def _preview(args: dict[str, Any]) -> str:
    plan = args.get("plan", "")
    head = plan.splitlines()[0] if plan else ""
    return f"plan: {head[:60]}"


class EnterPlanMode(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "enter_plan_mode"
    description: str = (
        "Enter plan mode. While plan mode is active, write / side-effecting "
        "tools (write_file, edit_file, bash, task_create, todo_write) are "
        "blocked by the permission layer; read tools stay available so you "
        "can gather context. Use this when the user asks you to plan before "
        "executing. Call exit_plan_mode when the plan is ready to execute."
    )
    args_schema: type[BaseModel] = EnterPlanModeParams  # pyright: ignore[reportIncompatibleVariableOverride]  # langchain BaseTool declares args_schema as mutable ArgsSchema|None; subclass narrows widely on purpose.
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
    _save_prior_mode: PriorModeSaver | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        mode_setter: ModeSetter,
        mode_getter: ModeGetter,
        save_prior_mode: PriorModeSaver | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._set_mode = mode_setter
        self._get_mode = mode_getter
        self._save_prior_mode = save_prior_mode

    def _run(self, plan: str) -> dict[str, Any]:
        previous = self._get_mode()
        if previous == "plan":
            # Re-enter is a no-op so the saved prior mode survives for exit_plan_mode to restore.
            return {
                "previous_mode": previous,
                "new_mode": "plan",
                "plan": plan,
                "note": "already in plan mode; no-op",
            }
        if self._save_prior_mode is not None:
            self._save_prior_mode(previous)
        self._set_mode("plan")
        return {
            "previous_mode": previous,
            "new_mode": "plan",
            "plan": plan,
        }
