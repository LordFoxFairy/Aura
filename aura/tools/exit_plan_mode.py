"""exit_plan_mode — leave plan mode and switch to ``default`` or ``accept_edits``.

Companion to ``enter_plan_mode``. Matches claude-code's
``ExitPlanModeTool`` semantics: the model signals "planning is done;
resume executing". The tool raises ``ToolError`` if called from any mode
other than ``plan`` so the LLM doesn't use it as a general mode-switcher.

Not destructive, not read-only (meta-tool). Deliberately exempted from
plan-mode enforcement in ``aura.core.hooks.permission`` — otherwise the
model couldn't call this from plan mode.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.schemas.tool import ToolError, tool_metadata
from aura.tools.enter_plan_mode import ModeGetter, ModeSetter

# Modes the tool is allowed to transition *into*. Deliberately narrow:
# ``bypass`` is intentionally excluded — plan → bypass would be a
# security surprise, and ``bypass`` can only be entered via the CLI
# ``--bypass-permissions`` flag. ``plan`` itself is also excluded (would
# be a no-op; use ``enter_plan_mode`` instead).
ExitTarget = Literal["default", "accept_edits"]


class ExitPlanModeParams(BaseModel):
    to_mode: ExitTarget = Field(
        default="default",
        description=(
            "Which mode to switch to after exiting plan. ``default`` (the "
            "normal rule / ask gate) or ``accept_edits`` (auto-allow file "
            "edits, prompt for everything else)."
        ),
    )


def _preview(args: dict[str, Any]) -> str:
    return f"exit plan -> {args.get('to_mode', 'default')}"


class ExitPlanMode(BaseTool):
    """Exit plan mode. Raises ToolError if not currently in plan mode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "exit_plan_mode"
    description: str = (
        "Leave plan mode and resume executing. Pass to_mode='accept_edits' "
        "to switch straight into auto-allow-for-edits; otherwise returns to "
        "the standard rule/ask gate. Raises an error if not currently in "
        "plan mode — use this only after enter_plan_mode."
    )
    args_schema: type[BaseModel] = ExitPlanModeParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_concurrency_safe=False,
        max_result_size_chars=500,
        args_preview=_preview,
    )
    _set_mode: ModeSetter = PrivateAttr()
    _get_mode: ModeGetter = PrivateAttr()

    def __init__(
        self,
        *,
        mode_setter: ModeSetter,
        mode_getter: ModeGetter,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._set_mode = mode_setter
        self._get_mode = mode_getter

    def _run(self, to_mode: ExitTarget = "default") -> dict[str, Any]:
        previous = self._get_mode()
        if previous != "plan":
            raise ToolError(
                f"exit_plan_mode called from mode {previous!r}; "
                "only valid when currently in 'plan' mode"
            )
        self._set_mode(to_mode)
        return {"previous_mode": "plan", "new_mode": to_mode}
