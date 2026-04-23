"""enter_plan_mode — LLM-programmatically enter plan mode with a plan.

Companion to ``exit_plan_mode``. Matches claude-code's ``EnterPlanModeTool``
semantics: the model declares "I am now going to plan, not execute"; the
permission hook then blocks every write / side-effecting tool call until
the model (or the user) exits plan mode.

Not destructive; not read-only. Meta-tool: it mutates the Agent's
permission-mode state, not the filesystem. Deliberately exempted from
plan-mode enforcement in ``aura.core.hooks.permission`` so it stays
reachable once plan mode is active (re-entering is a no-op, but the
schema allowance is the cleaner contract).

State is injected at Agent construction time — see ``Agent.__init__``'s
stateful-tools wiring block. The dependency here is a ``mode_setter``
closure that proxies to :meth:`Agent.set_mode` so the tool never holds a
reference to the Agent itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.schemas.tool import tool_metadata

# The callable the tool uses to flip the Agent's mode. Kept as a bare
# ``Callable`` alias (not a Protocol) so pydantic doesn't try to validate
# its internals; mirrored on the Agent side by ``Agent.set_mode``.
ModeSetter = Callable[[str], None]
# Companion reader — the tool records the previous mode in its envelope so
# the model can see whether it was a no-op vs an actual transition.
ModeGetter = Callable[[], str]


class EnterPlanModeParams(BaseModel):
    plan: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description=(
            "The plan the model intends to propose to the user. Markdown OK. "
            "Kept on the tool result so downstream rendering can surface it."
        ),
    )


def _preview(args: dict[str, Any]) -> str:
    plan = args.get("plan", "")
    head = plan.splitlines()[0] if plan else ""
    return f"plan: {head[:60]}"


class EnterPlanMode(BaseTool):
    """Enter plan mode. Blocks writes until exit_plan_mode is called."""

    # ``ModeSetter`` / ``ModeGetter`` are bare Callable aliases — not
    # pydantic models — so we need permission to store them without
    # validation. Same rationale as AskUserQuestion's ``asker`` slot.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "enter_plan_mode"
    description: str = (
        "Enter plan mode. While plan mode is active, write / side-effecting "
        "tools (write_file, edit_file, bash, task_create, todo_write) are "
        "blocked by the permission layer; read tools stay available so you "
        "can gather context. Use this when the user asks you to plan before "
        "executing. Call exit_plan_mode when the plan is ready to execute."
    )
    args_schema: type[BaseModel] = EnterPlanModeParams
    metadata: dict[str, Any] | None = tool_metadata(
        # Not destructive (no filesystem / network side effects) and not
        # read-only (mutates Agent mode state). Concurrency-unsafe: the
        # mode flip is shared state on the Agent.
        is_concurrency_safe=False,
        max_result_size_chars=8000,
        args_preview=_preview,
    )
    # PrivateAttr: pydantic would try to coerce a bare callable as a field;
    # stash it behind the model instead and expose via __init__ kwarg.
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

    def _run(self, plan: str) -> dict[str, Any]:
        previous = self._get_mode()
        if previous == "plan":
            return {
                "previous_mode": previous,
                "new_mode": "plan",
                "plan": plan,
                "note": "already in plan mode; no-op",
            }
        self._set_mode("plan")
        return {
            "previous_mode": previous,
            "new_mode": "plan",
            "plan": plan,
        }
