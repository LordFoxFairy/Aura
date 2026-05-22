"""Bash-command safety hook — Tier A hard floor.

Inserted at ``pre_tool[0]`` so it runs before permission. Safety is
"this class of command can't be safe regardless of opinion"; rules
cannot override. ``mode=bypass`` is honored (user opt-in to "run
anything"); OS still enforces real catastrophic floors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool

from aura.application.hooks import PreToolHook
from aura.application.permission.bash_safety import check_bash_safety
from aura.application.permission.decision import Decision
from aura.application.permission.denials import PermissionDenial
from aura.domain.permission.mode import DEFAULT_MODE, Mode
from aura.schemas.permissions import Allow, Replace
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult

_BASH_TOOL_NAMES: frozenset[str] = frozenset({"bash", "bash_background"})


def make_bash_safety_hook(
    *,
    mode_provider: Callable[[], Mode] | None = None,
    tool_names: frozenset[str] = _BASH_TOOL_NAMES,
) -> PreToolHook:
    if mode_provider is None:
        def _mode_provider() -> Mode:
            return DEFAULT_MODE
    else:
        _mode_provider = mode_provider

    async def _hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        tool_call_id: str = "",
        **_: Any,
    ) -> Allow | Replace:
        if tool.name not in tool_names:
            return Allow(decision=Decision(allow=True, reason="mode_bypass"))

        if _mode_provider() == "bypass":
            return Allow(decision=Decision(allow=True, reason="mode_bypass"))

        command = args.get("command")
        if not isinstance(command, str) or not command:
            return Allow(decision=Decision(allow=True, reason="mode_bypass"))

        violation = check_bash_safety(command)
        if violation is None:
            return Allow(decision=Decision(allow=True, reason="mode_bypass"))

        from aura.infrastructure.persistence import journal

        journal.write(
            "permission_decision",
            tool=tool.name,
            reason="safety_blocked",
            rule=None,
            mode=_mode_provider(),
            target=None,
            detail=violation.detail,
        )

        state.slots.turn_denials.append(
            PermissionDenial(
                tool_name=tool.name,
                tool_use_id=tool_call_id,
                tool_input=dict(args),
                reason="safety_blocked",
                target=None,
            )
        )

        return Replace(
            result=ToolResult(
                ok=False,
                error=(
                    f"bash safety blocked: {violation.detail} "
                    f"(reason={violation.reason})"
                ),
            ),
            decision=Decision(allow=False, reason="safety_blocked"),
        )

    return _hook
