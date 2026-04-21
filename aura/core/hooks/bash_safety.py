"""Bash-command safety hook — Tier A hard floor.

Factory returns a ``PreToolHook`` that short-circuits bash commands
flagged by ``aura.core.permissions.bash_safety.check_bash_safety``.
Stateless: the hook itself closes over nothing; the policy lives in a
separate module.

Wiring: inserted at ``pre_tool[0]`` in ``Agent.__init__`` so it runs
BEFORE any caller-supplied permission hook. Permission is "is the user
OK with this?"; safety is "this class of command can't be safe
regardless of opinion". Rules cannot override — that's the whole point.

Journal event ``bash_safety_blocked`` carries the full command (local
audit trail), the reason code, and a one-line detail. The detail string
is also surfaced verbatim in the ToolResult.error so the model can
course-correct.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from aura.core.hooks import PreToolHook
from aura.core.permissions.bash_safety import check_bash_safety
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult


def make_bash_safety_hook() -> PreToolHook:
    async def _hook(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        state: LoopState,
        **_: Any,
    ) -> ToolResult | None:
        if tool.name != "bash":
            return None

        command = args.get("command")
        if not isinstance(command, str) or not command:
            # Tool's own arg-validation rejects; don't pre-empt.
            return None

        violation = check_bash_safety(command)
        if violation is None:
            return None

        # Lazy import — mirrors must_read_first.py — keeps the journal
        # dependency out of the module-load path.
        from aura.core.persistence import journal

        journal.write(
            "bash_safety_blocked",
            reason=violation.reason,
            detail=violation.detail,
            command=command,
        )
        return ToolResult(
            ok=False,
            error=(
                f"bash safety blocked: {violation.detail} "
                f"(reason={violation.reason})"
            ),
        )

    return _hook
