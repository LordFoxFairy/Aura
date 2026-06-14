"""Non-interactive askers for one-shot CLI execution."""

from __future__ import annotations

from aura.application.permission.asker import AskerResponse, PermissionAsker
from aura.domain.tool import ToolError
from aura.infrastructure.persistence import journal
from aura.tools.ask_user import FormQuestionDict, UserAsker

PRINT_MODE_PERMISSION_FEEDBACK = "print_mode_permission_required"
PRINT_MODE_USER_QUESTION_ERROR = (
    "print mode cannot ask follow-up questions; "
    "rerun without -p (interactive REPL) or make the prompt self-contained"
)


def make_non_interactive_permission_asker() -> PermissionAsker:
    async def _ask(**_kwargs: object) -> AskerResponse:
        journal.write("permission_answered", tool="<non-interactive>", choice="deny")
        return AskerResponse(
            choice="deny",
            feedback=PRINT_MODE_PERMISSION_FEEDBACK,
        )

    return _ask


def make_non_interactive_user_asker() -> UserAsker:
    async def _ask(questions: list[FormQuestionDict]) -> dict[str, str]:
        journal.write("user_question_denied", count=len(questions))
        raise ToolError(PRINT_MODE_USER_QUESTION_ERROR)

    return _ask
