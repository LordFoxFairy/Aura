"""CLI ``UserAsker`` — thin shim over the unified form widget."""

from __future__ import annotations

from rich.console import Console

from aura.infrastructure.persistence import journal
from aura.tools.ask_user import FormQuestionDict, UserAsker
from cli.forms.widget import FormCancelled, render_form


def make_cli_user_asker(
    console: Console | None = None,  # noqa: ARG001 — kept for symmetry with permission asker
    *,
    timeout: float | None = None,  # noqa: ARG001 — render_form does not currently honor a timeout
) -> UserAsker:
    async def _ask(questions: list[FormQuestionDict]) -> dict[str, str]:
        try:
            return await render_form(list(questions))
        except FormCancelled:
            journal.write("user_question_cancelled", count=len(questions))
            return {q.get("question", ""): "" for q in questions}

    return _ask
