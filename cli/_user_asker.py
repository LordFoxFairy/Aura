"""CLI implementation of ``ask_user_question``'s ``UserAsker``.

Thin shim over :func:`cli.forms.render_form`: the unified widget already
renders 1..4 questions with options, descriptions, previews, and free-text
input. Cancellation (Esc / Ctrl+C) maps to empty answers so the LLM always
receives a well-typed result.
"""

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
    """Return an async ``UserAsker`` backed by the unified form widget."""

    async def _ask(questions: list[FormQuestionDict]) -> dict[str, str]:
        try:
            return await render_form(list(questions))
        except FormCancelled:
            journal.write("user_question_cancelled", count=len(questions))
            return {q.get("question", ""): "" for q in questions}

    return _ask
