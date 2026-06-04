"""CLI ``UserAsker`` — thin shim over the unified form widget."""

from __future__ import annotations

import asyncio

from aura.infrastructure.persistence import journal
from aura.tools.ask_user import FormQuestionDict, UserAsker
from cli.forms.widget import FormCancelled, render_form


def make_cli_user_asker(*, timeout: float | None = None) -> UserAsker:
    async def _ask(questions: list[FormQuestionDict]) -> dict[str, str]:
        form = render_form(list(questions))
        guarded = form if timeout is None else asyncio.wait_for(form, timeout)
        try:
            return await guarded
        except (FormCancelled, TimeoutError):
            journal.write("user_question_cancelled", count=len(questions))
            return {q.get("question", ""): "" for q in questions}

    return _ask
