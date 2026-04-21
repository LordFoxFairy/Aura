"""CLI implementation of the ``ask_user_question`` tool's asker callable.

Mirrors the pattern in ``aura/cli/permission.py``: wrap a prompt_toolkit
dialog so the tool stays transport-agnostic. The tool awaits whatever async
callable it's handed; this module is one concrete binding.

Cancellation (Esc in a dialog) returns an empty string so the LLM always
gets a well-typed ``{"answer": ""}`` result rather than raising. The LLM can
then decide whether to retry, pivot, or give up.
"""

from __future__ import annotations

from prompt_toolkit.shortcuts import input_dialog, radiolist_dialog

from aura.tools.ask_user import QuestionAsker


def make_cli_user_asker() -> QuestionAsker:
    """Return an async callable the ``ask_user_question`` tool can delegate to."""

    async def _ask(
        question: str, options: list[str] | None, default: str | None,
    ) -> str:
        if options:
            # Multi-choice: radiolist. Values are (value, label) tuples —
            # we use the option string as both so the returned value IS the
            # answer, no mapping step needed.
            values = [(opt, opt) for opt in options]
            # Defensive: schema already enforces default-in-options, but if
            # a caller bypasses validation, fall back to the first option
            # rather than letting radiolist crash.
            safe_default = default if default in options else options[0]
            app = radiolist_dialog(
                title="Aura asks",
                text=question,
                values=values,
                default=safe_default,
            )
            result = await app.run_async()
            # Esc / cancellation → None. Return empty string so the tool
            # result is always well-typed; the LLM sees {"answer": ""}.
            return result if isinstance(result, str) else ""

        # Free-text: single-line input dialog.
        app = input_dialog(
            title="Aura asks",
            text=question,
            default=default or "",
        )
        result = await app.run_async()
        return result if isinstance(result, str) else ""

    return _ask
