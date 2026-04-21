"""ask_user_question tool — LLM asks the user a clarifying question mid-turn.

Companion to ``todo_write`` — another stateful class-based tool whose injected
dependency (the ``asker`` callable here, ``LoopState`` there) is threaded in
per-Agent at ``Agent.__init__`` time. See ``aura/tools/todo_write.py`` for the
canonical template.

MVP semantics
-------------
- ``question`` is the prompt shown to the user (1..500 chars).
- ``options`` (optional, ≤6) switches the UI to a multi-choice picker.
- ``default`` (optional) pre-selects / pre-fills; when ``options`` is set
  ``default`` must be one of them (pydantic-enforced at construction).
- The result is ``{"answer": <str>}``. Cancellation at the CLI returns an
  empty string — the LLM sees that and decides how to recover.

The tool is NOT ``is_concurrency_safe``: the user has exactly one attention
stream, so the loop must never batch two ``ask_user_question`` calls under
``asyncio.gather``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aura.schemas.tool import tool_metadata

# Async callable the tool delegates to. CLI provides a prompt_toolkit-backed
# implementation; tests / SDK callers provide their own.
QuestionAsker = Callable[[str, list[str] | None, str | None], Awaitable[str]]


class AskUserQuestionParams(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=500,
        description="The clarifying question to ask the user.",
    )
    options: list[str] | None = Field(
        default=None, max_length=6,
        description="Optional multi-choice options (max 6). If given, user picks one.",
    )
    default: str | None = Field(
        default=None,
        description="Optional default; must be in ``options`` if both are set.",
    )

    @model_validator(mode="after")
    def _validate_default_in_options(self) -> AskUserQuestionParams:
        if (
            self.default is not None
            and self.options is not None
            and self.default not in self.options
        ):
            raise ValueError(
                f"default={self.default!r} not in options {self.options!r}"
            )
        return self


class AskUserQuestion(BaseTool):
    # ``QuestionAsker`` is a bare Callable alias — not a pydantic model — so
    # pydantic needs permission to store it on the instance without trying to
    # validate its internals. Same rationale as TodoWrite's ``LoopState`` slot.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "ask_user_question"
    description: str = (
        "Ask the user a clarifying question mid-turn. USE SPARINGLY — only "
        "when the information you need is genuinely absent and cannot be "
        "inferred. Never use for confirmation of decisions you should make "
        "on your own judgment. The user's answer comes back as the tool "
        "result on the next turn."
    )
    args_schema: type[BaseModel] = AskUserQuestionParams
    # No rule_matcher / args_preview: this tool is auto-allowed via
    # DEFAULT_ALLOW_RULES (prompting the user before letting the LLM prompt
    # the user would be nonsense). See aura/core/permissions/defaults.py.
    metadata: dict[str, Any] | None = tool_metadata(
        is_concurrency_safe=False,
    )
    asker: QuestionAsker

    def _run(
        self,
        question: str,
        options: list[str] | None = None,
        default: str | None = None,
    ) -> dict[str, Any]:
        # BaseTool marks ``_run`` abstract; we cannot ask the user from a sync
        # context (the CLI asker awaits a prompt_toolkit Application). Force
        # callers through the async path — the agent loop always uses ainvoke.
        raise NotImplementedError("ask_user_question is async-only; use ainvoke")

    async def _arun(
        self,
        question: str,
        options: list[str] | None = None,
        default: str | None = None,
    ) -> dict[str, Any]:
        answer = await self.asker(question, options, default)
        return {"answer": answer}
