"""ask_user_question tool — LLM asks the user 1..4 clarifying questions mid-turn.

Schema mirrors claude-code's ``AskUserQuestion``: a list of structured questions
with header chips, option lists (with descriptions + preview text), and an
optional free-text fallback. The injected ``UserAsker`` renders the form (via
``cli.forms.render_form`` in production) and returns ``{question_text: answer}``;
the tool flattens that into a single string the LLM consumes.

The tool is NOT ``is_concurrency_safe``: the user has exactly one attention
stream, so the loop must never batch two ``ask_user_question`` calls under
``asyncio.gather``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from aura.schemas.tool import ToolMetadata


class FormOptionDict(TypedDict, total=False):
    label: str
    description: str
    preview: str | None


class FormQuestionDict(TypedDict, total=False):
    question: str
    header: str
    multi_select: bool
    options: list[FormOptionDict] | None
    free_text_placeholder: str | None


# Async callable the tool delegates to. CLI provides a prompt_toolkit-backed
# implementation; tests / SDK callers provide their own. Returns a mapping of
# ``question_text -> answer_string`` matching the renderer's contract.
UserAsker = Callable[[list[FormQuestionDict]], Awaitable[dict[str, str]]]


class FormOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=50)
    description: str = Field(default="", max_length=200)
    preview: str | None = Field(default=None, max_length=5000)


class FormQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500)
    header: str = Field(min_length=1, max_length=12)
    multi_select: bool = False
    options: list[FormOption] | None = None
    free_text_placeholder: str | None = None


class AskUserQuestionParams(BaseModel):
    questions: list[FormQuestion] = Field(min_length=1, max_length=4)


def _format_answers(answers: dict[str, str]) -> str:
    """Render ``{question: answer}`` into the LLM-facing summary string.

    Matches claude-code's ``mapToolResultToToolResultBlockParam``: a single
    sentence the model can quote when summarizing what the user picked.
    """
    pairs = ", ".join(f'"{q}"="{a}"' for q, a in answers.items())
    return (
        f"User has answered your questions: {pairs}. "
        "You can now continue with the user's answers in mind."
    )


def _question_to_dict(q: FormQuestion) -> FormQuestionDict:
    payload: FormQuestionDict = {
        "question": q.question,
        "header": q.header,
        "multi_select": q.multi_select,
    }
    if q.options is not None:
        payload["options"] = [
            {
                "label": opt.label,
                "description": opt.description,
                "preview": opt.preview,
            }
            for opt in q.options
        ]
    if q.free_text_placeholder is not None:
        payload["free_text_placeholder"] = q.free_text_placeholder
    return payload


class AskUserQuestion(BaseTool):
    # ``UserAsker`` is a bare Callable alias — not a pydantic model — so
    # pydantic needs permission to store it on the instance without trying to
    # validate its internals. Same rationale as TodoWrite's ``LoopState`` slot.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "ask_user_question"
    description: str = (
        "Ask the user 1-4 structured clarifying questions mid-turn. USE "
        "SPARINGLY — only when the information you need is genuinely absent "
        "and cannot be inferred. Never use for confirmation of decisions you "
        "should make on your own judgment. Answers come back as a single "
        "summary string in the tool result."
    )
    args_schema: type[BaseModel] = AskUserQuestionParams
    # No rule_matcher / args_preview: this tool is auto-allowed via
    # DEFAULT_ALLOW_RULES (prompting the user before letting the LLM prompt
    # the user would be nonsense). See aura/core/permissions/defaults.py.
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=False,
        is_concurrency_safe=False,
        rule_matcher=None,
        args_preview=None,
        timeout_sec=None,
    )
    asker: UserAsker

    def _run(self, questions: list[dict[str, Any]]) -> dict[str, Any]:
        # BaseTool marks ``_run`` abstract; we cannot ask the user from a sync
        # context (the CLI asker awaits a prompt_toolkit Application). Force
        # callers through the async path — the agent loop always uses ainvoke.
        raise NotImplementedError("ask_user_question is async-only; use ainvoke")

    async def _arun(self, questions: list[dict[str, Any]]) -> dict[str, Any]:
        validated = [FormQuestion.model_validate(q) for q in questions]
        payload = [_question_to_dict(q) for q in validated]
        answers = await self.asker(payload)
        return {"text": _format_answers(answers)}
