"""ask_user_question — LLM asks the user 1-4 structured clarifying questions mid-turn."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from aura.domain.tool import ToolMetadata, coerce_json_collection


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

    @field_validator("options", mode="before")
    @classmethod
    def _coerce_options(cls, v: object) -> object:
        return coerce_json_collection(v)


class AskUserQuestionParams(BaseModel):
    questions: list[FormQuestion] = Field(min_length=1, max_length=4)

    @field_validator("questions", mode="before")
    @classmethod
    def _coerce_questions(cls, v: object) -> object:
        return coerce_json_collection(v)


class AskUserResult(TypedDict):
    text: str


def _format_answers(answers: dict[str, str]) -> str:
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
    # arbitrary_types_allowed: UserAsker is a bare Callable alias, not a pydantic model.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "ask_user_question"
    description: str = (
        "Ask the user 1-4 structured clarifying questions mid-turn. USE "
        "SPARINGLY — only when the information you need is genuinely absent "
        "and cannot be inferred. Never use for confirmation of decisions you "
        "should make on your own judgment. Answers come back as a single "
        "summary string in the tool result."
    )
    args_schema: type[BaseModel] = AskUserQuestionParams  # pyright: ignore[reportIncompatibleVariableOverride]  # langchain BaseTool declares args_schema as mutable ArgsSchema|None; subclass narrows widely on purpose.
    # Auto-allowed: prompting before letting the LLM prompt would be nonsense.
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=False,
        is_concurrency_safe=False,
        rule_matcher=None,
        args_preview=None,
        timeout_sec=None,
    )
    asker: UserAsker

    def _run(self, questions: list[dict[str, Any]]) -> AskUserResult:
        raise NotImplementedError("ask_user_question is async-only; use ainvoke")

    async def _arun(self, questions: list[dict[str, Any]]) -> AskUserResult:
        validated = [FormQuestion.model_validate(q) for q in questions]
        payload = [_question_to_dict(q) for q in validated]
        answers = await self.asker(payload)
        return {"text": _format_answers(answers)}
