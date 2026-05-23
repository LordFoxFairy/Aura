"""Tests for aura.tools.ask_user — the ``ask_user_question`` stateful tool.

Schema mirrors claude-code's AskUserQuestion: 1..4 structured questions with
header chips, option lists (label + description + optional preview), or
free-text fallback. Output is a single summary string the LLM consumes.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aura.schemas.tool_meta_access import meta_dict
from aura.tools.ask_user import (
    AskUserQuestion,
    AskUserQuestionParams,
    FormOption,
    FormQuestion,
    FormQuestionDict,
    UserAsker,
)


def _stub_asker(
    answers: dict[str, str] | None = None,
) -> tuple[UserAsker, list[list[FormQuestionDict]]]:
    """Return (asker, captured) where captured records each call's questions."""
    captured: list[list[FormQuestionDict]] = []

    async def _ask(questions: list[FormQuestionDict]) -> dict[str, str]:
        captured.append(list(questions))
        if answers is not None:
            return answers
        return {q.get("question", ""): "stub" for q in questions}

    return _ask, captured


def test_instantiation_requires_asker_field() -> None:
    with pytest.raises(ValidationError):
        AskUserQuestion()  # type: ignore[call-arg]  # exercising missing/extra arg path


async def test_arun_returns_summary_text() -> None:
    asker, captured = _stub_asker({"What?": "42"})
    tool = AskUserQuestion(asker=asker)
    out = await tool.ainvoke({
        "questions": [
            {
                "question": "What?",
                "header": "Pick",
                "options": [{"label": "42", "description": "the answer"}],
            },
        ],
    })
    assert out == {"text": (
        'User has answered your questions: "What?"="42". '
        "You can now continue with the user's answers in mind."
    )}
    assert captured[0][0]["question"] == "What?"
    assert captured[0][0]["options"] == [
        {"label": "42", "description": "the answer", "preview": None},
    ]


async def test_arun_concatenates_multiple_answers() -> None:
    asker, _ = _stub_asker({"Q1": "A1", "Q2": "A2"})
    tool = AskUserQuestion(asker=asker)
    out = await tool.ainvoke({
        "questions": [
            {"question": "Q1", "header": "First"},
            {"question": "Q2", "header": "Second"},
        ],
    })
    assert '"Q1"="A1"' in out["text"]
    assert '"Q2"="A2"' in out["text"]


def test_schema_rejects_empty_question_list() -> None:
    with pytest.raises(ValidationError):
        AskUserQuestionParams.model_validate({"questions": []})


def test_schema_rejects_more_than_four_questions() -> None:
    with pytest.raises(ValidationError):
        AskUserQuestionParams.model_validate({
            "questions": [
                {"question": f"Q{i}", "header": "H"} for i in range(5)
            ],
        })


def test_schema_rejects_header_over_12_chars() -> None:
    with pytest.raises(ValidationError):
        FormQuestion.model_validate({
            "question": "q", "header": "x" * 13,
        })


def test_schema_rejects_label_over_50_chars() -> None:
    with pytest.raises(ValidationError):
        FormOption.model_validate({"label": "x" * 51, "description": ""})


def test_schema_rejects_preview_over_5000_chars() -> None:
    with pytest.raises(ValidationError):
        FormOption.model_validate({
            "label": "ok", "description": "", "preview": "x" * 5001,
        })


def test_schema_accepts_free_text_question_without_options() -> None:
    params = AskUserQuestionParams.model_validate({
        "questions": [{
            "question": "describe",
            "header": "Free",
            "free_text_placeholder": "type here",
        }],
    })
    assert params.questions[0].options is None
    assert params.questions[0].free_text_placeholder == "type here"


def test_schema_rejects_empty_question_string() -> None:
    with pytest.raises(ValidationError):
        FormQuestion.model_validate({"question": "", "header": "x"})


def test_schema_rejects_question_over_500_chars() -> None:
    with pytest.raises(ValidationError):
        FormQuestion.model_validate({"question": "x" * 501, "header": "x"})


def test_tool_metadata_not_concurrency_safe() -> None:
    asker, _ = _stub_asker()
    tool = AskUserQuestion(asker=asker)
    meta = meta_dict(tool)
    assert meta.get("is_concurrency_safe") is False


def test_tool_name() -> None:
    asker, _ = _stub_asker()
    tool = AskUserQuestion(asker=asker)
    assert tool.name == "ask_user_question"


async def test_two_instances_are_independent() -> None:
    asker1, cap1 = _stub_asker({"Q": "one"})
    asker2, cap2 = _stub_asker({"Q": "two"})
    tool1 = AskUserQuestion(asker=asker1)
    tool2 = AskUserQuestion(asker=asker2)
    out1 = await tool1.ainvoke({"questions": [{"question": "Q", "header": "H"}]})
    out2 = await tool2.ainvoke({"questions": [{"question": "Q", "header": "H"}]})
    assert '"Q"="one"' in out1["text"]
    assert '"Q"="two"' in out2["text"]
    assert cap1[0][0]["question"] == "Q"
    assert cap2[0][0]["question"] == "Q"
