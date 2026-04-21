"""Tests for aura.tools.ask_user — the ``ask_user_question`` stateful tool.

Mirrors the TodoWrite pattern: class-based BaseTool, pydantic-injected
dependency (``asker`` here, ``state`` there), schema-validated args.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aura.tools.ask_user import (
    AskUserQuestion,
    AskUserQuestionParams,
    QuestionAsker,
)


def _stub_asker(return_value: str = "user answer") -> tuple[QuestionAsker, list[dict]]:
    """Return (asker, captured) where captured records each call's args."""
    captured: list[dict] = []

    async def _ask(question: str, options: list[str] | None, default: str | None) -> str:
        captured.append({"question": question, "options": options, "default": default})
        return return_value

    return _ask, captured


def test_instantiation_requires_asker_field() -> None:
    with pytest.raises(ValidationError):
        AskUserQuestion()  # type: ignore[call-arg]


async def test_arun_returns_asker_answer_in_dict() -> None:
    asker, captured = _stub_asker("42")
    tool = AskUserQuestion(asker=asker)
    out = await tool.ainvoke({"question": "What is the answer?"})
    assert out == {"answer": "42"}
    assert captured == [{"question": "What is the answer?", "options": None, "default": None}]


async def test_arun_passes_options_and_default_through() -> None:
    asker, captured = _stub_asker("b")
    tool = AskUserQuestion(asker=asker)
    out = await tool.ainvoke({
        "question": "Pick one",
        "options": ["a", "b", "c"],
        "default": "b",
    })
    assert out == {"answer": "b"}
    assert captured[0]["options"] == ["a", "b", "c"]
    assert captured[0]["default"] == "b"


def test_schema_rejects_default_not_in_options() -> None:
    with pytest.raises(ValidationError):
        AskUserQuestionParams.model_validate({
            "question": "pick",
            "options": ["a", "b"],
            "default": "c",
        })


def test_schema_accepts_default_when_options_absent() -> None:
    # default without options is allowed (pre-fill free-text input).
    params = AskUserQuestionParams.model_validate({
        "question": "free text",
        "default": "hello",
    })
    assert params.default == "hello"
    assert params.options is None


def test_schema_rejects_options_with_more_than_six_entries() -> None:
    with pytest.raises(ValidationError):
        AskUserQuestionParams.model_validate({
            "question": "too many",
            "options": ["a", "b", "c", "d", "e", "f", "g"],
        })


def test_schema_accepts_exactly_six_options() -> None:
    params = AskUserQuestionParams.model_validate({
        "question": "six",
        "options": ["a", "b", "c", "d", "e", "f"],
    })
    assert params.options is not None
    assert len(params.options) == 6


def test_schema_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        AskUserQuestionParams.model_validate({"question": ""})


def test_schema_rejects_question_over_500_chars() -> None:
    with pytest.raises(ValidationError):
        AskUserQuestionParams.model_validate({"question": "x" * 501})


def test_tool_metadata_not_concurrency_safe() -> None:
    asker, _ = _stub_asker()
    tool = AskUserQuestion(asker=asker)
    meta = tool.metadata or {}
    assert meta.get("is_concurrency_safe") is False


def test_tool_name() -> None:
    asker, _ = _stub_asker()
    tool = AskUserQuestion(asker=asker)
    assert tool.name == "ask_user_question"


async def test_two_instances_are_independent() -> None:
    asker1, cap1 = _stub_asker("one")
    asker2, cap2 = _stub_asker("two")
    tool1 = AskUserQuestion(asker=asker1)
    tool2 = AskUserQuestion(asker=asker2)
    out1 = await tool1.ainvoke({"question": "q1"})
    out2 = await tool2.ainvoke({"question": "q2"})
    assert out1 == {"answer": "one"}
    assert out2 == {"answer": "two"}
    assert cap1[0]["question"] == "q1"
    assert cap2[0]["question"] == "q2"
