"""Weak models stringify list/dict tool args; the boundary must decode them.

Guards the exact failure seen live: glm-4.6 called ``skill`` with
``arguments='["..."]'`` (a JSON-string, not a list) and Pydantic's lax mode
rejected it with ``list_type`` — silently breaking every skill/todo/ask call
from any model that double-encodes collection args.
"""

from __future__ import annotations

import pytest

from aura.domain.todos import TodoItem
from aura.domain.tool import coerce_json_collection
from aura.tools.ask_user import AskUserQuestionParams, FormQuestion
from aura.tools.skill import SkillParams
from aura.tools.todo_write import TodoWriteParams


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('["a","b"]', ["a", "b"]),
        ("[]", []),
        ('{"k":1}', {"k": 1}),
        ("  [1,2]  ", [1, 2]),
        (["a"], ["a"]),
        (None, None),
        ("plain", "plain"),
        ("[bad json", "[bad json"),
        ("", ""),
        (5, 5),
        (True, True),
    ],
)
def test_coerce_json_collection(raw: object, expected: object) -> None:
    assert coerce_json_collection(raw) == expected


def test_skill_params_coerces_json_string_arguments() -> None:
    params = SkillParams.model_validate({"name": "x", "arguments": '["a","b"]'})
    assert params.arguments == ["a", "b"]


def test_skill_params_coerces_empty_json_array_string() -> None:
    assert SkillParams.model_validate({"name": "x", "arguments": "[]"}).arguments == []


def test_skill_params_real_list_unchanged() -> None:
    assert SkillParams(name="x", arguments=["a"]).arguments == ["a"]


def test_skill_params_omitted_arguments_is_none() -> None:
    assert SkillParams(name="x").arguments is None


def test_todo_write_params_coerces_json_string_todos() -> None:
    raw = '[{"content":"c","status":"pending","active_form":"doing c"}]'
    params = TodoWriteParams.model_validate({"todos": raw})
    assert [t.content for t in params.todos] == ["c"]
    assert params.todos[0].status == "pending"


def test_todo_write_params_real_list_unchanged() -> None:
    item = TodoItem(content="c", status="in_progress", active_form="doing c")
    assert TodoWriteParams(todos=[item]).todos[0].status == "in_progress"


def test_ask_user_params_coerces_json_string_questions() -> None:
    raw = '[{"question":"Which?","header":"Pick"}]'
    params = AskUserQuestionParams.model_validate({"questions": raw})
    assert params.questions[0].question == "Which?"


def test_form_question_coerces_json_string_options() -> None:
    raw = '[{"label":"A","description":"opt a"}]'
    q = FormQuestion.model_validate({"question": "Which?", "header": "Pick", "options": raw})
    assert q.options is not None
    assert q.options[0].label == "A"


def test_ask_user_params_malformed_json_still_rejected() -> None:
    # A non-decodable string is left untouched → normal list_type rejection,
    # never a silent pass that would crash the form renderer downstream.
    with pytest.raises(ValueError, match="list"):
        AskUserQuestionParams.model_validate({"questions": "not json"})
