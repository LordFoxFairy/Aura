"""Tests for aura.tools.todo_write — stateful class bound to LoopState."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aura.schemas.state import LoopState
from aura.schemas.todos import TodoItem
from aura.tools.todo_write import TodoWrite, TodoWriteParams


async def test_single_pending_todo_sets_state_and_returns_message() -> None:
    state = LoopState()
    tool = TodoWrite(state=state)
    out = await tool.ainvoke(
        {"todos": [{"content": "a", "status": "pending", "active_form": "Doing a"}]}
    )
    assert state.custom["todos"] == [
        TodoItem(content="a", status="pending", active_form="Doing a")
    ]
    assert out == {"message": "Todos updated."}


async def test_empty_list_sets_empty_state() -> None:
    state = LoopState()
    tool = TodoWrite(state=state)
    out = await tool.ainvoke({"todos": []})
    assert state.custom["todos"] == []
    assert out == {"message": "Todos updated."}


def test_bad_status_enum_raises() -> None:
    with pytest.raises(ValidationError):
        TodoWriteParams.model_validate(
            {"todos": [{"content": "a", "status": "done", "active_form": "Doing a"}]}
        )


def test_missing_activeform_raises() -> None:
    with pytest.raises(ValidationError):
        TodoWriteParams.model_validate(
            {"todos": [{"content": "a", "status": "pending"}]}
        )


def test_empty_content_raises() -> None:
    with pytest.raises(ValidationError):
        TodoItem.model_validate(
            {"content": "", "status": "pending", "active_form": "Doing a"}
        )


def test_empty_activeform_raises() -> None:
    with pytest.raises(ValidationError):
        TodoItem.model_validate(
            {"content": "a", "status": "pending", "active_form": ""}
        )


async def test_second_call_replaces_first() -> None:
    state = LoopState()
    tool = TodoWrite(state=state)
    await tool.ainvoke(
        {"todos": [{"content": "a", "status": "pending", "active_form": "Doing a"}]}
    )
    await tool.ainvoke(
        {
            "todos": [
                {"content": "b", "status": "in_progress", "active_form": "Doing b"},
                {"content": "c", "status": "pending", "active_form": "Doing c"},
            ]
        }
    )
    assert state.custom["todos"] == [
        TodoItem(content="b", status="in_progress", active_form="Doing b"),
        TodoItem(content="c", status="pending", active_form="Doing c"),
    ]


async def test_two_instances_are_independent() -> None:
    state1 = LoopState()
    state2 = LoopState()
    tool1 = TodoWrite(state=state1)
    tool2 = TodoWrite(state=state2)
    await tool1.ainvoke(
        {"todos": [{"content": "a", "status": "pending", "active_form": "Doing a"}]}
    )
    await tool2.ainvoke(
        {"todos": [{"content": "b", "status": "pending", "active_form": "Doing b"}]}
    )
    assert state1.custom["todos"] == [
        TodoItem(content="a", status="pending", active_form="Doing a")
    ]
    assert state2.custom["todos"] == [
        TodoItem(content="b", status="pending", active_form="Doing b")
    ]


async def test_empty_list_overwrites_existing() -> None:
    state = LoopState()
    tool = TodoWrite(state=state)
    await tool.ainvoke(
        {
            "todos": [
                {"content": "a", "status": "pending", "active_form": "Doing a"},
                {"content": "b", "status": "in_progress", "active_form": "Doing b"},
                {"content": "c", "status": "completed", "active_form": "Doing c"},
            ]
        }
    )
    assert len(state.custom["todos"]) == 3
    await tool.ainvoke({"todos": []})
    assert state.custom["todos"] == []


async def test_stored_items_are_pydantic_instances() -> None:
    state = LoopState()
    tool = TodoWrite(state=state)
    await tool.ainvoke(
        {"todos": [{"content": "a", "status": "pending", "active_form": "Doing a"}]}
    )
    assert isinstance(state.custom["todos"][0], TodoItem)


def test_tool_metadata_and_name() -> None:
    state = LoopState()
    tool = TodoWrite(state=state)
    assert tool.name == "todo_write"
    meta = tool.metadata or {}
    assert meta.get("is_read_only") is False
    assert meta.get("is_destructive") is False
    assert meta.get("is_concurrency_safe") is False


def test_todo_write_metadata_includes_matcher_and_preview() -> None:
    from aura.tools.todo_write import TodoWrite

    state = LoopState()
    tool = TodoWrite(state=state)
    meta = tool.metadata or {}
    assert meta.get("rule_matcher") is None
    preview = meta.get("args_preview")
    assert callable(preview)
    assert preview({"todos": []}) == "todos: 0 items"


# ---------------------------------------------------------------------------
# in_progress cardinality validation — matches claude-code TodoWrite policy
# "exactly ONE in_progress at a time" (zero is also allowed; more than one
# is the scope-creep signal we reject).
# ---------------------------------------------------------------------------


def test_multiple_in_progress_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TodoWriteParams.model_validate({
            "todos": [
                {"content": "a", "status": "in_progress", "active_form": "Doing a"},
                {"content": "b", "status": "in_progress", "active_form": "Doing b"},
            ]
        })
    msg = str(exc_info.value)
    assert "in_progress" in msg
    assert "one" in msg.lower() or "2" in msg


def test_three_in_progress_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TodoWriteParams.model_validate({
            "todos": [
                {"content": "a", "status": "in_progress", "active_form": "Doing a"},
                {"content": "b", "status": "in_progress", "active_form": "Doing b"},
                {"content": "c", "status": "in_progress", "active_form": "Doing c"},
            ]
        })
    assert "3" in str(exc_info.value)


def test_zero_in_progress_allowed() -> None:
    # All-pending list is a valid state (initial plan before any work starts).
    p = TodoWriteParams.model_validate({
        "todos": [
            {"content": "a", "status": "pending", "active_form": "Doing a"},
            {"content": "b", "status": "pending", "active_form": "Doing b"},
        ]
    })
    assert len(p.todos) == 2


def test_one_in_progress_allowed() -> None:
    # Canonical shape: exactly one active item, others pending / completed.
    p = TodoWriteParams.model_validate({
        "todos": [
            {"content": "a", "status": "completed", "active_form": "Doing a"},
            {"content": "b", "status": "in_progress", "active_form": "Doing b"},
            {"content": "c", "status": "pending", "active_form": "Doing c"},
        ]
    })
    assert sum(1 for t in p.todos if t.status == "in_progress") == 1


def test_cardinality_error_message_is_informative() -> None:
    # The LLM sees this ValidationError wrapped in the tool result — the
    # message must be specific enough that it can correct the next call.
    with pytest.raises(ValidationError) as exc_info:
        TodoWriteParams.model_validate({
            "todos": [
                {"content": "a", "status": "in_progress", "active_form": "Doing a"},
                {"content": "b", "status": "in_progress", "active_form": "Doing b"},
            ]
        })
    msg = str(exc_info.value)
    # Must name the constraint, the violation, and the specific count.
    assert "in_progress" in msg
    assert "one" in msg.lower()
    assert "2" in msg
