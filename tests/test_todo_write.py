"""Tests for aura.tools.todo_write — stateful factory-bound tool."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aura.core.state import LoopState
from aura.tools.todo_write import (
    TodoItem,
    TodoWriteParams,
    make_todo_write_tool,
)


async def test_single_pending_todo_sets_state_and_returns_message() -> None:
    state = LoopState()
    tool = make_todo_write_tool(state)
    out = await tool.ainvoke(
        {"todos": [{"content": "a", "status": "pending", "activeForm": "Doing a"}]}
    )
    assert state.custom["todos"] == [
        {"content": "a", "status": "pending", "activeForm": "Doing a"}
    ]
    assert out == {"message": "Todos updated."}


async def test_empty_list_sets_empty_state() -> None:
    state = LoopState()
    tool = make_todo_write_tool(state)
    out = await tool.ainvoke({"todos": []})
    assert state.custom["todos"] == []
    assert out == {"message": "Todos updated."}


async def test_all_completed_auto_clears() -> None:
    state = LoopState()
    tool = make_todo_write_tool(state)
    await tool.ainvoke(
        {
            "todos": [
                {"content": "a", "status": "completed", "activeForm": "Doing a"},
                {"content": "b", "status": "completed", "activeForm": "Doing b"},
            ]
        }
    )
    assert state.custom["todos"] == []


def test_bad_status_enum_raises() -> None:
    with pytest.raises(ValidationError):
        TodoWriteParams.model_validate(
            {"todos": [{"content": "a", "status": "done", "activeForm": "Doing a"}]}
        )


def test_missing_activeform_raises() -> None:
    with pytest.raises(ValidationError):
        TodoWriteParams.model_validate(
            {"todos": [{"content": "a", "status": "pending"}]}
        )


def test_empty_content_raises() -> None:
    with pytest.raises(ValidationError):
        TodoItem.model_validate(
            {"content": "", "status": "pending", "activeForm": "Doing a"}
        )


def test_empty_activeform_raises() -> None:
    with pytest.raises(ValidationError):
        TodoItem.model_validate(
            {"content": "a", "status": "pending", "activeForm": ""}
        )


async def test_second_call_replaces_first() -> None:
    state = LoopState()
    tool = make_todo_write_tool(state)
    await tool.ainvoke(
        {"todos": [{"content": "a", "status": "pending", "activeForm": "Doing a"}]}
    )
    await tool.ainvoke(
        {
            "todos": [
                {"content": "b", "status": "in_progress", "activeForm": "Doing b"},
                {"content": "c", "status": "pending", "activeForm": "Doing c"},
            ]
        }
    )
    assert state.custom["todos"] == [
        {"content": "b", "status": "in_progress", "activeForm": "Doing b"},
        {"content": "c", "status": "pending", "activeForm": "Doing c"},
    ]


async def test_two_factories_are_independent() -> None:
    state1 = LoopState()
    state2 = LoopState()
    tool1 = make_todo_write_tool(state1)
    tool2 = make_todo_write_tool(state2)
    await tool1.ainvoke(
        {"todos": [{"content": "a", "status": "pending", "activeForm": "Doing a"}]}
    )
    await tool2.ainvoke(
        {"todos": [{"content": "b", "status": "pending", "activeForm": "Doing b"}]}
    )
    assert state1.custom["todos"] == [
        {"content": "a", "status": "pending", "activeForm": "Doing a"}
    ]
    assert state2.custom["todos"] == [
        {"content": "b", "status": "pending", "activeForm": "Doing b"}
    ]


def test_tool_metadata_and_name() -> None:
    state = LoopState()
    tool = make_todo_write_tool(state)
    assert tool.name == "todo_write"
    meta = tool.metadata or {}
    assert meta.get("is_read_only") is False
    assert meta.get("is_destructive") is False
    assert meta.get("is_concurrency_safe") is False
