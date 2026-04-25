"""Tests for ``--resume`` / ``/resume`` and the ``list_sessions`` API."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aura.cli.commands import build_default_registry, dispatch
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionMeta, SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _agent(
    tmp_path: Path,
    turns: list[FakeTurn] | None = None,
    *,
    storage: SessionStorage | None = None,
) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=turns or []),
        storage=storage or SessionStorage(tmp_path / "db"),
    )


# ---- list_sessions / SessionMeta -----------------------------------------

def test_list_sessions_returns_recent_first(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "db")
    # Save three distinct sessions, with explicit small sleeps so the
    # SQLite ``datetime('now')`` column captures distinct timestamps
    # for each insert (1-second granularity).
    storage.save("alpha", [HumanMessage(content="first prompt of alpha")])
    time.sleep(1.1)
    storage.save("beta", [HumanMessage(content="first prompt of beta")])
    time.sleep(1.1)
    storage.save("gamma", [HumanMessage(content="first prompt of gamma")])

    sessions = storage.list_sessions(limit=10)
    ids = [s.session_id for s in sessions]
    assert ids == ["gamma", "beta", "alpha"]


def test_list_sessions_respects_limit(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "db")
    for i in range(5):
        storage.save(f"s{i}", [HumanMessage(content=f"p{i}")])
        time.sleep(0.01)

    sessions = storage.list_sessions(limit=3)
    assert len(sessions) == 3


def test_list_sessions_extracts_first_user_prompt(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "db")
    storage.save(
        "abc",
        [
            HumanMessage(content="hello world"),
            AIMessage(content="hi back"),
            HumanMessage(content="follow-up — should NOT be the preview"),
        ],
    )
    sessions = storage.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].first_user_prompt == "hello world"


def test_list_sessions_truncates_long_prompt(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "db")
    long_prompt = "x" * 300
    storage.save("abc", [HumanMessage(content=long_prompt)])
    sessions = storage.list_sessions()
    assert sessions[0].first_user_prompt.endswith("…")
    # 79 chars + 1 ellipsis col = 80.
    assert len(sessions[0].first_user_prompt) == 80


def test_list_sessions_handles_empty_storage(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "db")
    assert storage.list_sessions() == []
    assert storage.session_count() == 0


def test_session_count_returns_distinct_count(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "db")
    for i in range(3):
        storage.save(f"s{i}", [HumanMessage(content="p")])
    assert storage.session_count() == 3


def test_session_meta_carries_message_count(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "db")
    storage.save(
        "abc",
        [
            HumanMessage(content="a"),
            AIMessage(content="b"),
            HumanMessage(content="c"),
        ],
    )
    sessions = storage.list_sessions()
    assert sessions[0].message_count == 3


def test_session_meta_timestamps_are_naive_datetimes(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "db")
    storage.save("abc", [HumanMessage(content="hi")])
    sessions = storage.list_sessions()
    assert isinstance(sessions[0].created_at, datetime)
    assert sessions[0].created_at.tzinfo is None


# ---- restore_session_into ------------------------------------------------

def test_resume_restores_full_history(tmp_path: Path) -> None:
    """Switching session_id makes the agent see the previous session's history.

    Setup: an Agent persists session "old" with 3 messages, swaps to
    session "old" via :meth:`Agent.resume_session`, then reads back —
    must see the full 3-message history.
    """
    storage = SessionStorage(tmp_path / "db")
    # Pre-populate "old" via the storage directly (simulates a prior run).
    storage.save(
        "old",
        [
            HumanMessage(content="first turn"),
            AIMessage(content="reply"),
            HumanMessage(content="second turn"),
        ],
    )
    # Build a fresh agent on the SAME storage so the swap can find "old".
    agent = _agent(tmp_path, storage=storage)
    assert agent.session_id == "default"

    count = agent.resume_session("old")
    assert count == 3
    assert agent.session_id == "old"
    # State counters reset on resume — fresh-session feel.
    assert agent.state.turn_count == 0
    assert agent.state.total_tokens_used == 0


async def test_resume_via_explicit_id_skips_picker(tmp_path: Path) -> None:
    """``/resume <id>`` restores directly without invoking the picker."""
    storage = SessionStorage(tmp_path / "db")
    storage.save("target", [HumanMessage(content="hello target")])

    agent = _agent(tmp_path, storage=storage)
    registry = build_default_registry(agent=agent)

    out = await dispatch("/resume target", agent, registry)
    assert out.handled
    assert out.kind == "print"
    assert "resumed session target" in out.text
    assert agent.session_id == "target"


async def test_resume_with_invalid_id_errors_clearly(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "db")
    storage.save("real", [HumanMessage(content="hi")])
    agent = _agent(tmp_path, storage=storage)
    registry = build_default_registry(agent=agent)

    out = await dispatch("/resume bogus-id", agent, registry)
    assert out.handled
    assert out.kind == "print"
    assert "not found" in out.text
    # Agent must NOT have been swapped to a bogus id.
    assert agent.session_id == "default"


async def test_resume_picker_shows_first_prompt_preview(
    tmp_path: Path,
) -> None:
    """The picker's row label embeds the session's first user prompt."""
    from aura.core.commands.builtin import session_label

    storage = SessionStorage(tmp_path / "db")
    storage.save(
        "session-id-here",
        [HumanMessage(content="fix the SSRF bug in web_fetch")],
    )
    sessions = storage.list_sessions()
    assert len(sessions) == 1
    label = session_label(sessions[0])
    assert "fix the SSRF bug" in label
    # Short id (8 chars) is at the start of the label.
    assert label.startswith("session-")


# ---- format_relative_time --------------------------------------------------

def test_format_relative_time_recent() -> None:
    from aura.core.commands.builtin import format_relative_time

    now = datetime(2026, 4, 25, 12, 0, 0)
    assert format_relative_time(datetime(2026, 4, 25, 11, 59, 58), now) == (
        "just now"
    )
    assert format_relative_time(datetime(2026, 4, 25, 11, 59, 50), now) == (
        "10s ago"
    )


def test_format_relative_time_minutes_hours_days() -> None:
    from aura.core.commands.builtin import format_relative_time

    now = datetime(2026, 4, 25, 12, 0, 0)
    assert format_relative_time(datetime(2026, 4, 25, 11, 0, 0), now) == (
        "1 hour ago"
    )
    assert format_relative_time(datetime(2026, 4, 25, 9, 0, 0), now) == (
        "3 hours ago"
    )
    assert format_relative_time(datetime(2026, 4, 23, 12, 0, 0), now) == (
        "2 days ago"
    )
    assert format_relative_time(datetime(2026, 4, 25, 11, 50, 0), now) == (
        "10 minutes ago"
    )


def test_format_relative_time_handles_clock_skew() -> None:
    from aura.core.commands.builtin import format_relative_time

    now = datetime(2026, 4, 25, 12, 0, 0)
    # ``when`` after ``now`` (clock skew between SQLite host and
    # caller); must not throw, must produce something sensible.
    assert format_relative_time(datetime(2026, 4, 25, 12, 0, 5), now) == (
        "just now"
    )


def test_session_meta_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    meta = SessionMeta(
        session_id="abc",
        created_at=datetime(2026, 4, 25),
        last_used_at=datetime(2026, 4, 25),
        message_count=3,
        first_user_prompt="hi",
    )
    with pytest.raises(FrozenInstanceError):
        meta.session_id = "xyz"  # type: ignore[misc]
