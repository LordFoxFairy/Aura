"""End-to-end checks for the ``cli.commands`` facade and default registry wiring."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from aura.application.commands.builtin import (
    CompactCommand,
    ContextCommand,
    ResumeCommand,
    format_relative_time,
    session_label,
)
from aura.application.commands.factory import build_default_registry
from aura.application.commands.factory import (
    build_default_registry as build_capability_default_registry,
)
from aura.application.commands.registry import dispatch
from aura.application.compact.result_types import CompactResult
from aura.application.session import AgentSession
from aura.config.schema import AuraConfig
from aura.infrastructure.llm import UnknownModelSpecError
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.persistence.storage_types import SessionMeta
from tests.conftest import FakeChatModel


def _agent(tmp_path: Path) -> AgentSession:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {
            "default": "openai:gpt-4o-mini",
            "opus": "openai:gpt-4o",
        },
        "tools": {"enabled": []},
    })
    return AgentSession(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "db"),
    )


def test_cli_build_default_registry_facade_points_at_factory_module() -> None:
    assert build_default_registry is build_capability_default_registry
    spec = importlib.util.find_spec("aura.application.commands.factory")
    assert spec is not None


def test_default_registry_has_builtin_set() -> None:
    # ``/team`` is gated by ``teams.enabled``; no-agent build can't read
    # config so the safe default is off — see test_teams_feature_flag.py.
    r = build_default_registry()
    names = {c.name for c in r.list()}
    assert names == {
        "/help", "/exit", "/clear", "/compact", "/context", "/model", "/export",
        "/stats",
        "/tasks", "/task-get", "/task-stop",
        "/status", "/diff", "/log", "/mcp",
        "/resume",
    }


def test_default_registry_remaining_commands_are_owned_by_capabilities() -> None:
    r = build_default_registry()
    commands = {cmd.name: cmd for cmd in r.list()}
    assert commands["/export"].__class__.__module__ == "aura.application.commands.export"
    assert commands["/mcp"].__class__.__module__ == "aura.application.commands.mcp"
    assert commands["/stats"].__class__.__module__ == "aura.application.commands.stats"


@pytest.mark.asyncio
async def test_dispatch_non_slash_not_handled(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    r = build_default_registry()
    result = await dispatch("hello there", agent, r)
    assert result.handled is False
    assert result.kind == "noop"
    assert result.text == ""


@pytest.mark.asyncio
async def test_dispatch_help_prints_command_list(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    r = build_default_registry()
    result = await dispatch("/help", agent, r)
    assert result.handled is True
    # /help renders as a modal view (kind="view") since v0.13 — REPL
    # wraps it in a framed panel and waits for Enter. Old "print" kind
    # let the help scroll away before users could read it.
    assert result.kind == "view"
    # Don't pin exact wording — registry enumerates dynamically now.
    assert "/help" in result.text
    assert "/exit" in result.text
    assert "/clear" in result.text
    assert "/model" in result.text


@pytest.mark.asyncio
async def test_dispatch_exit(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    r = build_default_registry()
    result = await dispatch("/exit", agent, r)
    assert result.handled is True
    assert result.kind == "exit"


@pytest.mark.asyncio
async def test_dispatch_clear_calls_agent_clear_session() -> None:
    mock_agent = MagicMock(spec=AgentSession)
    r = build_default_registry()
    result = await dispatch("/clear", mock_agent, r)
    assert mock_agent.clear_session.called
    assert result.handled and result.kind == "print"
    assert "cleared" in result.text


@pytest.mark.asyncio
async def test_dispatch_model_no_arg_shows_status(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    r = build_default_registry()
    result = await dispatch("/model", agent, r)
    assert result.handled is True
    assert result.kind == "print"
    assert "current:" in result.text
    assert "openai:gpt-4o-mini" in result.text
    assert "opus" in result.text


@pytest.mark.asyncio
async def test_dispatch_model_with_router_alias() -> None:
    mock_agent = MagicMock(spec=AgentSession)
    mock_agent.current_model = "openai:gpt-4o-mini"

    def _flip(spec: str) -> None:
        mock_agent.current_model = spec

    mock_agent.switch_model.side_effect = _flip
    r = build_default_registry()
    result = await dispatch("/model opus", mock_agent, r)
    mock_agent.switch_model.assert_called_once_with("opus")
    assert result.handled is True
    assert result.kind == "print"
    assert "openai:gpt-4o-mini" in result.text
    assert "opus" in result.text


@pytest.mark.asyncio
async def test_dispatch_model_unknown_returns_error_text() -> None:
    mock_agent = MagicMock(spec=AgentSession)
    mock_agent.switch_model.side_effect = UnknownModelSpecError(
        "model spec", "bogus-not-an-alias is not a router alias"
    )
    r = build_default_registry()
    result = await dispatch("/model bogus-not-an-alias", mock_agent, r)
    assert result.handled is True
    assert result.kind == "print"
    assert "error:" in result.text
    assert "bogus" in result.text


@pytest.mark.asyncio
async def test_dispatch_unknown_slash_command(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    r = build_default_registry()
    result = await dispatch("/foo", agent, r)
    assert result.handled is True
    assert result.kind == "print"
    assert "unknown command" in result.text


@pytest.mark.asyncio
async def test_dispatch_empty_line_not_handled(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    r = build_default_registry()
    result = await dispatch("", agent, r)
    assert result.handled is False


@pytest.mark.asyncio
async def test_dispatch_whitespace_line_not_handled(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    r = build_default_registry()
    result = await dispatch("   ", agent, r)
    assert result.handled is False


def test_default_registry_exposes_frontmatter_metadata() -> None:
    """Every built-in exposes the new metadata fields.

    Baseline: all built-ins default ``allowed_tools`` to the empty tuple
    (no tool-gating opinion) and carry either ``None`` or a string
    ``argument_hint``. This locks down the shape for downstream
    consumers (REPL completion, permission layer) so they can treat
    ``cmd.allowed_tools`` / ``cmd.argument_hint`` as present on every
    registered command.
    """
    r = build_default_registry()
    for cmd in r.list():
        assert cmd.allowed_tools == (), (
            f"{cmd.name} must default allowed_tools to () — got "
            f"{cmd.allowed_tools!r}"
        )
        assert cmd.argument_hint is None or isinstance(
            cmd.argument_hint, str
        )


def test_default_registry_model_command_advertises_hint() -> None:
    """Spot-check: ``/model`` carries a meaningful hint end-to-end."""
    r = build_default_registry()
    model_cmd = next(c for c in r.list() if c.name == "/model")
    assert model_cmd.argument_hint == "[spec]"


# --- /compact -------------------------------------------------------------


def _compact_agent(before: int, after: int) -> MagicMock:
    agent = MagicMock(spec=AgentSession)
    agent.compact = AsyncMock(
        return_value=CompactResult(
            before_tokens=before, after_tokens=after, source="manual"
        )
    )
    return agent


@pytest.mark.asyncio
async def test_compact_reports_token_delta_and_requests_manual_source() -> None:
    """/compact must surface the before/after token drop so users can see
    the reclaim worked, and must always run the *manual* compaction path."""
    agent = _compact_agent(before=1200, after=300)
    result = await CompactCommand().handle("", agent)
    agent.compact.assert_awaited_once_with(source="manual")
    assert result.handled and result.kind == "print"
    assert "1200" in result.text and "300" in result.text


@pytest.mark.asyncio
async def test_compact_is_idempotent_when_already_minimal() -> None:
    """A no-op compaction (before == after) must still report cleanly and
    not crash when invoked repeatedly — users mash /compact."""
    agent = _compact_agent(before=0, after=0)
    first = await CompactCommand().handle("", agent)
    second = await CompactCommand().handle("", agent)
    assert first == second
    assert "0 -> 0" in first.text
    assert agent.compact.await_count == 2


# --- /context -------------------------------------------------------------


def _context_agent(
    *,
    built: list[BaseMessage],
    history: list[BaseMessage],
    window: int,
) -> MagicMock:
    agent = MagicMock(spec=AgentSession)
    agent.context.build.return_value = built
    agent.storage.load.return_value = history
    agent.session_id = "ctx-session"
    agent.context_window = window
    # None policy => compact_summary_messages returns history as-is, keeping
    # the per-section math deterministic without a live microcompact engine.
    agent.microcompact_policy = None
    return agent


@pytest.mark.asyncio
async def test_context_buckets_each_section_by_content_tag() -> None:
    """/context must attribute tokens to the correct section so the budget
    readout is trustworthy: system vs memory vs skills vs files vs other."""
    built: list[BaseMessage] = [
        SystemMessage(content="you are a careful agent system prompt"),
        HumanMessage(content="<project-memory>remembered facts</project-memory>"),
        HumanMessage(content="<skills-available>brainstorming</skills-available>"),
        HumanMessage(content="<recent-file path='a.py'>contents</recent-file>"),
        HumanMessage(content="plain unlabelled user turn lands in other"),
    ]
    agent = _context_agent(built=built, history=[], window=100_000)
    result = await ContextCommand().handle("", agent)
    assert result.handled and result.kind == "view"
    for label in ("system", "memory", "skills", "files", "other", "total"):
        assert label in result.text
    # Each labelled bucket must be strictly positive — nothing collapsed to 0.
    for label in ("system", "memory", "skills", "files", "other"):
        line = next(ln for ln in result.text.splitlines() if ln.strip().startswith(label))
        assert int(line.rsplit(":", 1)[1].split("(")[0].strip()) > 0


@pytest.mark.asyncio
async def test_context_zero_window_does_not_divide_by_zero() -> None:
    """A misconfigured/zero context window must yield 0% rather than raise —
    the readout is diagnostic and must never crash the REPL."""
    agent = _context_agent(
        built=[SystemMessage(content="sys")], history=[], window=0
    )
    result = await ContextCommand().handle("", agent)
    assert "(0% of 0)" in result.text


@pytest.mark.asyncio
async def test_context_compact_candidates_appear_only_past_tail() -> None:
    """The 'compact' line estimates the *manual /compact summary prompt*
    over history beyond the 6-message live tail; a long history must produce
    a non-zero estimate, a short one must stay at 0."""
    long_hist: list[BaseMessage] = [
        HumanMessage(content=f"turn {i} with enough words to score tokens")
        for i in range(12)
    ]
    long_agent = _context_agent(built=[], history=long_hist, window=50_000)
    long_text = (await ContextCommand().handle("", long_agent)).text
    long_compact = next(
        ln for ln in long_text.splitlines() if ln.strip().startswith("compact")
    )
    assert int(long_compact.split(":", 1)[1].split("(")[0].strip()) > 0

    short_hist: list[BaseMessage] = [HumanMessage(content="only one turn")]
    short_agent = _context_agent(built=[], history=short_hist, window=50_000)
    short_text = (await ContextCommand().handle("", short_agent)).text
    short_compact = next(
        ln for ln in short_text.splitlines() if ln.strip().startswith("compact")
    )
    assert int(short_compact.split(":", 1)[1].split("(")[0].strip()) == 0


@pytest.mark.asyncio
async def test_context_non_string_content_is_coerced_not_dropped() -> None:
    """Multimodal/list message content must not break tag-matching — it is
    stringified, so a list-content turn still counts toward a section."""
    built: list[BaseMessage] = [
        HumanMessage(content=[{"type": "text", "text": "<recent-file>x</recent-file>"}]),
    ]
    agent = _context_agent(built=built, history=[], window=80_000)
    result = await ContextCommand().handle("", agent)
    files_line = next(
        ln for ln in result.text.splitlines() if ln.strip().startswith("files")
    )
    assert int(files_line.split(":", 1)[1].strip()) > 0


# --- format_relative_time -------------------------------------------------


@pytest.mark.parametrize(
    ("delta_seconds", "expected"),
    [
        (0, "just now"),
        (5, "just now"),
        (-30, "just now"),  # future clamp: host clock skew must not raise
        (10, "10s ago"),
        (59, "59s ago"),
        (60, "1 minute ago"),
        (120, "2 minutes ago"),
        (3600, "1 hour ago"),
        (7200, "2 hours ago"),
        (86400, "1 day ago"),
        (172800, "2 days ago"),
    ],
)
def test_format_relative_time_buckets(delta_seconds: int, expected: str) -> None:
    """Coarse 'X ago' labels must bucket + pluralize correctly across every
    boundary, and clamp future timestamps to 'just now'."""
    now = datetime(2026, 1, 1, 12, 0, 0)
    when = now - timedelta(seconds=delta_seconds)
    assert format_relative_time(when, now) == expected


def test_format_relative_time_defaults_now_to_wallclock() -> None:
    """Omitting ``now`` must fall back to the wall clock, not raise — the
    picker calls it without an explicit reference time."""
    assert format_relative_time(datetime.now()) == "just now"


# --- session_label --------------------------------------------------------


def _meta(prompt: str) -> SessionMeta:
    when = datetime(2026, 1, 1)
    return SessionMeta(
        session_id="abcdef123456",
        created_at=when,
        last_used_at=when,
        message_count=3,
        first_user_prompt=prompt,
    )


def test_session_label_truncates_id_and_shows_prompt() -> None:
    """Picker rows must show a short id plus the first prompt for recognition."""
    assert session_label(_meta("fix the parser")) == "session-abcdef12  fix the parser"


def test_session_label_falls_back_when_prompt_empty() -> None:
    """An empty first prompt must render a placeholder, never a blank row."""
    assert session_label(_meta("")) == "session-abcdef12  (no prompt)"


# --- /resume --------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_no_arg_lists_recent_sessions() -> None:
    """Bare /resume must enumerate recent sessions with relative timestamps
    and a usage hint so users can pick an id."""
    agent = MagicMock(spec=AgentSession)
    recent = datetime.now() - timedelta(minutes=5)
    agent.storage.list_sessions.return_value = [
        SessionMeta(
            session_id="sess0001xyz",
            created_at=recent,
            last_used_at=recent,
            message_count=2,
            first_user_prompt="hello",
        )
    ]
    result = await ResumeCommand().handle("", agent)
    agent.storage.list_sessions.assert_called_once_with(limit=10)
    assert result.kind == "view"
    assert "recent sessions:" in result.text
    assert "session-sess0001" in result.text
    assert "5 minutes ago" in result.text
    assert "/resume <session_id>" in result.text


@pytest.mark.asyncio
async def test_resume_no_arg_no_sessions_reports_empty() -> None:
    """With no saved sessions, bare /resume must say so rather than show an
    empty list the user can't act on."""
    agent = MagicMock(spec=AgentSession)
    agent.storage.list_sessions.return_value = []
    result = await ResumeCommand().handle("   ", agent)
    assert result.kind == "print"
    assert "no saved sessions" in result.text


@pytest.mark.asyncio
async def test_resume_with_valid_id_restores_and_reports_count() -> None:
    """A valid id must restore the session and confirm how many messages
    came back, so the user trusts the switch happened."""
    agent = MagicMock(spec=AgentSession)
    agent.resume_session.return_value = 7
    result = await ResumeCommand().handle("  target-id  ", agent)
    agent.resume_session.assert_called_once_with("target-id")
    assert result.kind == "print"
    assert "resumed session target-id (7 messages)" in result.text


@pytest.mark.asyncio
async def test_resume_unknown_id_reports_not_found() -> None:
    """An unknown id must surface a clear not-found message instead of
    leaking the KeyError from storage."""
    agent = MagicMock(spec=AgentSession)
    agent.resume_session.side_effect = KeyError("missing")
    result = await ResumeCommand().handle("ghost", agent)
    assert result.kind == "print"
    assert "session 'ghost' not found" in result.text
