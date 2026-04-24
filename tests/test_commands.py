"""Tests for the aura.cli.commands façade.

Exercises the public surface (``build_default_registry`` + async
``dispatch``) end-to-end — mechanics of the underlying registry and each
built-in command are covered in ``test_command_registry.py``. These tests
are the regression guard against anyone breaking the façade import path
or its wiring of the four default commands.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aura.cli.commands import build_default_registry, dispatch
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.llm import UnknownModelSpecError
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel


def _agent(tmp_path: Path) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {
            "default": "openai:gpt-4o-mini",
            "opus": "openai:gpt-4o",
        },
        "tools": {"enabled": []},
    })
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "db"),
    )


def test_default_registry_has_builtin_set() -> None:
    r = build_default_registry()
    names = {c.name for c in r.list()}
    assert names == {
        "/help", "/exit", "/clear", "/compact", "/model", "/export",
        "/stats",
        "/tasks", "/task-get", "/task-stop",
        "/status", "/diff", "/log", "/mcp",
        "/buddy",
    }


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
    assert result.kind == "print"
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
    mock_agent = MagicMock(spec=Agent)
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
    mock_agent = MagicMock(spec=Agent)
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
    mock_agent = MagicMock(spec=Agent)
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
