"""Tests for aura.cli.commands.dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from aura.cli.commands import dispatch
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


def test_dispatch_non_slash_not_handled(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = dispatch("hello there", agent)
    assert result.handled is False
    assert result.kind == "noop"
    assert result.text == ""


def test_dispatch_help_prints_command_list(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = dispatch("/help", agent)
    assert result.handled is True
    assert result.kind == "print"
    assert "/exit" in result.text
    assert "/model" in result.text


def test_dispatch_exit(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = dispatch("/exit", agent)
    assert result.handled is True
    assert result.kind == "exit"


def test_dispatch_clear_calls_agent_clear_session() -> None:
    mock_agent = MagicMock(spec=Agent)
    result = dispatch("/clear", mock_agent)
    assert mock_agent.clear_session.called
    assert result.handled and result.kind == "print"
    assert "cleared" in result.text


def test_dispatch_model_no_arg_shows_status(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = dispatch("/model", agent)
    assert result.handled is True
    assert result.kind == "print"
    assert "default" in result.text
    assert "opus" in result.text


def test_dispatch_model_with_router_alias(tmp_path: Path) -> None:
    mock_agent = MagicMock(spec=Agent)
    result = dispatch("/model opus", mock_agent)
    mock_agent.switch_model.assert_called_once_with("opus")
    assert result.handled is True
    assert result.kind == "print"
    assert "switched to opus" in result.text


def test_dispatch_model_unknown_returns_error_text(tmp_path: Path) -> None:
    mock_agent = MagicMock(spec=Agent)
    mock_agent.switch_model.side_effect = UnknownModelSpecError(
        "model spec", "bogus-not-an-alias is not a router alias"
    )
    result = dispatch("/model bogus-not-an-alias", mock_agent)
    assert result.handled is True
    assert result.kind == "print"
    assert "error:" in result.text
    assert "bogus" in result.text


def test_dispatch_unknown_slash_command(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = dispatch("/foo", agent)
    assert result.handled is True
    assert result.kind == "print"
    assert "unknown command" in result.text


def test_dispatch_empty_line_not_handled(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = dispatch("", agent)
    assert result.handled is False


def test_dispatch_whitespace_line_not_handled(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = dispatch("   ", agent)
    assert result.handled is False
