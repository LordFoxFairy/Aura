"""Tests for the CommandRegistry abstraction (v0.1.1).

Covers registration mechanics, dispatch routing, and the four built-in
commands migrated from the old hardcoded if-else dispatcher.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.commands import (
    Command,
    CommandRegistry,
    CommandResult,
    CommandSource,
)
from aura.core.commands.builtin import (
    ClearCommand,
    ExitCommand,
    HelpCommand,
    ModelCommand,
)
from aura.core.llm import UnknownModelSpecError
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel

if TYPE_CHECKING:
    pass


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


class _StubCommand:
    """Minimal Command protocol impl for tests."""

    def __init__(
        self,
        name: str,
        *,
        source: CommandSource = "builtin",
        description: str = "stub",
        text: str = "ok",
    ) -> None:
        self.name = name
        self.description = description
        self.source: CommandSource = source
        self._text = text
        self.last_arg: str | None = None
        self.last_agent: Agent | None = None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        self.last_arg = arg
        self.last_agent = agent
        return CommandResult(handled=True, kind="print", text=self._text)


# ---------------------------------------------------------------------------
# Registry mechanics
# ---------------------------------------------------------------------------


def test_register_adds_command() -> None:
    r = CommandRegistry()
    cmd = _StubCommand("/foo")
    r.register(cmd)
    assert cmd in r.list()


def test_register_rejects_duplicate_name() -> None:
    r = CommandRegistry()
    r.register(_StubCommand("/foo"))
    with pytest.raises(ValueError):
        r.register(_StubCommand("/foo"))


def test_unregister_is_idempotent_on_missing_name() -> None:
    r = CommandRegistry()
    # Should not raise.
    r.unregister("/never-registered")
    r.register(_StubCommand("/foo"))
    r.unregister("/foo")
    r.unregister("/foo")
    assert r.list() == []


def test_list_returns_sorted_by_name() -> None:
    r = CommandRegistry()
    r.register(_StubCommand("/zebra"))
    r.register(_StubCommand("/apple"))
    r.register(_StubCommand("/mango"))
    names = [c.name for c in r.list()]
    assert names == ["/apple", "/mango", "/zebra"]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_passes_through_non_slash(tmp_path: Path) -> None:
    r = CommandRegistry()
    agent = _agent(tmp_path)
    result = await r.dispatch("hello there", agent)
    assert result.handled is False
    assert result.kind == "noop"
    assert result.text == ""


@pytest.mark.asyncio
async def test_dispatch_unknown_command_prints_hint(tmp_path: Path) -> None:
    r = CommandRegistry()
    agent = _agent(tmp_path)
    result = await r.dispatch("/nope", agent)
    assert result.handled is True
    assert result.kind == "print"
    assert "unknown command" in result.text
    assert "/help" in result.text


@pytest.mark.asyncio
async def test_dispatch_calls_registered_handler_with_arg(
    tmp_path: Path,
) -> None:
    r = CommandRegistry()
    cmd = _StubCommand("/greet", text="hello world")
    r.register(cmd)
    agent = _agent(tmp_path)

    result = await r.dispatch("/greet  alice bob  ", agent)

    assert cmd.last_arg == "alice bob"
    assert cmd.last_agent is agent
    assert result.handled is True
    assert result.text == "hello world"


# ---------------------------------------------------------------------------
# Built-in commands
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_help_command_enumerates_all_registered(tmp_path: Path) -> None:
    r = CommandRegistry()
    help_cmd = HelpCommand(registry=r)
    r.register(help_cmd)
    r.register(_StubCommand("/alpha", description="alpha desc"))
    r.register(_StubCommand("/beta", description="beta desc"))
    agent = _agent(tmp_path)

    result = await r.dispatch("/help", agent)

    assert result.handled is True
    assert result.kind == "print"
    assert "/help" in result.text
    assert "/alpha" in result.text
    assert "/beta" in result.text


@pytest.mark.asyncio
async def test_exit_command_returns_kind_exit(tmp_path: Path) -> None:
    r = CommandRegistry()
    r.register(ExitCommand())
    agent = _agent(tmp_path)

    result = await r.dispatch("/exit", agent)

    assert result.handled is True
    assert result.kind == "exit"


@pytest.mark.asyncio
async def test_clear_command_invokes_agent_clear_session() -> None:
    r = CommandRegistry()
    r.register(ClearCommand())
    mock_agent = MagicMock(spec=Agent)

    result = await r.dispatch("/clear", mock_agent)

    assert mock_agent.clear_session.called
    assert result.handled is True
    assert result.kind == "print"
    assert "cleared" in result.text


@pytest.mark.asyncio
async def test_model_command_delegates_to_agent_switch_model() -> None:
    r = CommandRegistry()
    r.register(ModelCommand())
    mock_agent = MagicMock(spec=Agent)

    result = await r.dispatch("/model opus", mock_agent)

    mock_agent.switch_model.assert_called_once_with("opus")
    assert result.handled is True
    assert result.kind == "print"
    assert "switched to opus" in result.text


@pytest.mark.asyncio
async def test_model_command_handles_unknown_model_spec_error() -> None:
    r = CommandRegistry()
    r.register(ModelCommand())
    mock_agent = MagicMock(spec=Agent)
    mock_agent.switch_model.side_effect = UnknownModelSpecError(
        "model spec", "bogus-not-an-alias is not a router alias"
    )

    result = await r.dispatch("/model bogus-not-an-alias", mock_agent)

    assert result.handled is True
    assert result.kind == "print"
    assert "error:" in result.text
    assert "bogus" in result.text


# ---------------------------------------------------------------------------
# Forward-facing: external source registration (Skills/MCP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_command_from_skill_source_registers_and_dispatches(
    tmp_path: Path,
) -> None:
    """Future-proof: a Skill-provided command with source='skill' should
    register and dispatch identically to a builtin."""
    r = CommandRegistry()
    skill_cmd = _StubCommand(
        "/skill-thing", source="skill", text="from the skill"
    )
    r.register(skill_cmd)
    agent = _agent(tmp_path)

    result = await r.dispatch("/skill-thing do-this", agent)

    assert skill_cmd.last_arg == "do-this"
    assert result.handled is True
    assert result.text == "from the skill"

    # And list() includes it so /help can enumerate.
    assert skill_cmd in r.list()


# ---------------------------------------------------------------------------
# Protocol structural typing
# ---------------------------------------------------------------------------


def test_stub_command_satisfies_command_protocol() -> None:
    """Anchor that duck-typed commands work as registry entries.

    Pure structural type check — no @runtime_checkable, so we assign to a
    typed name and lean on mypy (make check) for the real enforcement.
    """
    cmd: Command = _StubCommand("/foo")
    assert cmd.name == "/foo"
