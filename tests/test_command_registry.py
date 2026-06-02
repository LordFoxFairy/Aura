"""Tests for the CommandRegistry abstraction (v0.1.1).

Covers registration mechanics, dispatch routing, and the four built-in
commands migrated from the old hardcoded if-else dispatcher.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_type_hints
from unittest.mock import MagicMock

import pytest

from aura.application.commands import (
    Command,
    CommandRegistry,
    CommandResult,
    CommandSource,
)
from aura.application.commands.builtin import (
    ClearCommand,
    ExitCommand,
    HelpCommand,
    ModelCommand,
)
from aura.application.commands.factory import build_default_registry
from aura.application.commands.registry import CommandRegistry as CapabilityCommandRegistry
from aura.application.session import AgentSession
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.infrastructure.llm import UnknownModelSpecError
from aura.infrastructure.persistence.storage import SessionStorage
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


def test_command_protocol_references_agent_session_directly() -> None:
    hints = get_type_hints(
        Command.handle,
        globalns={"AgentT": AgentSession, "CommandResult": CommandResult},
    )
    assert hints["agent"] is AgentSession


class _StubCommand:
    """Minimal Command protocol impl for tests."""

    def __init__(
        self,
        name: str,
        *,
        source: CommandSource = "builtin",
        description: str = "stub",
        text: str = "ok",
        allowed_tools: tuple[str, ...] = (),
        argument_hint: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.source: CommandSource = source
        self.allowed_tools: tuple[str, ...] = allowed_tools
        self.argument_hint: str | None = argument_hint
        self._text = text
        self.last_arg: str | None = None
        self.last_agent: object | None = None

    async def handle(self, arg: str, agent: object) -> CommandResult:
        self.last_arg = arg
        self.last_agent = agent
        return CommandResult(handled=True, kind="print", text=self._text)


def test_core_command_registry_facade_points_at_capabilities_module() -> None:
    assert CommandRegistry is CapabilityCommandRegistry
    assert CommandRegistry.__module__ == "aura.application.commands.registry"


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


def test_default_registry_accepts_agent_mcp_commands_without_cast(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    stub = _StubCommand("/mcp-prompt", source="mcp", text="from mcp")
    agent._mcp_commands = [stub]

    registry = build_default_registry(agent)

    commands = {cmd.name: cmd for cmd in registry.list()}
    assert commands["/mcp-prompt"] is stub


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
    # /help migrated to "view" kind in v0.13 — REPL renders it as a
    # modal panel that waits for Enter to dismiss, instead of a plain
    # print that scrolls into the backlog.
    assert result.kind == "view"
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
    # Simulate the live-spec flip: current_model returns "old" before
    # switch_model completes, then "opus" after.
    mock_agent.current_model = "openai:gpt-4o-mini"

    def _flip(spec: str) -> None:
        mock_agent.current_model = spec

    mock_agent.switch_model.side_effect = _flip

    result = await r.dispatch("/model opus", mock_agent)

    mock_agent.switch_model.assert_called_once_with("opus")
    assert result.handled is True
    assert result.kind == "print"
    assert "openai:gpt-4o-mini" in result.text
    assert "opus" in result.text


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


def test_stub_command_satisfies_command_protocol() -> None:
    """Anchor that duck-typed commands work as registry entries.

    Pure structural type check — no @runtime_checkable, so we assign to a
    typed name and lean on mypy (make check) for the real enforcement.
    """
    cmd: Command = _StubCommand("/foo")
    assert cmd.name == "/foo"


def test_builtin_commands_expose_default_frontmatter_fields() -> None:
    """Built-ins all carry the new fields with sensible defaults.

    ``/help`` has no args, so ``argument_hint is None`` and
    ``allowed_tools == ()``. The DATA presence matters — enforcement is
    a later layer (permission).
    """
    help_cmd = HelpCommand(registry=CommandRegistry())
    assert help_cmd.allowed_tools == ()
    assert help_cmd.argument_hint is None

    exit_cmd = ExitCommand()
    assert exit_cmd.allowed_tools == ()
    assert exit_cmd.argument_hint is None


def test_model_command_carries_argument_hint() -> None:
    """``/model`` accepts an optional spec — the hint must advertise that."""
    cmd = ModelCommand()
    assert cmd.allowed_tools == ()
    assert cmd.argument_hint == "[spec]"


def test_registry_list_preserves_frontmatter_fields() -> None:
    """``list()`` round-trips new fields untouched — no filtering."""
    r = CommandRegistry()
    stub = _StubCommand(
        "/debug",
        description="debug a bug",
        allowed_tools=("bash", "read_file"),
        argument_hint="<bug_description>",
    )
    r.register(stub)
    (listed,) = r.list()
    assert listed.allowed_tools == ("bash", "read_file")
    assert listed.argument_hint == "<bug_description>"


@pytest.mark.asyncio
async def test_help_output_includes_argument_hint_when_present(
    tmp_path: Path,
) -> None:
    """/help must render the argument_hint inline next to the command name.

    Verifies the end-user surface claude-code's slash-command picker
    exposes — without this, users can't discover that a command takes
    arguments without inspecting source.
    """
    r = CommandRegistry()
    help_cmd = HelpCommand(registry=r)
    r.register(help_cmd)
    r.register(
        _StubCommand(
            "/debug",
            description="debug a bug",
            argument_hint="<bug_description>",
        )
    )
    agent = _agent(tmp_path)
    result = await r.dispatch("/help", agent)
    assert "/debug <bug_description>" in result.text
    assert "debug a bug" in result.text
