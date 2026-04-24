"""Tests for the ``/export`` slash command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from aura.cli.commands import build_default_registry
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.commands.export import ExportCommand
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel


def _agent(tmp_path: Path) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "db"),
    )


def _seed_simple_history(agent: Agent) -> None:
    history = [
        HumanMessage(content="hello"),
        AIMessage(content="hi there"),
        HumanMessage(content="read the readme"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "README.md"},
                    "id": "tc-1",
                },
            ],
        ),
        ToolMessage(
            content="# My project\n\nA thing.",
            tool_call_id="tc-1",
            name="read_file",
        ),
        AIMessage(content="it's a project about things."),
    ]
    agent._storage.save(agent.session_id, history)


# ---------------------------------------------------------------------------
# Registration + dispatch
# ---------------------------------------------------------------------------


def test_export_command_registered_in_default_registry() -> None:
    r = build_default_registry()
    names = {c.name for c in r.list()}
    assert "/export" in names


# ---------------------------------------------------------------------------
# Default path / format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_with_no_args_writes_default_md(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_simple_history(agent)

    fake_home = tmp_path / "home"
    with patch.object(Path, "expanduser", lambda self: (
        fake_home / str(self)[2:] if str(self).startswith("~/") else self
    )):
        result = await ExportCommand().handle("", agent)

    assert result.handled is True
    assert result.kind == "print"
    assert "exported 2 turns" in result.text

    exports_dir = fake_home / ".aura/exports"
    files = list(exports_dir.glob("aura-session-*.md"))
    assert len(files) == 1
    body = files[0].read_text(encoding="utf-8")
    assert body.startswith("# Aura session export")
    await agent.aclose()


# ---------------------------------------------------------------------------
# Explicit paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_writes_markdown_to_specified_path(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    _seed_simple_history(agent)
    out = tmp_path / "session.md"

    result = await ExportCommand().handle(str(out), agent)

    assert result.handled is True
    assert str(out) in result.text
    body = out.read_text(encoding="utf-8")
    assert "# Aura session export" in body
    assert "## Turn 1 (user)" in body
    assert "## Turn 1 (assistant)" in body
    assert "## Turn 2 (user)" in body
    await agent.aclose()


@pytest.mark.asyncio
async def test_export_writes_json_to_specified_path(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_simple_history(agent)
    out = tmp_path / "session.json"

    result = await ExportCommand().handle(str(out), agent)

    assert result.handled is True
    body = out.read_text(encoding="utf-8")
    parsed = json.loads(body)
    assert parsed["session_id"] == agent.session_id
    assert isinstance(parsed["messages"], list)
    assert len(parsed["messages"]) == 6
    assert parsed["messages"][0] == {"role": "human", "content": "hello"}
    await agent.aclose()


@pytest.mark.asyncio
async def test_export_format_json_flag_no_path(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_simple_history(agent)

    fake_home = tmp_path / "home"
    with patch.object(Path, "expanduser", lambda self: (
        fake_home / str(self)[2:] if str(self).startswith("~/") else self
    )):
        result = await ExportCommand().handle("--format json", agent)

    assert result.handled is True
    exports_dir = fake_home / ".aura/exports"
    files = list(exports_dir.glob("aura-session-*.json"))
    assert len(files) == 1
    parsed = json.loads(files[0].read_text(encoding="utf-8"))
    assert "messages" in parsed
    await agent.aclose()


# ---------------------------------------------------------------------------
# Markdown content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_markdown_includes_envelope_metadata(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_simple_history(agent)
    out = tmp_path / "s.md"

    await ExportCommand().handle(str(out), agent)

    body = out.read_text(encoding="utf-8")
    assert "session_id: default" in body
    assert "model: openai:gpt-4o-mini" in body
    assert "turns: 2" in body
    # Envelope + section separator.
    assert "\n---\n" in body
    await agent.aclose()


@pytest.mark.asyncio
async def test_markdown_includes_tool_calls_and_results(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    _seed_simple_history(agent)
    out = tmp_path / "s.md"

    await ExportCommand().handle(str(out), agent)

    body = out.read_text(encoding="utf-8")
    # Tool call line for read_file with README.md arg.
    assert "### Tool calls" in body
    assert "read_file" in body
    assert "README.md" in body
    # Tool result block appears as a "tool: read_file" section with a
    # fenced code body containing the file contents.
    assert "## Turn 2 (tool: read_file)" in body
    assert "# My project" in body
    await agent.aclose()


# ---------------------------------------------------------------------------
# JSON content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_parses_back_and_preserves_tool_calls(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    _seed_simple_history(agent)
    out = tmp_path / "s.json"

    await ExportCommand().handle(str(out), agent)

    parsed = json.loads(out.read_text(encoding="utf-8"))
    ai_with_calls = parsed["messages"][3]
    assert ai_with_calls["role"] == "ai"
    assert "tool_calls" in ai_with_calls
    assert ai_with_calls["tool_calls"][0]["name"] == "read_file"
    assert ai_with_calls["tool_calls"][0]["args"] == {"path": "README.md"}
    tool_msg = parsed["messages"][4]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "tc-1"
    await agent.aclose()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_empty_session_does_not_crash(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    out = tmp_path / "empty.md"

    result = await ExportCommand().handle(str(out), agent)

    assert result.handled is True
    assert "0 turns" in result.text
    body = out.read_text(encoding="utf-8")
    assert "# Aura session export" in body
    assert "turns: 0" in body
    await agent.aclose()


@pytest.mark.asyncio
async def test_export_bad_path_returns_error_result(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_simple_history(agent)
    # A file-as-parent path guarantees mkdir/write fails on every OS and
    # tmpfs without relying on permission-sensitive system paths.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file", encoding="utf-8")
    bad = blocker / "session.md"

    result = await ExportCommand().handle(str(bad), agent)

    assert result.handled is True
    assert result.kind == "print"
    assert result.text.startswith("error:")
    await agent.aclose()


@pytest.mark.asyncio
async def test_export_unknown_extension_falls_back_to_markdown(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    _seed_simple_history(agent)
    out = tmp_path / "session.txt"

    result = await ExportCommand().handle(str(out), agent)

    assert result.handled is True
    assert "note:" in result.text
    body = out.read_text(encoding="utf-8")
    assert body.startswith("# Aura session export")
    await agent.aclose()


@pytest.mark.asyncio
async def test_export_rejects_unknown_flag(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = await ExportCommand().handle("--bogus", agent)
    assert result.handled is True
    assert result.text.startswith("error:")
    await agent.aclose()


@pytest.mark.asyncio
async def test_export_rejects_bad_format_value(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    result = await ExportCommand().handle("--format yaml", agent)
    assert result.handled is True
    assert result.text.startswith("error:")
    assert "yaml" in result.text
    await agent.aclose()
