"""Tests for the ``/export`` slash command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

from aura.application.commands.export import (
    ExportCommand,
    _content_as_str,
    _format_tool_call,
    _guess_lang,
    _parse_args,
    _resolve_target,
)
from aura.application.commands.factory import build_default_registry
from aura.application.session import AgentSession
from aura.config.schema import AuraConfig
from aura.infrastructure.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel


def _agent(tmp_path: Path) -> AgentSession:
    cfg = AuraConfig.model_validate(
        {
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }
    )
    return AgentSession(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "db"),
    )


def _seed_simple_history(agent: AgentSession) -> None:
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
    agent.storage.save(agent.session_id, history)


def test_export_command_registered_in_default_registry() -> None:
    r = build_default_registry()
    names = {c.name for c in r.list()}
    assert "/export" in names


def test_export_command_owned_by_capabilities_module() -> None:
    assert ExportCommand.__module__ == "aura.application.commands.export"


@pytest.mark.asyncio
async def test_export_with_no_args_writes_default_md(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    _seed_simple_history(agent)

    fake_home = tmp_path / "home"
    with patch.object(
        Path,
        "expanduser",
        lambda self: fake_home / str(self)[2:] if str(self).startswith("~/") else self,
    ):
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
    with patch.object(
        Path,
        "expanduser",
        lambda self: fake_home / str(self)[2:] if str(self).startswith("~/") else self,
    ):
        result = await ExportCommand().handle("--format json", agent)

    assert result.handled is True
    exports_dir = fake_home / ".aura/exports"
    files = list(exports_dir.glob("aura-session-*.json"))
    assert len(files) == 1
    parsed = json.loads(files[0].read_text(encoding="utf-8"))
    assert "messages" in parsed
    await agent.aclose()


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


# --------------------------------------------------------------------------
# _parse_args boundary matrix — a malformed command line must surface a
# user-facing error, never silently drop or mis-route the export.
# --------------------------------------------------------------------------


def test_parse_args_empty_string_yields_defaults() -> None:
    """Bare ``/export`` (no tokens) must fall through to default path/format."""
    assert _parse_args("") == (None, None)


def test_parse_args_dangling_format_flag_raises() -> None:
    """``--format`` as the final token has no value — refuse rather than write md silently."""
    with pytest.raises(ValueError, match="requires an argument"):
        _parse_args("--format")


def test_parse_args_extra_positional_raises() -> None:
    """A second bare path is ambiguous; the user must pick one target, not have us guess."""
    with pytest.raises(ValueError, match="unexpected extra argument"):
        _parse_args("a.md b.md")


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("--format md out.json", ("out.json", "md")),
        ("out.json --format json", ("out.json", "json")),
        ("--format json", (None, "json")),
    ],
)
def test_parse_args_flag_before_or_after_path(
    arg: str,
    expected: tuple[str | None, str | None],
) -> None:
    """``--format`` may precede or follow the path; both orders must parse identically."""
    assert _parse_args(arg) == expected


# --------------------------------------------------------------------------
# _resolve_target boundary matrix — directory targets, explicit-format
# override, and extension inference each pick a different write path.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("trailing", ["/", "\\"])
def test_resolve_target_trailing_slash_treated_as_directory(
    trailing: str,
    tmp_path: Path,
) -> None:
    """A path ending in a separator is a directory: we must mint a timestamped file inside it."""
    base = str(tmp_path / "out") + trailing
    path, fmt, note = _resolve_target(base, None)
    assert fmt == "md"
    assert note == ""
    assert path.name.startswith("aura-session-")
    assert path.suffix == ".md"


def test_resolve_target_existing_dir_with_json_flag(tmp_path: Path) -> None:
    """When the target is a real directory, an explicit ``--format json`` must win the extension."""
    target = tmp_path / "exports"
    target.mkdir()
    path, fmt, note = _resolve_target(str(target), "json")
    assert fmt == "json"
    assert path.parent == target
    assert path.suffix == ".json"


def test_resolve_target_explicit_format_overrides_extension(tmp_path: Path) -> None:
    """``--format json`` on a ``.md`` file must honour the flag, not the misleading suffix."""
    path, fmt, note = _resolve_target(str(tmp_path / "weird.md"), "json")
    assert fmt == "json"
    assert note == ""
    assert path.suffix == ".md"


def test_resolve_target_json_extension_infers_json(tmp_path: Path) -> None:
    """A .json suffix with no flag must infer JSON so users needn't repeat the format."""
    path, fmt, note = _resolve_target(str(tmp_path / "s.JSON"), None)
    assert fmt == "json"
    assert note == ""


def test_resolve_target_known_md_extension_no_note(tmp_path: Path) -> None:
    """A recognised markdown suffix must NOT emit a fallback note (no false warning)."""
    _path, fmt, note = _resolve_target(str(tmp_path / "s.markdown"), None)
    assert fmt == "md"
    assert note == ""


# --------------------------------------------------------------------------
# _content_as_str — LangChain content is a str | list union; every shape
# must flatten losslessly so the export never drops assistant reasoning.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", ""),
        ("plain", "plain"),
        ([], ""),
        (["a", "b"], "ab"),
        ([{"text": "hi"}], "hi"),
        ([{"type": "image", "url": "x"}], '{"type": "image", "url": "x"}'),
        ([{"text": 123}], '{"text": 123}'),
        ([42], "42"),
        (None, "None"),
    ],
)
def test_content_as_str_union_shapes(content: object, expected: str) -> None:
    """Every block kind (str / dict±text / scalar / None) must serialize, not crash."""
    assert _content_as_str(content) == expected


# --------------------------------------------------------------------------
# _format_tool_call — short args inline, long/multiline args fenced; a
# non-JSON-serializable arg must degrade to ``str()`` instead of raising.
# --------------------------------------------------------------------------


def test_format_tool_call_short_args_inline() -> None:
    """Short args render inline so the transcript stays scannable for one-glance review."""
    tc: ToolCall = {"name": "ls", "args": {"path": "."}, "id": "x", "type": "tool_call"}
    out = _format_tool_call(tc)
    assert out == '- `ls({"path": "."})`'
    assert "```" not in out


def test_format_tool_call_missing_name_falls_back_to_question_mark() -> None:
    """A toolcall with no name must still render ('?') rather than crash the whole export."""
    tc: ToolCall = {"name": "", "args": {}, "id": "x", "type": "tool_call"}
    assert _format_tool_call(tc).startswith("- `?(")


def test_format_tool_call_long_args_use_fenced_block() -> None:
    """Args over 80 chars become a fenced json block so wide payloads stay readable."""
    tc: ToolCall = {
        "name": "write_file",
        "args": {"path": "x" * 100},
        "id": "x",
        "type": "tool_call",
    }
    out = _format_tool_call(tc)
    assert "```json" in out
    assert "- `write_file`:" in out


def test_format_tool_call_unserializable_short_args_degrade_inline() -> None:
    """A non-serializable short arg ({set}) must fall back to str(), never raise mid-export."""
    tc: ToolCall = {
        "name": "f",
        "args": {"s": {1, 2, 3}},
        "id": "x",
        "type": "tool_call",
    }
    out = _format_tool_call(tc)
    assert out.startswith("- `f(")
    assert "{1, 2, 3}" in out


def test_format_tool_call_unserializable_long_args_degrade_to_block() -> None:
    """A long non-serializable arg must still fence cleanly via str() without a JSON error."""
    tc: ToolCall = {
        "name": "f",
        "args": {f"k{i}": {i} for i in range(20)},
        "id": "x",
        "type": "tool_call",
    }
    out = _format_tool_call(tc)
    assert "- `f`:" in out
    assert "```json" in out


# --------------------------------------------------------------------------
# _guess_lang — fence language hints; wrong hints harm copy-paste fidelity.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "body", "expected"),
    [
        ("bash", "ls -la", "bash"),
        ("BASH_BACKGROUND", "x", "bash"),
        ("shell", "x", "bash"),
        ("read_file", "any", ""),
        ("write_file", "any", ""),
        ("edit_file", "any", ""),
        ("grep", '{"k": 1}', "json"),
        ("grep", "[1, 2, 3]", "json"),
        ("grep", "{not json}", ""),
        ("grep", "plain text", ""),
        ("grep", "", ""),
    ],
)
def test_guess_lang_matrix(tool: str, body: str, expected: str) -> None:
    """Fence hint must match the tool family / payload so rendered blocks highlight correctly."""
    assert _guess_lang(tool, body) == expected


# --------------------------------------------------------------------------
# Markdown render — non-Human/AI/Tool messages (SystemMessage) and a tool
# message with no name must export verbatim/safely, never be silently lost.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_markdown_renders_system_message_verbatim(tmp_path: Path) -> None:
    """A SystemMessage hits the catch-all branch and must appear verbatim for a lossless export."""
    agent = _agent(tmp_path)
    history: list[BaseMessage] = [
        SystemMessage(content="you are a careful agent"),
        HumanMessage(content="hi"),
    ]
    agent.storage.save(agent.session_id, history)
    out = tmp_path / "sys.md"

    await ExportCommand().handle(str(out), agent)

    body = out.read_text(encoding="utf-8")
    assert "## Turn 0 (system)" in body
    assert "you are a careful agent" in body
    await agent.aclose()


@pytest.mark.asyncio
async def test_markdown_unnamed_tool_message_defaults_to_tool(tmp_path: Path) -> None:
    """A ToolMessage with name=None must render as 'tool', not raise on the missing label."""
    agent = _agent(tmp_path)
    history: list[BaseMessage] = [
        HumanMessage(content="go"),
        AIMessage(
            content="",
            tool_calls=[{"name": "x", "args": {}, "id": "t1"}],
        ),
        ToolMessage(content="result body", tool_call_id="t1"),
    ]
    agent.storage.save(agent.session_id, history)
    out = tmp_path / "noname.md"

    await ExportCommand().handle(str(out), agent)

    body = out.read_text(encoding="utf-8")
    assert "## Turn 1 (tool: tool)" in body
    assert "result body" in body
    await agent.aclose()


@pytest.mark.asyncio
async def test_markdown_assistant_before_first_human_groups_turn_zero(
    tmp_path: Path,
) -> None:
    """Post-compact histories can start with an AIMessage; it must group under turn 0, not crash."""
    agent = _agent(tmp_path)
    history: list[BaseMessage] = [
        AIMessage(content="resumed summary"),
        HumanMessage(content="continue"),
    ]
    agent.storage.save(agent.session_id, history)
    out = tmp_path / "precompact.md"

    await ExportCommand().handle(str(out), agent)

    body = out.read_text(encoding="utf-8")
    assert "## Turn 0 (assistant)" in body
    assert "resumed summary" in body
    assert "## Turn 1 (user)" in body
    await agent.aclose()


# --------------------------------------------------------------------------
# Idempotency — exporting the same session twice must be stable (same turn
# count, parseable output), never corrupt or partially overwrite.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_twice_is_idempotent_and_stable(tmp_path: Path) -> None:
    """Re-export to the same path fully overwrites to identical content (no append drift)."""
    agent = _agent(tmp_path)
    _seed_simple_history(agent)
    out = tmp_path / "twice.json"

    first = await ExportCommand().handle(str(out), agent)
    first_body = out.read_text(encoding="utf-8")
    second = await ExportCommand().handle(str(out), agent)
    second_body = out.read_text(encoding="utf-8")

    assert first.text == second.text == first.text
    first_parsed = json.loads(first_body)
    second_parsed = json.loads(second_body)
    # exported_at timestamps may differ; turn structure must not.
    assert first_parsed["turns"] == second_parsed["turns"] == 2
    assert first_parsed["messages"] == second_parsed["messages"]
    await agent.aclose()


@pytest.mark.asyncio
async def test_export_failure_does_not_count_turns(tmp_path: Path) -> None:
    """A write failure returns an error string, not a misleading 'exported N turns'."""
    agent = _agent(tmp_path)
    _seed_simple_history(agent)
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")
    bad = blocker / "nested.json"

    result = await ExportCommand().handle(str(bad), agent)

    assert result.text.startswith("error: could not write")
    assert "exported" not in result.text
    await agent.aclose()
