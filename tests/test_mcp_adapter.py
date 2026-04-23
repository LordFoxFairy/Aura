"""Tests for aura.core.mcp.adapter — metadata + command wrappers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from aura.core.mcp.adapter import add_aura_metadata, make_mcp_command


class _Params(BaseModel):
    x: str = ""


def _mk_tool(name: str) -> StructuredTool:
    async def _coro(x: str = "") -> dict[str, Any]:
        return {}
    return StructuredTool(
        name=name,
        description="fixture",
        args_schema=_Params,
        coroutine=_coro,
    )


def test_add_aura_metadata_attaches_conservative_flags() -> None:
    tool = _mk_tool("search")
    out = add_aura_metadata(tool, server_name="github")
    md = out.metadata or {}
    # MCP servers don't self-declare these; defaults are conservative.
    assert md.get("is_destructive") is True
    assert md.get("is_concurrency_safe") is False
    assert md.get("is_read_only") is False
    assert md.get("max_result_size_chars") == 30_000
    assert md.get("rule_matcher") is None
    # args_preview is a generic callable producing a string.
    preview = md.get("args_preview")
    assert callable(preview)
    assert isinstance(preview({"q": "hi"}), str)


def test_add_aura_metadata_namespaces_tool_name() -> None:
    tool = _mk_tool("search")
    add_aura_metadata(tool, server_name="github")
    assert tool.name == "mcp__github__search"


def test_add_aura_metadata_preserves_already_namespaced_name() -> None:
    tool = _mk_tool("mcp__github__search")
    add_aura_metadata(tool, server_name="github")
    assert tool.name == "mcp__github__search"


@pytest.mark.asyncio
async def test_make_mcp_command_source_is_mcp_and_name_prefixed() -> None:
    client = MagicMock()
    client.get_prompt = AsyncMock(return_value=[HumanMessage(content="PROMPT-BODY")])
    cmd = make_mcp_command(
        server_name="github",
        prompt_name="summarize_pr",
        prompt_description="summarize a PR",
        client=client,
    )
    assert cmd.source == "mcp"
    assert cmd.name == "/github__summarize_pr"
    assert cmd.description == "summarize a PR"


def test_mcp_command_no_args_has_empty_frontmatter_defaults() -> None:
    """Argument-less MCP prompt → empty tuple + None, symmetric with skills.

    MCP prompts don't self-declare tool-gating; the Command surface
    must still carry the fields so ``/help`` can treat every command
    uniformly.
    """
    client = MagicMock()
    cmd = make_mcp_command(
        server_name="gh",
        prompt_name="list_repos",
        prompt_description="list repos",
        client=client,
    )
    assert cmd.allowed_tools == ()
    assert cmd.argument_hint is None


def test_mcp_command_argument_hint_derived_from_prompt_args() -> None:
    """Declared MCP prompt args → human-readable argument_hint.

    Required args render as ``<name>``, optional as ``[name]`` — mirrors
    the conventional "required angle / optional square" CLI idiom and
    lets the slash-command picker surface the call shape without
    re-querying the MCP server.
    """

    class _A:
        def __init__(self, name: str, *, required: bool = False) -> None:
            self.name = name
            self.required = required

    client = MagicMock()
    cmd = make_mcp_command(
        server_name="gh",
        prompt_name="summarize_pr",
        prompt_description="summarize a PR",
        client=client,
        prompt_arguments=[_A("pr_id", required=True), _A("style")],
    )
    assert cmd.allowed_tools == ()
    assert cmd.argument_hint == "<pr_id> [style]"


@pytest.mark.asyncio
async def test_make_mcp_command_handle_fetches_body_and_prints() -> None:
    client = MagicMock()
    client.get_prompt = AsyncMock(return_value=[HumanMessage(content="PROMPT-BODY")])
    cmd = make_mcp_command(
        server_name="github",
        prompt_name="summarize_pr",
        prompt_description="summarize a PR",
        client=client,
    )
    agent = MagicMock()
    result = await cmd.handle("", agent)
    assert result.handled is True
    assert result.kind == "print"
    assert "PROMPT-BODY" in result.text
    # No declared arguments and no tokens → empty-dict forward. We always
    # pass ``arguments=`` now for a single code path.
    client.get_prompt.assert_awaited_once_with(
        "github", "summarize_pr", arguments={}
    )


# ---------------------------------------------------------------------------
# Argument forwarding — parity with claude-code's ``zipObject`` behaviour in
# ``src/services/mcp/client.ts`` around lines 2055/2077.
# ---------------------------------------------------------------------------


class _PromptArg:
    """Lightweight stand-in for ``mcp.types.PromptArgument``.

    The adapter reads ``.name`` and ``.required`` via ``getattr``; using a
    plain class here avoids pulling the real pydantic model into the test
    module (it would need importing ``mcp`` just for a duck-typed shape).
    """

    def __init__(self, name: str, *, required: bool = False) -> None:
        self.name = name
        self.required = required


@pytest.mark.asyncio
async def test_prompt_command_forwards_positional_args_as_named_dict() -> None:
    client = MagicMock()
    client.get_prompt = AsyncMock(return_value=[HumanMessage(content="BODY")])
    cmd = make_mcp_command(
        server_name="github",
        prompt_name="summarize_pr",
        prompt_description="summarize a PR",
        client=client,
        prompt_arguments=[_PromptArg("foo"), _PromptArg("bar")],
    )
    agent = MagicMock()
    result = await cmd.handle("alpha beta", agent)
    assert result.handled is True
    client.get_prompt.assert_awaited_once_with(
        "github",
        "summarize_pr",
        arguments={"foo": "alpha", "bar": "beta"},
    )
    assert "BODY" in result.text


@pytest.mark.asyncio
async def test_prompt_command_with_no_args_forwards_empty_dict() -> None:
    client = MagicMock()
    client.get_prompt = AsyncMock(return_value=[HumanMessage(content="BODY")])
    cmd = make_mcp_command(
        server_name="github",
        prompt_name="list_repos",
        prompt_description="list repos",
        client=client,
        prompt_arguments=[],  # explicitly declares no args
    )
    agent = MagicMock()
    await cmd.handle("", agent)
    client.get_prompt.assert_awaited_once_with(
        "github", "list_repos", arguments={}
    )


@pytest.mark.asyncio
async def test_prompt_command_missing_required_arg_errors_without_fetch() -> None:
    client = MagicMock()
    client.get_prompt = AsyncMock(return_value=[HumanMessage(content="BODY")])
    cmd = make_mcp_command(
        server_name="github",
        prompt_name="summarize_pr",
        prompt_description="summarize a PR",
        client=client,
        prompt_arguments=[
            _PromptArg("foo", required=True),
            _PromptArg("bar", required=True),
        ],
    )
    agent = MagicMock()
    result = await cmd.handle("alpha", agent)
    # Error surfaced, server NOT called.
    assert result.handled is True
    assert result.kind == "print"
    assert "missing required argument" in result.text
    assert "bar" in result.text
    client.get_prompt.assert_not_awaited()


@pytest.mark.asyncio
async def test_prompt_command_extra_positional_args_dropped_silently() -> None:
    """Claude-code's ``zipObject`` drops extra values — we match that.

    Providing more tokens than declared ``arg_names`` must NOT error
    (it would break ergonomic cases like "/pr-summary short summary please"
    where the server only declares a single free-text arg).
    """
    client = MagicMock()
    client.get_prompt = AsyncMock(return_value=[HumanMessage(content="BODY")])
    cmd = make_mcp_command(
        server_name="github",
        prompt_name="summarize_pr",
        prompt_description="summarize a PR",
        client=client,
        prompt_arguments=[_PromptArg("foo"), _PromptArg("bar")],
    )
    agent = MagicMock()
    result = await cmd.handle("alpha beta gamma", agent)
    assert result.handled is True
    client.get_prompt.assert_awaited_once_with(
        "github",
        "summarize_pr",
        arguments={"foo": "alpha", "bar": "beta"},
    )


@pytest.mark.asyncio
async def test_prompt_command_optional_arg_unprovided_omitted_from_dict() -> None:
    """Optional args that aren't provided → absent from the forwarded dict.

    Rationale: MCP servers treat a *missing* key differently from an
    explicit ``None`` — e.g. a template with ``{{foo|default:'x'}}``
    renders the default only if the key is absent. Forwarding ``None``
    for unprovided optionals would defeat that.
    """
    client = MagicMock()
    client.get_prompt = AsyncMock(return_value=[HumanMessage(content="BODY")])
    cmd = make_mcp_command(
        server_name="gh",
        prompt_name="p",
        prompt_description="p",
        client=client,
        prompt_arguments=[
            _PromptArg("foo", required=True),
            _PromptArg("bar", required=False),
        ],
    )
    agent = MagicMock()
    await cmd.handle("only-foo", agent)
    client.get_prompt.assert_awaited_once_with(
        "gh", "p", arguments={"foo": "only-foo"}
    )


@pytest.mark.asyncio
async def test_prompt_command_fetch_failure_is_surfaced_not_raised() -> None:
    client = MagicMock()
    client.get_prompt = AsyncMock(side_effect=RuntimeError("server gone"))
    cmd = make_mcp_command(
        server_name="gh",
        prompt_name="p",
        prompt_description="p",
        client=client,
        prompt_arguments=[_PromptArg("foo")],
    )
    agent = MagicMock()
    result = await cmd.handle("alpha", agent)
    assert result.handled is True
    assert result.kind == "print"
    assert "server gone" in result.text
