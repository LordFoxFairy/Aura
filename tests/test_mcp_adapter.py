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
    client.get_prompt.assert_awaited_once_with("github", "summarize_pr")
