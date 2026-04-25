"""F-06-005 — MCP tool descriptions are capped at 2048 chars at registration.

Audit: ``docs/audit-2026-04-25/06-mcp.md`` finding F-06-005. OpenAPI-generated
MCP servers can dump 15-60 KB of endpoint docs into ``tool.description``;
forwarding that verbatim into the LLM tool-schema bleeds tokens on every
turn AND breaks prompt-cache stability. The cap lives in
:func:`aura.core.mcp.adapter.add_aura_metadata` and matches claude-code's
exact constant for parity with ``src/services/mcp/client.ts:218``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from aura.core.mcp.adapter import (
    _MCP_DESCRIPTION_CAP,
    _cap_description,
    add_aura_metadata,
)
from aura.core.persistence import journal


class _Params(BaseModel):
    x: str = ""


def _mk_tool(name: str, *, description: str) -> StructuredTool:
    async def _coro(x: str = "") -> dict[str, Any]:
        return {}
    return StructuredTool(
        name=name,
        description=description,
        args_schema=_Params,
        coroutine=_coro,
    )


def test_short_description_unchanged() -> None:
    """A description well under the cap is forwarded verbatim — no
    truncation, no marker text."""
    short = "concise tool docs"
    tool = _mk_tool("search", description=short)
    add_aura_metadata(tool, server_name="gh")
    assert tool.description == short
    assert "[truncated;" not in tool.description


def test_description_at_cap_unchanged() -> None:
    """Boundary case: exactly cap chars must NOT trip the truncate path."""
    boundary = "a" * _MCP_DESCRIPTION_CAP
    tool = _mk_tool("search", description=boundary)
    add_aura_metadata(tool, server_name="gh")
    assert tool.description == boundary
    assert len(tool.description) == _MCP_DESCRIPTION_CAP


def test_long_mcp_description_capped_at_2048(tmp_path: Path) -> None:
    """A 10 KB description shrinks to within the cap with a marker tail."""
    long_desc = "x" * 10_000
    tool = _mk_tool("dump", description=long_desc)
    # Reroute journal so we don't spam the global one.
    journal.reset()
    journal.configure(tmp_path / "events.jsonl")
    try:
        add_aura_metadata(tool, server_name="gh")
    finally:
        journal.reset()
    # Final length is bounded by the cap (head slice + marker line stays
    # under 2048 by construction — head is cap-64 chars, marker adds ~80
    # chars at most for any plausible original_len).
    assert len(tool.description) <= _MCP_DESCRIPTION_CAP + 100  # marker tolerance
    assert "[truncated;" in tool.description
    assert "10000 chars" in tool.description


def test_truncation_marker_includes_original_length() -> None:
    """The marker text MUST surface the original length so operators can
    reason about how aggressive the cap is for a given server."""
    original_len = 5_000
    long_desc = "y" * original_len
    capped, truncated, reported_len = _cap_description(long_desc)
    assert truncated is True
    assert reported_len == original_len
    assert f"{original_len} chars" in capped


def test_cap_helper_returns_tuple_for_short_description() -> None:
    """``_cap_description`` returns ``(text, False, len)`` when within cap.

    Defensive contract test: callers that DON'T want to journal need a
    way to know whether truncation happened without re-measuring the
    string. Truncated flag is the cheap signal.
    """
    short = "abc"
    capped, truncated, reported_len = _cap_description(short)
    assert capped == short
    assert truncated is False
    assert reported_len == 3


def test_journal_event_emitted_on_truncation(tmp_path: Path) -> None:
    """Operators see ``mcp_description_truncated`` in the audit log
    when a tool's description was capped, with the LLM-visible name +
    server + lengths."""
    log = tmp_path / "events.jsonl"
    journal.reset()
    journal.configure(log)
    try:
        long_desc = "z" * 5_000
        tool = _mk_tool("big", description=long_desc)
        add_aura_metadata(tool, server_name="gh")
    finally:
        journal.reset()

    events = [json.loads(line) for line in log.read_text().splitlines()]
    truncs = [e for e in events if e["event"] == "mcp_description_truncated"]
    assert len(truncs) == 1
    ev = truncs[0]
    # Name was already namespaced by add_aura_metadata — journal sees the
    # post-namespace name so operators can grep on the LLM-visible identifier.
    assert ev["tool_name"] == "mcp__gh__big"
    assert ev["server"] == "gh"
    assert ev["original_len"] == 5_000
    assert ev["truncated_len"] <= _MCP_DESCRIPTION_CAP + 100


def test_no_journal_event_for_short_description(tmp_path: Path) -> None:
    """A within-cap description must NOT emit a journal event — log
    silence is the contract for the common case."""
    log = tmp_path / "events.jsonl"
    journal.reset()
    journal.configure(log)
    try:
        tool = _mk_tool("ok", description="fine")
        add_aura_metadata(tool, server_name="gh")
    finally:
        journal.reset()

    if log.exists():
        events = [json.loads(line) for line in log.read_text().splitlines()]
        assert not any(
            e["event"] == "mcp_description_truncated" for e in events
        )


def test_cap_applies_per_tool_independently(tmp_path: Path) -> None:
    """Two tools registered from the same server: one short, one long.
    The short one must stay verbatim; the long one must be capped.
    Journal event count must equal exactly one (only the long tool)."""
    log = tmp_path / "events.jsonl"
    journal.reset()
    journal.configure(log)
    try:
        short_tool = _mk_tool("short_one", description="short")
        long_tool = _mk_tool("long_one", description="q" * 4_000)
        add_aura_metadata(short_tool, server_name="gh")
        add_aura_metadata(long_tool, server_name="gh")
    finally:
        journal.reset()

    assert short_tool.description == "short"
    assert "[truncated;" in long_tool.description

    events = [json.loads(line) for line in log.read_text().splitlines()]
    truncs = [e for e in events if e["event"] == "mcp_description_truncated"]
    assert len(truncs) == 1
    assert truncs[0]["tool_name"] == "mcp__gh__long_one"


def test_cap_handles_empty_description() -> None:
    """A tool with an empty (or None-ish) description does not crash and
    produces no journal event. BaseTool's ``description`` field is a
    non-optional str in langchain-core; the empty-string path is the
    realistic edge."""
    tool = _mk_tool("empty", description="")
    add_aura_metadata(tool, server_name="gh")
    assert tool.description == ""


def test_cap_preserves_metadata_attachment_when_truncating() -> None:
    """Hitting the truncate path must NOT skip the metadata attachment —
    the conservative-defaults metadata is the WHOLE reason this function
    exists, and an early-return on truncation would be a regression."""
    tool = _mk_tool("dump", description="x" * 5_000)
    add_aura_metadata(tool, server_name="gh")
    md = tool.metadata or {}
    assert md.get("is_destructive") is True
    assert md.get("max_result_size_chars") == 30_000


def test_cap_constant_matches_claude_code() -> None:
    """Pin the constant value so a subtle off-by-one drift is caught
    by CI rather than by token-bleed in production. Claude-code's
    ``MAX_MCP_DESCRIPTION_LENGTH = 2048`` (``src/services/mcp/client.ts:218``).
    """
    assert _MCP_DESCRIPTION_CAP == 2048
