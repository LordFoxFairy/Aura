"""Tests for aura.core.permission — resolve(), PermissionDecision, PermissionSession."""
from __future__ import annotations

import dataclasses

import pytest
from pydantic import BaseModel

from aura.core.permission import (
    PermissionDecision,
    PermissionSession,
    resolve,
)
from aura.tools.base import AuraTool, ToolResult

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _SpyAsker:
    def __init__(self, return_value: bool) -> None:
        self.return_value = return_value
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def __call__(self, tool: AuraTool, params: BaseModel) -> bool:
        self.calls.append((tool.name, params.model_dump() if hasattr(params, "model_dump") else {}))
        return self.return_value


class _EmptyParams(BaseModel):
    pass


class _FakeTool:
    def __init__(self, name: str, *, is_read_only: bool, is_destructive: bool) -> None:
        self.name = name
        self.description = "fake"
        self.input_model: type[BaseModel] = _EmptyParams
        self.is_read_only = is_read_only
        self.is_destructive = is_destructive
        self.is_concurrency_safe = True

    async def acall(self, params: BaseModel) -> ToolResult:
        return ToolResult(ok=True)


# ---------------------------------------------------------------------------
# resolve() policy tests
# ---------------------------------------------------------------------------


async def test_resolve_readonly_tool_allows_without_asking() -> None:
    tool = _FakeTool("read_file", is_read_only=True, is_destructive=False)
    asker = _SpyAsker(return_value=False)
    decision = await resolve(tool, _EmptyParams(), asker=asker, session=PermissionSession())
    assert decision.allow is True
    assert decision.reason is None
    assert len(asker.calls) == 0


async def test_resolve_allowlisted_tool_allows_without_asking() -> None:
    tool = _FakeTool("write_file", is_read_only=False, is_destructive=True)
    session = PermissionSession(allowlist={"write_file"})
    asker = _SpyAsker(return_value=False)
    decision = await resolve(tool, _EmptyParams(), asker=asker, session=session)
    assert decision.allow is True
    assert len(asker.calls) == 0


async def test_resolve_destructive_tool_asks_user_allow() -> None:
    tool = _FakeTool("write_file", is_read_only=False, is_destructive=True)
    asker = _SpyAsker(return_value=True)
    decision = await resolve(tool, _EmptyParams(), asker=asker, session=PermissionSession())
    assert decision.allow is True
    assert decision.reason is None
    assert len(asker.calls) == 1


async def test_resolve_destructive_tool_asks_user_deny() -> None:
    tool = _FakeTool("write_file", is_read_only=False, is_destructive=True)
    asker = _SpyAsker(return_value=False)
    decision = await resolve(tool, _EmptyParams(), asker=asker, session=PermissionSession())
    assert decision.allow is False
    assert decision.reason == "denied by user"
    assert len(asker.calls) == 1


async def test_resolve_does_not_mutate_allowlist_on_ask() -> None:
    tool = _FakeTool("write_file", is_read_only=False, is_destructive=True)
    session = PermissionSession()
    asker = _SpyAsker(return_value=True)
    await resolve(tool, _EmptyParams(), asker=asker, session=session)
    assert "write_file" not in session.allowlist
    assert session.allowlist == set()


# ---------------------------------------------------------------------------
# PermissionDecision dataclass tests
# ---------------------------------------------------------------------------


def test_permission_decision_is_frozen() -> None:
    decision = PermissionDecision(allow=True)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        decision.allow = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PermissionSession tests
# ---------------------------------------------------------------------------


def test_permission_session_default_allowlist_empty() -> None:
    session = PermissionSession()
    assert session.allowlist == set()


def test_permission_session_allowlist_mutates_independently() -> None:
    session_a = PermissionSession()
    session_b = PermissionSession()
    session_a.allowlist.add("tool_x")
    assert "tool_x" not in session_b.allowlist
    assert session_b.allowlist == set()
