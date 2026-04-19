"""Tests for aura.core.hooks.permission.make_permission_hook."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks.permission import (
    PermissionSession,
    make_permission_hook,
)
from aura.schemas.state import LoopState
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool


class _P(BaseModel):
    pass


def _noop() -> dict[str, Any]:
    return {}


def _read_only_tool() -> BaseTool:
    return build_tool(
        name="reader", description="r", args_schema=_P, func=_noop,
        is_read_only=True,
    )


def _destructive_tool(name: str = "writer") -> BaseTool:
    return build_tool(
        name=name, description="w", args_schema=_P, func=_noop,
        is_destructive=True,
    )


@dataclass
class _SpyAsker:
    answer: bool
    calls: list[str] = field(default_factory=list)

    async def __call__(self, tool: BaseTool, args: dict[str, Any]) -> bool:
        self.calls.append(tool.name)
        return self.answer


async def test_read_only_tool_allows_without_asking() -> None:
    spy = _SpyAsker(answer=False)
    session = PermissionSession()
    hook = make_permission_hook(asker=spy, session=session)
    result = await hook(tool=_read_only_tool(), args={}, state=LoopState())
    assert result is None
    assert spy.calls == []


async def test_destructive_tool_in_allowlist_allows_without_asking() -> None:
    spy = _SpyAsker(answer=False)
    session = PermissionSession(allowlist={"writer"})
    hook = make_permission_hook(asker=spy, session=session)
    result = await hook(tool=_destructive_tool(), args={}, state=LoopState())
    assert result is None
    assert spy.calls == []


async def test_destructive_tool_not_allowlisted_asks_user_and_allows() -> None:
    spy = _SpyAsker(answer=True)
    session = PermissionSession()
    hook = make_permission_hook(asker=spy, session=session)
    result = await hook(tool=_destructive_tool(), args={}, state=LoopState())
    assert result is None
    assert spy.calls == ["writer"]


async def test_destructive_tool_not_allowlisted_and_asker_denies() -> None:
    spy = _SpyAsker(answer=False)
    session = PermissionSession()
    hook = make_permission_hook(asker=spy, session=session)
    result = await hook(tool=_destructive_tool(), args={}, state=LoopState())
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error == "permission denied by user"


async def test_hook_does_not_mutate_allowlist_on_allow() -> None:
    spy = _SpyAsker(answer=True)
    session = PermissionSession()
    hook = make_permission_hook(asker=spy, session=session)
    await hook(tool=_destructive_tool(), args={}, state=LoopState())
    assert session.allowlist == set()


async def test_hook_protocol_matches_PreToolHook() -> None:
    spy = _SpyAsker(answer=True)
    session = PermissionSession()
    hook = make_permission_hook(asker=spy, session=session)
    result = await hook(tool=_destructive_tool(), args={}, state=LoopState())
    assert result is None


async def test_session_allowlist_default_empty() -> None:
    session = PermissionSession()
    assert session.allowlist == set()


async def test_session_allowlist_instances_independent() -> None:
    s1 = PermissionSession()
    s2 = PermissionSession()
    s1.allowlist.add("writer")
    assert s2.allowlist == set()
