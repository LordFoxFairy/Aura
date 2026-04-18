"""Tests for aura.core.permission.make_permission_hook."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from aura.core.permission import (
    PermissionSession,
    make_permission_hook,
)
from aura.core.state import LoopState
from aura.tools.base import AuraTool, ToolResult, build_tool


class _P(BaseModel):
    pass


async def _noop(params: BaseModel) -> ToolResult:
    return ToolResult(ok=True)


def _read_only_tool() -> AuraTool:
    return build_tool(
        name="reader", description="r", input_model=_P, call=_noop,
        is_read_only=True,
    )


def _destructive_tool(name: str = "writer") -> AuraTool:
    return build_tool(
        name=name, description="w", input_model=_P, call=_noop,
        is_destructive=True,
    )


@dataclass
class _SpyAsker:
    answer: bool
    calls: list[str] = field(default_factory=list)

    async def __call__(self, tool: AuraTool, params: BaseModel) -> bool:
        self.calls.append(tool.name)
        return self.answer


async def test_read_only_tool_allows_without_asking() -> None:
    spy = _SpyAsker(answer=False)
    session = PermissionSession()
    hook = make_permission_hook(asker=spy, session=session)
    result = await hook(tool=_read_only_tool(), params=_P(), state=LoopState())
    assert result is None
    assert spy.calls == []


async def test_destructive_tool_in_allowlist_allows_without_asking() -> None:
    spy = _SpyAsker(answer=False)
    session = PermissionSession(allowlist={"writer"})
    hook = make_permission_hook(asker=spy, session=session)
    result = await hook(tool=_destructive_tool(), params=_P(), state=LoopState())
    assert result is None
    assert spy.calls == []


async def test_destructive_tool_not_allowlisted_asks_user_and_allows() -> None:
    spy = _SpyAsker(answer=True)
    session = PermissionSession()
    hook = make_permission_hook(asker=spy, session=session)
    result = await hook(tool=_destructive_tool(), params=_P(), state=LoopState())
    assert result is None
    assert spy.calls == ["writer"]


async def test_destructive_tool_not_allowlisted_and_asker_denies() -> None:
    spy = _SpyAsker(answer=False)
    session = PermissionSession()
    hook = make_permission_hook(asker=spy, session=session)
    result = await hook(tool=_destructive_tool(), params=_P(), state=LoopState())
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error == "permission denied by user"


async def test_hook_does_not_mutate_allowlist_on_allow() -> None:
    spy = _SpyAsker(answer=True)
    session = PermissionSession()
    hook = make_permission_hook(asker=spy, session=session)
    await hook(tool=_destructive_tool(), params=_P(), state=LoopState())
    assert session.allowlist == set()


async def test_hook_protocol_matches_PreToolHook() -> None:
    spy = _SpyAsker(answer=True)
    session = PermissionSession()
    hook = make_permission_hook(asker=spy, session=session)
    result = await hook(tool=_destructive_tool(), params=_P(), state=LoopState())
    assert result is None


async def test_session_allowlist_default_empty() -> None:
    session = PermissionSession()
    assert session.allowlist == set()


async def test_session_allowlist_instances_independent() -> None:
    s1 = PermissionSession()
    s2 = PermissionSession()
    s1.allowlist.add("writer")
    assert s2.allowlist == set()
