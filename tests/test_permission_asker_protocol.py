"""Both transports satisfy ``PermissionAsker`` and IPC round-trips ``scope``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from aura.application.permission.asker import PermissionAsker
from aura.domain.permission.rule import Rule
from aura.tools.base import build_tool
from cli._permission_asker import make_cli_asker
from desktop.host.session_service import IpcAsker


class _Args(BaseModel):
    value: str = ""


def _tool() -> Any:
    return build_tool(
        name="demo",
        description="demo",
        args_schema=_Args,
        func=lambda value="": {"value": value},
        is_destructive=False,
    )


def test_make_cli_asker_satisfies_protocol() -> None:
    """The CLI closure factory returns a value matching ``PermissionAsker``."""
    asker = make_cli_asker()
    assert isinstance(asker, PermissionAsker)


def test_ipc_asker_satisfies_protocol() -> None:
    """The desktop IPC class structurally satisfies ``PermissionAsker``."""
    def _emit(payload: dict[str, Any]) -> None:
        return None

    asker = IpcAsker(emit=_emit)
    assert isinstance(asker, PermissionAsker)


@pytest.mark.asyncio
async def test_ipc_asker_propagates_project_scope_for_always() -> None:
    """Frontend says ``scope="project"`` → installed at project scope."""
    emitted: list[dict[str, Any]] = []

    def _emit(payload: dict[str, Any]) -> None:
        emitted.append(payload)

    asker = IpcAsker(emit=_emit)
    rule = Rule("demo", None)

    task = asyncio.create_task(
        asker(tool=_tool(), args={"value": "x"}, rule_hint=rule),
    )
    # Yield until the request lands in _pending.
    for _ in range(50):
        if emitted:
            break
        await asyncio.sleep(0.01)
    assert emitted, "permission_request was never emitted"

    assert asker.feed_response({
        "id": emitted[0]["id"],
        "choice": "always",
        "scope": "project",
    })
    response = await asyncio.wait_for(task, timeout=1)
    assert response.choice == "always"
    assert response.scope == "project"
    assert response.rule == rule


@pytest.mark.asyncio
async def test_ipc_asker_defaults_scope_to_session_when_missing() -> None:
    """Missing or unknown ``scope`` → conservative default of session."""
    emitted: list[dict[str, Any]] = []

    def _emit(payload: dict[str, Any]) -> None:
        emitted.append(payload)

    asker = IpcAsker(emit=_emit)
    rule = Rule("demo", None)

    task = asyncio.create_task(
        asker(tool=_tool(), args={"value": "x"}, rule_hint=rule),
    )
    for _ in range(50):
        if emitted:
            break
        await asyncio.sleep(0.01)
    assert emitted

    # No "scope" key — legacy frontend behavior.
    assert asker.feed_response({
        "id": emitted[0]["id"],
        "choice": "always",
    })
    response = await asyncio.wait_for(task, timeout=1)
    assert response.scope == "session"


@pytest.mark.asyncio
async def test_ipc_asker_rejects_bogus_scope_with_session_fallback() -> None:
    """Unrecognised scope falls back to ``"session"`` — never trust the wire."""
    emitted: list[dict[str, Any]] = []

    def _emit(payload: dict[str, Any]) -> None:
        emitted.append(payload)

    asker = IpcAsker(emit=_emit)
    rule = Rule("demo", None)

    task = asyncio.create_task(
        asker(tool=_tool(), args={"value": "x"}, rule_hint=rule),
    )
    for _ in range(50):
        if emitted:
            break
        await asyncio.sleep(0.01)

    assert asker.feed_response({
        "id": emitted[0]["id"],
        "choice": "always",
        "scope": "global",  # not in the Literal — must fall back
    })
    response = await asyncio.wait_for(task, timeout=1)
    assert response.scope == "session"


