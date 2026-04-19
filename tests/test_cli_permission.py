"""Tests for aura.cli.permission.make_cli_asker."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.cli.permission import make_cli_asker
from aura.core.hooks.permission import PermissionSession
from aura.tools.base import build_tool


class _P(BaseModel):
    msg: str


def _noop(msg: str) -> dict[str, Any]:
    return {}


def _tool() -> BaseTool:
    return build_tool(
        name="t", description="t", args_schema=_P, func=_noop, is_destructive=True,
    )


async def test_answer_y_allows_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    session = PermissionSession()
    asker = make_cli_asker(session)
    allowed = await asker(_tool(), {"msg": "x"})
    assert allowed is True
    assert session.allowlist == set()


async def test_answer_N_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "N")
    session = PermissionSession()
    asker = make_cli_asker(session)
    assert await asker(_tool(), {"msg": "x"}) is False


async def test_empty_answer_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    asker = make_cli_asker(PermissionSession())
    assert await asker(_tool(), {"msg": "x"}) is False


async def test_answer_a_allows_and_adds_to_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "a")
    session = PermissionSession()
    asker = make_cli_asker(session)
    assert await asker(_tool(), {"msg": "x"}) is True
    assert session.allowlist == {"t"}


async def test_eof_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_eof(prompt: str) -> str:
        raise EOFError()

    monkeypatch.setattr("builtins.input", _raise_eof)
    asker = make_cli_asker(PermissionSession())
    assert await asker(_tool(), {"msg": "x"}) is False
