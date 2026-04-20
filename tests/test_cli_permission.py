"""Tests for aura.cli.permission.make_cli_asker (Task 7 — AskerResponse return)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.cli.permission import make_cli_asker
from aura.core.hooks.permission import AskerResponse
from aura.core.permissions.rule import Rule
from aura.tools.base import build_tool


class _P(BaseModel):
    msg: str


def _noop(msg: str) -> dict[str, Any]:
    return {}


def _tool() -> BaseTool:
    return build_tool(
        name="t", description="t", args_schema=_P, func=_noop, is_destructive=True,
    )


_HINT = Rule(tool="t", content=None)


async def test_answer_y_returns_accept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    asker = make_cli_asker()
    resp = await asker(tool=_tool(), args={"msg": "x"}, rule_hint=_HINT)
    assert isinstance(resp, AskerResponse)
    assert resp.choice == "accept"
    assert resp.rule is None


async def test_answer_N_returns_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "N")
    asker = make_cli_asker()
    resp = await asker(tool=_tool(), args={"msg": "x"}, rule_hint=_HINT)
    assert resp.choice == "deny"


async def test_empty_answer_returns_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    asker = make_cli_asker()
    resp = await asker(tool=_tool(), args={"msg": "x"}, rule_hint=_HINT)
    assert resp.choice == "deny"


async def test_answer_a_returns_always_session_with_rule_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "a")
    asker = make_cli_asker()
    resp = await asker(tool=_tool(), args={"msg": "x"}, rule_hint=_HINT)
    assert resp.choice == "always"
    assert resp.scope == "session"
    assert resp.rule == _HINT


async def test_eof_returns_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_eof(prompt: str) -> str:
        raise EOFError()

    monkeypatch.setattr("builtins.input", _raise_eof)
    asker = make_cli_asker()
    resp = await asker(tool=_tool(), args={"msg": "x"}, rule_hint=_HINT)
    assert resp.choice == "deny"
