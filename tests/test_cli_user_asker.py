"""Tests for cli._user_asker — answer passthrough + timeout/cancel mapping."""

from __future__ import annotations

import asyncio

import pytest

from aura.tools.ask_user import FormQuestionDict
from cli import _user_asker
from cli._user_asker import make_cli_user_asker


@pytest.mark.asyncio
async def test_user_asker_passes_through_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake(_q: list[FormQuestionDict]) -> dict[str, str]:
        return {"q": "a"}

    monkeypatch.setattr(_user_asker, "render_form", _fake)
    out = await make_cli_user_asker()([{"question": "q"}])
    assert out == {"q": "a"}


@pytest.mark.asyncio
async def test_user_asker_timeout_returns_blank_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _hang(_q: list[FormQuestionDict]) -> dict[str, str]:
        await asyncio.sleep(10)
        return {}

    monkeypatch.setattr(_user_asker, "render_form", _hang)
    out = await make_cli_user_asker(timeout=0.01)([{"question": "q"}])
    assert out == {"q": ""}
