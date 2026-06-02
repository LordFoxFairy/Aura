"""Tests for cli._permission_asker — translation between form answers and
:class:`AskerResponse`. Renderer is monkeypatched; only the mapping matters."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from pydantic import BaseModel

from aura.tools.ask_user import FormQuestionDict
from aura.tools.base import build_tool
from cli import _permission_asker
from cli._permission_asker import make_cli_asker


class _BashParams(BaseModel):
    command: str


class _WriteParams(BaseModel):
    path: str
    content: str = ""


class _GenericParams(BaseModel):
    arg: str = ""


def _bash_tool() -> Any:
    return build_tool(
        name="bash",
        description="bash",
        args_schema=_BashParams,
        func=lambda command: {"stdout": command, "stderr": "", "exit_code": 0},
        is_destructive=True,
        args_preview=lambda args: f"cmd: {args.get('command', '')}",
        rule_matcher=type("M", (), {"key": "command"}),
    )


def _write_tool() -> Any:
    return build_tool(
        name="write_file",
        description="write",
        args_schema=_WriteParams,
        func=lambda path, content="": {"path": path},
        is_destructive=True,
        rule_matcher=type("M", (), {"key": "path"}),
    )


def _generic_tool() -> Any:
    return build_tool(
        name="grep",
        description="grep",
        args_schema=_GenericParams,
        func=lambda arg="": {"matches": []},
    )


def _patch_render(
    monkeypatch: pytest.MonkeyPatch,
    answers: dict[str, str],
) -> list[list[FormQuestionDict]]:
    """Replace render_form with a stub returning ``answers``; record calls."""
    captured: list[list[FormQuestionDict]] = []

    async def _fake(questions: list[FormQuestionDict]) -> dict[str, str]:
        captured.append(list(questions))
        return answers

    monkeypatch.setattr(_permission_asker, "render_form", _fake)
    return captured


@pytest.mark.asyncio
async def test_bash_allow_once_maps_to_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_render(monkeypatch, {
        "Allow this tool call?": "Allow once",
        "Feedback (optional)": "",
    })
    asker = make_cli_asker()
    tool = _bash_tool()
    rule_hint: Any = None
    resp = await asker(tool=tool, args={"command": "ls"}, rule_hint=rule_hint)
    assert resp.choice == "accept"
    assert resp.feedback == ""
    assert captured[0][0]["header"] == "Allow bash?"


@pytest.mark.asyncio
async def test_bash_allow_command_installs_session_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render(monkeypatch, {
        "Allow this tool call?": "Allow always for this command",
        "Feedback (optional)": "",
    })
    asker = make_cli_asker()
    rule_hint: Any = None  # deliberately off-type to exercise path
    resp = await asker(tool=_bash_tool(), args={"command": "ls -la"}, rule_hint=rule_hint)
    assert resp.choice == "always"
    assert resp.scope == "session"
    assert resp.rule is not None and resp.rule.content == "ls -la"


@pytest.mark.asyncio
async def test_bash_allow_prefix_installs_project_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render(monkeypatch, {
        "Allow this tool call?": "Allow always for the prefix",
        "Feedback (optional)": "",
    })
    asker = make_cli_asker()
    rule_hint: Any = None  # deliberately off-type to exercise path
    resp = await asker(tool=_bash_tool(), args={"command": "git status"}, rule_hint=rule_hint)
    assert resp.choice == "always"
    assert resp.scope == "project"
    assert resp.rule is not None and resp.rule.content == "git"


@pytest.mark.asyncio
async def test_bash_deny_carries_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render(monkeypatch, {
        "Allow this tool call?": "Deny",
        "Feedback (optional)": "too risky",
    })
    asker = make_cli_asker()
    rule_hint: Any = None  # deliberately off-type to exercise path
    resp = await asker(tool=_bash_tool(), args={"command": "rm -rf /"}, rule_hint=rule_hint)
    assert resp.choice == "deny"
    assert resp.feedback == "too risky"


@pytest.mark.asyncio
async def test_write_allow_dir_installs_dir_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render(monkeypatch, {
        "Allow this tool call?": "Allow always for this dir",
        "Feedback (optional)": "",
    })
    asker = make_cli_asker()
    rule_hint: Any = None  # deliberately off-type to exercise path
    resp = await asker(
        tool=_write_tool(),
        args={"path": "src/foo/bar.py", "content": ""},
        rule_hint=rule_hint,
    )
    assert resp.choice == "always"
    assert resp.scope == "project"
    assert resp.rule is not None and resp.rule.content == "src/foo/*"


@pytest.mark.asyncio
async def test_write_allow_path_uses_derived_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_render(monkeypatch, {
        "Allow this tool call?": "Allow always for this path",
        "Feedback (optional)": "",
    })
    asker = make_cli_asker()
    rule_hint: Any = None  # deliberately off-type to exercise path
    resp = await asker(
        tool=_write_tool(),
        args={"path": "src/foo.py", "content": ""},
        rule_hint=rule_hint,
    )
    assert resp.choice == "always"
    assert resp.scope == "project"
    assert resp.rule is not None and resp.rule.content == "src/foo.py"
    assert captured[0][0]["header"] == "Allow write?"


@pytest.mark.asyncio
async def test_generic_allow_always_session_rule_when_no_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_render(monkeypatch, {
        "Allow this tool call?": "Allow always",
        "Feedback (optional)": "",
    })
    asker = make_cli_asker()
    rule_hint: Any = None  # deliberately off-type to exercise path
    resp = await asker(tool=_generic_tool(), args={"arg": ""}, rule_hint=rule_hint)
    assert resp.choice == "always"
    assert resp.scope == "session"
    assert resp.rule is not None and resp.rule.content is None
    assert captured[0][0]["header"] == "Allow tool?"


@pytest.mark.asyncio
async def test_cancel_maps_to_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    from cli.forms.widget import FormCancelled

    async def _cancel(_q: list[FormQuestionDict]) -> dict[str, str]:
        raise FormCancelled

    monkeypatch.setattr(_permission_asker, "render_form", _cancel)
    asker: Callable[..., Awaitable[Any]] = make_cli_asker()
    resp = await asker(
        tool=_generic_tool(), args={"arg": ""}, rule_hint=None,
    )
    assert resp.choice == "deny"
